"""Testes rápidos (sem rede, sem modelos pesados): rodam em poucos segundos."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from hospital_monitor.alerts import Alert, AlertManager
from hospital_monitor.anomaly.evaluation import evaluate_interval_events
from hospital_monitor.anomaly.mobility import analyze_mobility
from hospital_monitor.anomaly.prescriptions import analyze_prescriptions
from hospital_monitor.anomaly.vitals import VitalSignsMonitor, news2, run_monitor
from hospital_monitor.audio.clinical_nlp import find_critical_terms, sentiment
from hospital_monitor.audio.voice_indicators import POPULATION_REFERENCE, score_indicators
from hospital_monitor.data.generators import generate_mobility, generate_vitals, prescription
from hospital_monitor.fusion import FusionEngine, risk_timeline
from hospital_monitor.video.analyzer import VideoAnalyzer, evaluate_events
from hospital_monitor.video.keypoints import angle_between
from hospital_monitor.video.synthetic import SessionScript, pose_at

T0 = datetime(2026, 11, 19)


# ---------------------------------------------------------------- vídeo
def test_angle_between_right_angle():
    assert angle_between(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0, 1.0])) == pytest.approx(90)


def _pose_df(normal: bool) -> pd.DataFrame:
    rows = []
    for i in range(int(38 * 25)):
        t = i / 25
        kp, _ = pose_at(t, normal)
        span = kp.max(0) - kp.min(0)
        lying = span[0] > 1.3 * span[1]
        row = {"frame": i, "t": t, "motion": 0.0 if t > 31.5 and not normal else 0.3, "pose": not lying,
               "blob_aspect": span[0] / span[1]}
        if not lying:
            row.update(VideoAnalyzer._pose_features(kp, np.ones(17)))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["fps"] = 25
    return df


def test_video_detectors_find_injected_events():
    events, summary = VideoAnalyzer(None).detect(_pose_df(False))
    ev = evaluate_events(events, SessionScript().events)
    assert ev["recall"] == 1.0
    assert len(summary["repeticoes_abducao"]) == 6


def test_video_control_session_has_no_events():
    events, _ = VideoAnalyzer(None).detect(_pose_df(True))
    assert events == []


# ---------------------------------------------------------------- áudio
def test_negation_is_respected():
    terms = {t["normalizado"]: t["negado"] for t in find_critical_terms("Nega febre, mas sente falta de ar.")}
    assert terms["febre"] is True
    assert terms["dispneia"] is False


def test_sentiment_negative_for_symptoms():
    assert sentiment("Estou muito cansado, com dor e medo de piorar.")["sentimento"] == "negativo"


def test_voice_indicators_relative_to_baseline():
    base = dict(POPULATION_REFERENCE)
    tired = dict(base, intensidade_db=base["intensidade_db"] - 9, taxa_fala_sil_s=base["taxa_fala_sil_s"] * 0.65,
                 pausas_por_min=base["pausas_por_min"] * 1.7, f0_media_hz=base["f0_media_hz"] * 0.88)
    assert score_indicators(tired, base)["fadiga"]["detectado"]
    assert not score_indicators(base, base)["fadiga"]["detectado"]


# ---------------------------------------------------------------- sinais vitais
def test_news2_scores():
    assert news2(75, 125, 98, 16, 36.8)[0] == 0
    assert news2(135, 88, 90, 26, 39.5)[0] >= 7


def test_vitals_monitor_detects_tachycardia_and_ignores_artifact():
    df, truth = generate_vitals("X", T0, 30, [{"type": "taquicardia", "start_h": 20, "dur_min": 25},
                                              {"type": "artefato_sensor", "start_h": 10, "dur_min": 2}], seed=1)
    alerts, _ = run_monitor(df, methods={"news2", "zscore"})
    ev = evaluate_interval_events(alerts, truth, patient_days=30 / 24)
    assert ev["recall"] == 1.0
    assert ev["artefatos_suprimidos"] == 1
    assert ev["falsos_alarmes"] == 0


def test_monitor_is_incremental():
    mon = VitalSignsMonitor("Y", methods={"news2"})
    sample = {"timestamp": T0, "fc": 140, "pas": 85, "spo2": 88, "fr": 30, "temp": 39.5}
    out = []
    for k in range(6):
        out += mon.update({**sample, "timestamp": T0 + timedelta(minutes=k)})
    assert any(a.type == "news2_elevado" and a.severity == "critica" for a in out)


# ---------------------------------------------------------------- prescrições e mobilidade
def test_prescription_decimal_error_and_duplicity():
    orders = pd.DataFrame([
        prescription("A", T0, "insulina_regular", 10, "UI", 4, "SC"),
        prescription("A", T0 + timedelta(hours=10), "insulina_regular", 100, "UI", 4, "SC", "ajuste"),
        prescription("B", T0, "varfarina", 5, "mg", 1, "VO"),
        prescription("B", T0 + timedelta(hours=8), "enoxaparina", 80, "mg", 2, "SC"),
    ])
    types = {(a.patient_id, a.type) for a in analyze_prescriptions(orders)[0]}
    assert ("A", "dose_acima_maxima") in types and ("A", "mudanca_abrupta_dose") in types
    assert ("B", "duplicidade_terapeutica") in types


def test_mobility_fall_detected():
    df, _ = generate_mobility("M", T0, 30, [{"type": "queda", "start_h": 12, "dur_min": 20}], seed=3)
    assert any(a.type == "queda" for a in analyze_mobility(df))


# ---------------------------------------------------------------- alertas e fusão
def _alert(src, typ, sev, minutes, pid="P"):
    return Alert(pid, src, typ, sev, typ, (T0 + timedelta(minutes=minutes)).isoformat(timespec="seconds"))


def test_alert_manager_deduplicates_within_cooldown():
    mgr = AlertManager(cooldown_minutes=30)
    assert mgr.publish(_alert("sinais_vitais", "taquicardia", "alta", 0))
    assert not mgr.publish(_alert("sinais_vitais", "taquicardia", "alta", 10))
    assert mgr.publish(_alert("sinais_vitais", "taquicardia", "critica", 12))  # escalonou


def test_fusion_rule_requires_two_modalities():
    eng = FusionEngine()
    assert eng.add(_alert("audio", "dificuldade_respiratoria", "alta", 0)) == []
    fused = eng.add(_alert("sinais_vitais", "dessaturacao", "alta", 60))
    assert fused and fused[0].type == "suspeita_insuficiencia_respiratoria"
    assert fused[0].severity == "critica"


def test_risk_timeline_bounded():
    alerts = [_alert("audio", "fadiga", "critica", 0), _alert("sinais_vitais", "taquicardia", "alta", 5)]
    r = risk_timeline(alerts, ["P"], T0, 12)
    assert r["risco"].between(0, 1).all()
    assert r["risco"].iloc[1] > r["risco"].iloc[-1]  # decai com o tempo
