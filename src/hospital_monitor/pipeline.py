"""Orquestra o fluxo multimodal ponta a ponta.

dados → análise por modalidade (vídeo, áudio, sinais vitais, prescrições,
mobilidade) → alertas → replay em ordem temporal (tempo real simulado) com
fusão multimodal incremental → painel, relatórios e métricas.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import scenario
from .alerts import AlertManager, ConsoleSink, JsonlSink, WebhookSink
from .anomaly.evaluation import evaluate_interval_events
from .anomaly.mobility import analyze_mobility
from .anomaly.prescriptions import analyze_prescriptions
from .anomaly.vitals import run_monitor, train_isolation_forest
from .audio.analyzer import analyze_consultation, run_dataset
from .data.generators import random_vitals_cohort
from .fusion import FusionEngine, risk_timeline
from .video.analyzer import analyze_video, events_to_alerts


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------- dados
def ensure_data(data_dir: Path) -> None:
    from .audio.synthetic import generate_consultations
    from .video.synthetic import generate_session

    vdir = data_dir / "video"
    if not (vdir / "sessao_fisioterapia.mp4").exists():
        log("Gerando vídeos sintéticos de fisioterapia (sessão com anomalias + controle)...")
        generate_session(vdir)
        generate_session(vdir, normal=True)
    adir = data_dir / "audio"
    if not (adir / "manifest.json").exists():
        log("Gerando gravações sintéticas de consultas (macOS `say`)...")
        generate_consultations(adir)


# ---------------------------------------------------------------------- vídeo
def run_video(data_dir: Path, out_dir: Path, backend_name: str) -> tuple[dict, list]:
    from .video.pose_backends import GroundTruthBackend, make_backend

    vdir = data_dir / "video"
    results, alerts = {}, []
    detector = None
    for stem in ("sessao_fisioterapia", "sessao_controle"):
        meta = json.loads((vdir / f"{stem}_meta.json").read_text())
        truth = np.load(vdir / f"{stem}_keypoints.npy")
        ref = cv2.imread(str(vdir / meta["referencia"]))
        if backend_name in {"gt", "ground-truth"}:
            backend = GroundTruthBackend(truth, noise_px=1.5, hide_lying=True)
        else:
            try:
                backend = make_backend(backend_name, **({"weights": str(_model("yolov8n-pose.pt")), "conf": 0.15}
                                                        if backend_name.startswith("yolo") else {}))
                if detector is None:
                    from ultralytics import YOLO

                    detector = YOLO(str(_model("yolov8n.pt")))
            except Exception as exc:  # sem ultralytics/pesos: usa keypoints de referência
                log(f"  backend {backend_name} indisponível ({exc}); usando ground-truth.")
                backend = GroundTruthBackend(truth, noise_px=1.5, hide_lying=True)
        log(f"Vídeo: analisando {stem}.mp4 com {backend.name}...")
        res = analyze_video(vdir / f"{stem}.mp4", out_dir / "video" / stem, backend, detector, meta["zonas"], ref,
                            truth_keypoints=truth, expected_events=meta["eventos_esperados"], annotate=True)
        res.pop("_df")
        events = res.pop("_events")
        results[stem] = res
        if stem == "sessao_fisioterapia":
            pid, _name, hour = scenario.VIDEO_SCHEDULE[0]
            alerts += events_to_alerts(events, pid, scenario.START + timedelta(hours=hour), f"{stem}.mp4")
    return results, alerts


def _model(name: str) -> Path:
    root = Path(__file__).resolve().parents[2] / "models"
    return root / name if (root / name).exists() else Path(name)


# ---------------------------------------------------------------------- áudio
def run_audio(data_dir: Path, out_dir: Path, transcriber: str | None) -> tuple[dict, list]:
    adir = data_dir / "audio"
    log("Áudio: avaliando 16 consultas (4 pacientes × linha de base, controle, fadiga/resp., disartria)...")
    dataset = run_dataset(adir, out_dir / "audio", scenario.START, transcriber)
    dataset.pop("_alerts")
    baselines = {r["paciente"]: r["biomarcadores"] for r in dataset["consultas"] if r["condicao"] == "baseline"}
    alerts, timeline = [], []
    log("Áudio: processando consultas agendadas do cenário integrado...")
    for pid, cond, hour in scenario.AUDIO_SCHEDULE:
        when = scenario.START + timedelta(hours=hour)
        res = analyze_consultation(adir / f"{pid}_{cond}.wav", pid, when,
                                   None if cond == "baseline" else baselines.get(pid), transcriber)
        al = res.pop("_alerts")
        res.pop("_series")
        if cond == "baseline":
            al = [a for a in al if a.type == "termos_criticos"]
        alerts += al
        timeline.append({"paciente": pid, "condicao": cond, "momento": when.isoformat(),
                         "transcricao": res["transcricao"]["texto"], "motor": res["transcricao"]["motor"],
                         "indicadores": {k: v["escore"] for k, v in res["indicadores"].items()},
                         "alertas": [a.type for a in al]})
    dataset["cenario"] = timeline
    (out_dir / "audio" / "audio_report.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2, default=float),
                                                         encoding="utf-8")
    return dataset, alerts


# ---------------------------------------------------------------------- sinais vitais
def run_vitals(sc: dict, out_dir: Path, compare_methods: bool, cohort_size: int = 30) -> tuple[dict, list, pd.DataFrame]:
    vitals = sc["sinais_vitais"]
    first = vitals[vitals["timestamp"] < pd.Timestamp(scenario.START) + pd.Timedelta(hours=24)]
    cohort, cohort_truth = random_vitals_cohort(cohort_size, scenario.START, 72, seed=7)
    cfirst = cohort[cohort["timestamp"] < pd.Timestamp(scenario.START) + pd.Timedelta(hours=24)]
    log("Sinais vitais: treinando Isolation Forest com as primeiras 24 h (sem eventos)...")
    ifm, stats = train_isolation_forest(pd.concat([first, cfirst]))
    log("Sinais vitais: monitor em fluxo no cenário (6 pacientes × 72 h, 1 amostra/min)...")
    alerts, hist = run_monitor(vitals, ifm, stats)
    truth = [e for e in sc["gabarito"] if e.modality == "sinais_vitais"]
    result = {"cenario": evaluate_interval_events(alerts, truth, patient_days=len(scenario.PATIENTS) * 3)}
    configs = {"NEWS2": {"news2"}, "Z-score robusto": {"zscore"}, "CUSUM sazonal": {"cusum"},
               "Isolation Forest": {"iforest"}, "Ensemble (todos)": {"news2", "zscore", "cusum", "iforest"}}
    if not compare_methods:
        configs = {"Ensemble (todos)": configs["Ensemble (todos)"]}
    comp = {}
    for name, methods in configs.items():
        log(f"Sinais vitais: coorte de avaliação ({cohort_size} pacientes) — {name}...")
        al, _ = run_monitor(cohort, ifm, stats, methods=methods)
        ev = evaluate_interval_events(al, cohort_truth, patient_days=cohort_size * 3)
        ev.pop("detalhe")
        comp[name] = ev
    result["coorte"] = {"pacientes": cohort_size, "eventos": comp[next(iter(comp))]["eventos"],
                        "artefatos": comp[next(iter(comp))]["artefatos"], "metodos": comp}
    hist.to_csv(out_dir / "vitals_monitor.csv", index=False)
    return result, alerts, hist


# ---------------------------------------------------------------------- pipeline
def run(out_dir: Path, data_dir: Path, pose_backend: str = "yolo", transcriber: str | None = None,
        compare_methods: bool = True, verbose_alerts: bool = True, physionet: bool = False) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_data(data_dir)
    sc = scenario.build()
    sc["sinais_vitais"].to_csv(data_dir / "sinais_vitais.csv", index=False)
    sc["mobilidade"].to_csv(data_dir / "mobilidade.csv", index=False)
    sc["prescricoes"].to_csv(data_dir / "prescricoes.csv", index=False)
    pd.DataFrame([asdict(e) for e in sc["gabarito"]]).to_csv(data_dir / "gabarito_eventos.csv", index=False)

    video, video_alerts = run_video(data_dir, out_dir, pose_backend)
    audio, audio_alerts = run_audio(data_dir, out_dir, transcriber)
    vitals, vital_alerts, vitals_hist = run_vitals(sc, out_dir, compare_methods)

    log("Prescrições: analisando evolução terapêutica...")
    rx_alerts, rx_table = analyze_prescriptions(sc["prescricoes"])
    rx_table.to_csv(out_dir / "prescricoes_analisadas.csv", index=False)
    found = {(a.patient_id, a.type) for a in rx_alerts}
    rx_eval = {"esperados": len(scenario.EXPECTED_PRESCRIPTION),
               "detectados": len(scenario.EXPECTED_PRESCRIPTION & found),
               "nao_esperados": sorted(f"{p}:{t}" for p, t in found - scenario.EXPECTED_PRESCRIPTION),
               "controle_P105_sem_alerta": not any(a.patient_id == "P-105" for a in rx_alerts)}

    log("Mobilidade: analisando padrões de movimentação...")
    mob_alerts = analyze_mobility(sc["mobilidade"])
    mob_truth = [e for e in sc["gabarito"] if e.modality == "mobilidade"]
    mob_eval = evaluate_interval_events(mob_alerts, mob_truth, patient_days=len(scenario.PATIENTS) * 3,
                                        pre_min=10, post_min=24 * 60)

    log("Replay em tempo real simulado + fusão multimodal...")
    all_alerts = sorted(video_alerts + audio_alerts + vital_alerts + rx_alerts + mob_alerts, key=lambda a: a.timestamp)
    jsonl = out_dir / "alertas.jsonl"
    jsonl.unlink(missing_ok=True)
    sinks = [JsonlSink(jsonl), WebhookSink()]
    if verbose_alerts:
        sinks.append(ConsoleSink("alta"))
    manager = AlertManager(sinks, cooldown_minutes=30)
    fusion = FusionEngine()
    for alert in all_alerts:
        if manager.publish(alert):
            for fused in fusion.add(alert):
                manager.publish(fused)
    published = manager.history
    risk = risk_timeline(published, list(scenario.PATIENTS), scenario.START, scenario.HOURS)
    risk.to_csv(out_dir / "risco_pacientes.csv", index=False)
    fused_by_patient = {a.patient_id: a.type for a in fusion.fused}
    fusion_eval = {"esperado": scenario.EXPECTED_FUSION,
                   "obtido": {p: [a.type for a in fusion.fused if a.patient_id == p] for p in scenario.PATIENTS},
                   "acertos": sum(any(a.patient_id == p and a.type == t for a in fusion.fused)
                                  for p, t in scenario.EXPECTED_FUSION.items()),
                   "controle_P105_sem_fusao": "P-105" not in fused_by_patient}

    # tempo entre o 1º alerta acionável do paciente e o alerta fundido (consolidação da evidência)
    lead = {}
    for f in fusion.fused:
        if scenario.EXPECTED_FUSION.get(f.patient_id) == f.type:
            first = min(pd.Timestamp(a.timestamp) for a in published
                        if a.patient_id == f.patient_id and a.source != "fusao" and a.severity not in {"info", "baixa"})
            lead[f.patient_id] = round((pd.Timestamp(f.timestamp) - first).total_seconds() / 60, 1)

    physio = None
    if physionet:
        from .data.physionet import validate_bidmc

        log("PhysioNet: validando falsos alarmes em registros reais (BIDMC)...")
        physio = validate_bidmc(out_dir / "physionet")

    patients = {}
    for pid, info in scenario.PATIENTS.items():
        pa = [a for a in published if a.patient_id == pid]
        last = risk[risk["patient_id"] == pid].iloc[-1]
        peak = risk[risk["patient_id"] == pid]["risco"].max()
        patients[pid] = {**info, "alertas": len(pa),
                         "alertas_por_origem": pd.Series([a.source for a in pa]).value_counts().to_dict() if pa else {},
                         "fusao": [a.type for a in pa if a.source == "fusao"],
                         "risco_pico": round(float(peak), 3), "risco_final": float(last["risco"]),
                         "nivel_final": last["nivel"]}

    summary = {
        "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "periodo": {"inicio": scenario.START.isoformat(), "horas": scenario.HOURS},
        "tempo_execucao_s": round(time.time() - t0, 1),
        "video": video, "audio": {"avaliacao": audio["avaliacao"], "cenario": audio["cenario"]},
        "sinais_vitais": vitals,
        "prescricoes": {"avaliacao": rx_eval, "alertas": len(rx_alerts)},
        "mobilidade": {k: v for k, v in mob_eval.items()},
        "fusao": {**fusion_eval, "alertas": [a.to_dict() for a in fusion.fused], "minutos_ate_fusao": lead},
        "alertas": manager.summary(),
        "pacientes": patients,
        "physionet": physio,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    from .reporting import build_figures, handoff_report

    log("Gerando figuras e relatório de passagem de plantão...")
    build_figures(out_dir, data_dir, sc, vitals_hist, risk, published, summary)
    (out_dir / "passagem_de_plantao.md").write_text(handoff_report(summary, published), encoding="utf-8")
    log(f"Concluído em {time.time() - t0:.0f}s. Saídas em {out_dir}/")
    return summary
