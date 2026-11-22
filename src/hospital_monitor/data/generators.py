"""Geradores de dados clínicos sintéticos com gabarito (sinais vitais, prescrições e mobilidade).

Os dados reproduzem faixas fisiológicas, ritmo circadiano e ruído autorregressivo.
Eventos clínicos são injetados com início/fim conhecidos para permitir medir
recall, falsos alarmes e latência de detecção. Artefatos de sensor (oxímetro
desconectado) também são injetados — o sistema deve reconhecê-los e NÃO alertar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

VITALS = ["fc", "pas", "pad", "spo2", "fr", "temp"]
UNITS = {"fc": "bpm", "pas": "mmHg", "pad": "mmHg", "spo2": "%", "fr": "irpm", "temp": "°C"}

# deltas máximos por tipo de evento e rampa de instalação (min)
EVENT_PROFILES = {
    "taquicardia": ({"fc": 48, "fr": 4}, 2),
    "bradicardia": ({"fc": -34, "pas": -12, "pad": -6}, 3),
    "dessaturacao": ({"spo2": -10, "fr": 10, "fc": 24}, 40),
    "hipotensao": ({"pas": -40, "pad": -22, "fc": 20}, 5),
    "crise_hipertensiva": ({"pas": 62, "pad": 28, "fc": 8}, 15),
    "deterioracao_sepse": ({"fc": 38, "temp": 2.0, "fr": 11, "pas": -34, "pad": -18, "spo2": -3}, None),
    "febre": ({"temp": 1.8, "fc": 14}, 60),
    "artefato_sensor": ({"spo2": None}, 0),
}


@dataclass
class ClinicalEvent:
    patient_id: str
    modality: str
    type: str
    start: str
    end: str
    artifact: bool = False
    note: str = ""


def _ar1(n: int, sd: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate_vitals(patient_id: str, start: datetime, hours: float, events: list[dict], seed: int = 0,
                    baseline: dict | None = None) -> tuple[pd.DataFrame, list[ClinicalEvent]]:
    """Gera série minuto a minuto. `events`: dicts com type, start_h, dur_min (e opcional scale)."""
    rng = np.random.default_rng(seed)
    n = int(hours * 60)
    t = np.arange(n)
    ts = pd.date_range(start, periods=n, freq="min")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60
    b = {"fc": rng.normal(78, 7), "pas": rng.normal(122, 9), "spo2": min(99.0, rng.normal(97.2, 0.8)),
         "fr": rng.normal(16, 1.5), "temp": rng.normal(36.7, 0.15)}
    if baseline:
        b.update(baseline)
    circ = np.cos(2 * np.pi * (hour - 15) / 24)
    sig = {
        "fc": b["fc"] + 5 * circ + _ar1(n, 2.8, 0.92, rng),
        "pas": b["pas"] + 6 * circ + _ar1(n, 4.0, 0.9, rng),
        "spo2": b["spo2"] + _ar1(n, 0.6, 0.85, rng),
        "fr": b["fr"] + 1.0 * circ + _ar1(n, 1.0, 0.8, rng),
        "temp": b["temp"] + 0.3 * np.cos(2 * np.pi * (hour - 17) / 24) + _ar1(n, 0.06, 0.97, rng),
    }
    sig["pad"] = sig["pas"] * 0.62 + _ar1(n, 2.0, 0.8, rng)
    truth = []
    for ev in events:
        deltas, ramp = EVENT_PROFILES[ev["type"]]
        s = int(ev["start_h"] * 60)
        dur = int(ev["dur_min"])
        e = min(n, s + dur)
        scale = ev.get("scale", 1.0)
        if ev["type"] == "artefato_sensor":
            sig["spo2"][s:e] = rng.uniform(52, 68, e - s)
        else:
            prof = np.zeros(n)
            if ramp is None:  # deterioração lenta: rampa linear até o fim
                prof[s:e] = np.linspace(0, 1, e - s)
                prof[e:] = 1.0
            else:
                r = max(1, ramp)
                prof[s:e] = np.clip((t[s:e] - s) / r, 0, 1)
                rec = min(n, e + r)
                prof[e:rec] = np.linspace(1, 0, rec - e)
            for k, d in deltas.items():
                sig[k] = sig[k] + scale * d * prof
        end_idx = n - 1 if ramp is None else min(e, n) - 1  # deterioração persiste até o fim do registro
        truth.append(ClinicalEvent(patient_id, "sinais_vitais", ev["type"], ts[s].isoformat(),
                                   ts[end_idx].isoformat(), ev["type"] == "artefato_sensor", ev.get("note", "")))
    df = pd.DataFrame({"timestamp": ts, "patient_id": patient_id})
    df["fc"] = np.round(sig["fc"]).astype(int)
    df["pas"] = np.round(sig["pas"]).astype(int)
    df["pad"] = np.round(sig["pad"]).astype(int)
    df["spo2"] = np.clip(np.round(sig["spo2"]), 50, 100).astype(int)
    df["fr"] = np.clip(np.round(sig["fr"]), 6, 50).astype(int)
    df["temp"] = np.round(sig["temp"], 1)
    return df, truth


def random_vitals_cohort(n_patients: int, start: datetime, hours: float = 72, seed: int = 7) -> tuple[pd.DataFrame, list[ClinicalEvent]]:
    """Coorte de avaliação: cada paciente recebe 0–3 eventos aleatórios + artefatos."""
    rng = np.random.default_rng(seed)
    types = ["taquicardia", "bradicardia", "dessaturacao", "hipotensao", "crise_hipertensiva", "febre",
             "deterioracao_sepse"]
    durations = {"taquicardia": (15, 40), "bradicardia": (20, 60), "dessaturacao": (60, 180), "hipotensao": (30, 90),
                 "crise_hipertensiva": (45, 120), "febre": (240, 480), "deterioracao_sepse": (720, 1200)}
    frames, truth = [], []
    for i in range(n_patients):
        pid = f"C-{i + 1:03d}"
        evs = []
        used = []
        for _ in range(int(rng.integers(0, 4))):
            ty = str(rng.choice(types))
            dur = int(rng.integers(*durations[ty]))
            for _try in range(20):
                s_h = float(rng.uniform(26, hours - dur / 60 - 2))  # primeiras 24 h = linha de base
                if all(abs(s_h - u) > 8 for u in used):
                    break
            used.append(s_h)
            evs.append({"type": ty, "start_h": s_h, "dur_min": dur, "scale": float(rng.uniform(0.85, 1.15))})
            if ty == "deterioracao_sepse":
                break
        for _ in range(int(rng.integers(0, 3))):
            evs.append({"type": "artefato_sensor", "start_h": float(rng.uniform(2, hours - 1)),
                        "dur_min": int(rng.integers(1, 4))})
        df, tr = generate_vitals(pid, start, hours, evs, seed=1000 + i)
        frames.append(df)
        truth += tr
    return pd.concat(frames, ignore_index=True), truth


# ---------------------------------------------------------------------- mobilidade
def generate_mobility(patient_id: str, start: datetime, hours: float, events: list[dict], seed: int = 0,
                      activity_level: float = 1.0) -> tuple[pd.DataFrame, list[ClinicalEvent]]:
    """Índice de atividade (0–100) por minuto — sensor de leito/acelerômetro ou movimento do vídeo."""
    rng = np.random.default_rng(seed)
    n = int(hours * 60)
    ts = pd.date_range(start, periods=n, freq="min")
    hour = ts.hour.to_numpy()
    day = (hour >= 7) & (hour < 22)
    act = np.where(day, rng.gamma(2.0, 4.0, n), rng.gamma(1.2, 1.2, n)) * activity_level
    # reposicionamentos noturnos a cada ~2 h e caminhadas diurnas
    for i in range(0, n, 120):
        if not day[i]:
            act[i:i + 3] += rng.uniform(8, 15)
    for d in range(int(hours // 24) + 1):
        for h in (10, 16):
            s = d * 1440 + h * 60 + int(rng.integers(0, 30))
            if s < n:
                act[s:s + 25] += rng.uniform(25, 40) * activity_level
    truth = []
    for ev in events:
        s = int(ev["start_h"] * 60)
        dur = int(ev.get("dur_min", 60))
        e = min(n, s + dur)
        ty = ev["type"]
        if ty == "queda":
            act[s] = 100
            act[s + 1:e] = rng.uniform(0, 0.5, e - s - 1)
        elif ty == "imobilidade":
            act[s:e] = rng.uniform(0, 1.0, e - s)
        elif ty == "agitacao_noturna":
            act[s:e] = rng.gamma(4.0, 7.0, e - s)
        elif ty == "declinio_mobilidade":
            factor = np.ones(n)
            factor[s:e] = np.linspace(1, ev.get("final", 0.3), e - s)
            factor[e:] = ev.get("final", 0.3)
            act = act * factor
        truth.append(ClinicalEvent(patient_id, "mobilidade", ty, ts[s].isoformat(), ts[e - 1].isoformat(),
                                   note=ev.get("note", "")))
    df = pd.DataFrame({"timestamp": ts, "patient_id": patient_id, "atividade": np.clip(np.round(act, 1), 0, 100)})
    return df, truth


# ---------------------------------------------------------------------- prescrições
def prescription(patient_id: str, when: datetime, drug: str, dose: float, unit: str, freq: int, route: str,
                 action: str = "inicio", prescriber: str = "CRM-0000") -> dict:
    return {"patient_id": patient_id, "timestamp": when.isoformat(timespec="minutes"), "medicamento": drug,
            "dose": dose, "unidade": unit, "vezes_dia": freq, "via": route, "acao": action, "prescritor": prescriber}


def events_to_frame(events: list[ClinicalEvent]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in events])


def hours(start: datetime, h: float) -> datetime:
    return start + timedelta(hours=h)
