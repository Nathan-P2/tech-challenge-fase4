"""Avaliação de detectores contra eventos com gabarito (recall, falsos alarmes, latência)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..alerts import Alert


def evaluate_interval_events(alerts: list[Alert], truth: list, patient_days: float,
                             pre_min: float = 10, post_min: float = 60) -> dict:
    real = [e for e in truth if not e.artifact]
    artifacts = [e for e in truth if e.artifact]
    actionable = [a for a in alerts if a.severity != "info"]
    by_patient: dict[str, list[Alert]] = {}
    for a in actionable:
        by_patient.setdefault(a.patient_id, []).append(a)

    rows, matched = [], set()
    for ev in real:
        s = pd.Timestamp(ev.start) - timedelta(minutes=pre_min)
        e = pd.Timestamp(ev.end) + timedelta(minutes=post_min)
        hits = [a for a in by_patient.get(ev.patient_id, []) if s <= pd.Timestamp(a.timestamp) <= e]
        matched.update(id(a) for a in hits)
        first = min((pd.Timestamp(a.timestamp) for a in hits), default=None)
        rows.append({"paciente": ev.patient_id, "evento": ev.type, "detectado": bool(hits),
                     "latencia_min": round((first - pd.Timestamp(ev.start)).total_seconds() / 60, 1) if first is not None else None,
                     "alertas": len(hits)})
    false_alarms = [a for a in actionable if id(a) not in matched]
    suppressed = 0
    for art in artifacts:
        s, e = pd.Timestamp(art.start), pd.Timestamp(art.end) + timedelta(minutes=5)
        noisy = [a for a in by_patient.get(art.patient_id, []) if s <= pd.Timestamp(a.timestamp) <= e
                 and id(a) not in matched]
        suppressed += not noisy
    df = pd.DataFrame(rows)
    per_type = {}
    if len(df):
        for ty, g in df.groupby("evento"):
            lat = g["latencia_min"].dropna()
            per_type[ty] = {"n": int(len(g)), "recall": round(float(g["detectado"].mean()), 3),
                            "latencia_mediana_min": round(float(lat.median()), 1) if len(lat) else None}
    lat_all = df["latencia_min"].dropna() if len(df) else pd.Series(dtype=float)
    return {
        "eventos": int(len(real)),
        "recall": round(float(df["detectado"].mean()), 3) if len(df) else None,
        "latencia_mediana_min": round(float(lat_all.median()), 1) if len(lat_all) else None,
        "falsos_alarmes": len(false_alarms),
        "falsos_alarmes_por_paciente_dia": round(len(false_alarms) / max(patient_days, 1e-9), 3),
        "precisao_alertas": round(1 - len(false_alarms) / max(len(actionable), 1), 3),
        "artefatos": len(artifacts),
        "artefatos_suprimidos": suppressed,
        "por_tipo": per_type,
        "detalhe": rows,
    }


def summarize_numeric(values) -> dict:
    arr = np.asarray(list(values), float)
    if arr.size == 0:
        return {}
    return {"media": float(arr.mean()), "mediana": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())}
