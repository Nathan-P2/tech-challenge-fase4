"""Validação com dados reais: PhysioNet/Computing in Cardiology Challenge 2012.

Base aberta (ODC-BY) com séries temporais das primeiras 48 h de internação em UTI
e o desfecho de óbito hospitalar. Não há rótulos de "anomalia", então a
validação mede se os detectores do projeto carregam **sinal prognóstico real**:
pacientes que evoluíram a óbito deveriam acumular mais anomalias e maior risco.

Fonte: https://physionet.org/content/challenge-2012/1.0.0/
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from ..anomaly.vitals import news2

BASE = "https://physionet.org/files/challenge-2012/1.0.0/"
VARS = {"HR": "fc", "SysABP": "pas_inv", "NISysABP": "pas_ni", "RespRate": "fr", "Temp": "temp", "SaO2": "spo2",
        "GCS": "gcs"}


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - URL fixa do PhysioNet
            dest.write_bytes(resp.read())
    return dest


def download(cache: Path, n: int = 800) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(_fetch(BASE + "Outcomes-a.txt", cache / "Outcomes-a.txt"))
    ids = outcomes["RecordID"].head(n).tolist()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: _fetch(f"{BASE}set-a/{i}.txt", cache / f"{i}.txt"), ids))
    return outcomes.head(n)


def hourly(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw["Parameter"].isin(VARS)]
    raw = raw[raw["Value"] > 0]
    raw["hora"] = raw["Time"].str.split(":").map(lambda x: int(x[0]))
    piv = raw.pivot_table(index="hora", columns="Parameter", values="Value", aggfunc="mean")
    piv = piv.reindex(range(0, 48)).rename(columns=VARS)
    for c in VARS.values():
        if c not in piv:
            piv[c] = np.nan
    piv["pas"] = piv["pas_inv"].fillna(piv["pas_ni"])
    return piv[["fc", "pas", "fr", "temp", "spo2", "gcs"]].ffill(limit=4)


def patient_scores(h: pd.DataFrame) -> dict:
    news, z_alerts = [], 0
    for _, r in h.iterrows():
        if r[["fc", "pas", "fr"]].isna().any():
            continue
        score, _ = news2(r["fc"], r["pas"], r["spo2"] if np.isfinite(r["spo2"]) else 97,
                         r["fr"], r["temp"] if np.isfinite(r["temp"]) else 37.0)
        if np.isfinite(r["gcs"]) and r["gcs"] <= 14:
            score += 3  # nível de consciência alterado (ACVPU ≠ A)
        news.append(score)
    base = h.iloc[:12]
    # só desvios na direção clínica desfavorável (FC↑, PAS↓, FR↑, Temp↑), como no CUSUM do monitor
    for col, floor, sign in (("fc", 3, 1), ("pas", 5, -1), ("fr", 1.5, 1), ("temp", 0.2, 1)):
        med = base[col].median()
        mad = max((base[col] - med).abs().median() * 1.4826, floor) if np.isfinite(med) else np.nan
        if np.isfinite(mad):
            z_alerts += int((sign * (h[col].iloc[12:] - med) / mad >= 4).sum())
    return {"news2_max": max(news) if news else np.nan, "news2_media": float(np.mean(news)) if news else np.nan,
            "horas_news2_ge5": int(sum(s >= 5 for s in news)), "anomalias_z": z_alerts}


def validate_challenge2012(out_dir: Path, n: int = 800, cache: Path | None = None) -> dict:
    cache = cache or Path("data/physionet/challenge2012")
    outcomes = download(cache, n)
    rows, mats = [], []
    for rid in outcomes["RecordID"]:
        h = hourly(cache / f"{rid}.txt")
        rows.append({"RecordID": rid, **patient_scores(h)})
        x = h[["fc", "pas", "fr", "temp"]].dropna()
        mats.append((rid, x))
    df = pd.DataFrame(rows).merge(outcomes, on="RecordID")
    # Isolation Forest não supervisionado nas primeiras 12 h de todos os pacientes
    train = np.vstack([x[x.index < 12].to_numpy() for _, x in mats if len(x[x.index < 12])])
    mu, sd = train.mean(0), train.std(0) + 1e-6
    iso = IsolationForest(n_estimators=200, random_state=42).fit((train - mu) / sd)
    if_scores = {}
    for rid, x in mats:
        later = x[x.index >= 12]
        if len(later):
            s = iso.decision_function((later.to_numpy() - mu) / sd)
            if_scores[rid] = float((s < -0.05).sum())
    df["if_horas_anomalas"] = df["RecordID"].map(if_scores)
    df["risco_combinado"] = (df["news2_max"].rank(pct=True) + df["anomalias_z"].rank(pct=True)
                             + df["if_horas_anomalas"].rank(pct=True)) / 3
    y = df["In-hospital_death"]
    result = {"pacientes": int(len(df)), "obitos": int(y.sum()), "taxa_obito": round(float(y.mean()), 3), "auc": {}}
    for col in ("news2_max", "horas_news2_ge5", "anomalias_z", "if_horas_anomalas", "risco_combinado", "SAPS-I"):
        m = df[col].notna() & (df[col] >= 0)
        result["auc"][col] = round(float(roc_auc_score(y[m], df.loc[m, col])), 3)
    surv, died = df[y == 0], df[y == 1]
    result["media_por_grupo"] = {
        col: {"sobreviventes": round(float(surv[col].mean()), 2), "obitos": round(float(died[col].mean()), 2)}
        for col in ("news2_max", "horas_news2_ge5", "anomalias_z", "if_horas_anomalas")}
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "challenge2012_escores.csv", index=False)
    (out_dir / "challenge2012_validacao.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


# compatibilidade com o nome usado pelo pipeline
validate_bidmc = validate_challenge2012
