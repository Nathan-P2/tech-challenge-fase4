"""Detecção de anomalias em sinais vitais — processamento em fluxo (amostra a amostra).

Quatro detectores complementares rodam sobre cada amostra (1/min):

1. **NEWS2** (National Early Warning Score 2): limiares clínicos validados.
2. **Z-score robusto**: desvio em relação à linha de base do próprio paciente
   (mediana/MAD das últimas 6 h), capta mudanças súbitas ainda dentro da faixa
   "normal" populacional.
3. **CUSUM sazonal**: compara médias de 15 min com o mesmo horário do dia
   anterior (remove o ritmo circadiano) e acumula pequenos desvios persistentes —
   detecta deterioração lenta (ex.: sepse) antes de cruzar limiares absolutos.
4. **Isolation Forest** multivariado: combinações incomuns de sinais
   (ex.: FC alta + PA baixa + SpO2 em queda).

Um filtro de artefato descarta quedas instantâneas e breves de SpO2 sem
alteração concomitante de FC/FR (oxímetro deslocado), reduzindo fadiga de alarmes.
Episódios exigem persistência mínima antes de gerar alerta.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..alerts import Alert

SIGNALS = ["fc", "pas", "spo2", "fr", "temp"]
# desvio-padrão esperado da diferença entre médias de 15 min em dias consecutivos (paciente estável)
CUSUM_SD = {"fc": 4.0, "pas": 6.0, "spo2": 1.0, "fr": 1.5, "temp": 0.2}
LABELS = {"fc": "FC", "pas": "PAS", "pad": "PAD", "spo2": "SpO₂", "fr": "FR", "temp": "Temp"}


def news2(fc: float, pas: float, spo2: float, fr: float, temp: float) -> tuple[int, dict]:
    """NEWS2 (escala SpO2 1, paciente alerta, ar ambiente)."""
    def band(v, rules):
        for lo, hi, pts in rules:
            if lo <= v <= hi:
                return pts
        return 3

    parts = {
        "fr": band(fr, [(12, 20, 0), (9, 11, 1), (21, 24, 2)]),
        "spo2": band(spo2, [(96, 100, 0), (94, 95, 1), (92, 93, 2)]),
        "pas": band(pas, [(111, 219, 0), (101, 110, 1), (91, 100, 2)]),
        "fc": band(fc, [(51, 90, 0), (41, 50, 1), (91, 110, 1), (111, 130, 2)]),
        "temp": band(temp, [(36.1, 38.0, 0), (35.1, 36.0, 1), (38.1, 39.0, 1), (39.1, 45, 2)]),
    }
    if fr <= 8:
        parts["fr"] = 3
    return int(sum(parts.values())), parts


@dataclass
class VitalsConfig:
    baseline_minutes: int = 360
    min_baseline: int = 120
    z_threshold: float = 4.5
    z_persist: int = 8
    cusum_k: float = 0.5       # folga (em desvios-padrão)
    cusum_h: float = 6.0       # limiar de decisão
    cusum_step: int = 15       # minutos por passo do CUSUM
    news_persist: int = 5
    if_threshold: float = -0.12
    if_persist: int = 10
    if_every: int = 5
    artifact_spo2_drop: float = 12
    artifact_max_min: int = 4
    cooldown_min: int = 180


@dataclass
class _Episode:
    kind: str
    start: datetime
    count: int = 0
    alerted: bool = False
    peak: float = 0.0
    detail: dict = field(default_factory=dict)


class VitalSignsMonitor:
    """Monitor em tempo real de um paciente. Chame `update()` a cada nova amostra."""

    def __init__(self, patient_id: str, config: VitalsConfig | None = None, iforest: IsolationForest | None = None,
                 if_stats: tuple[np.ndarray, np.ndarray] | None = None, methods: set[str] | None = None):
        self.pid = patient_id
        self.cfg = config or VitalsConfig()
        self.buf: deque = deque(maxlen=self.cfg.baseline_minutes)
        self.iforest = iforest
        self.if_stats = if_stats
        self.methods = methods or {"news2", "zscore", "cusum", "iforest"}
        self.cusum_pos = {s: 0.0 for s in SIGNALS}
        self.cusum_neg = {s: 0.0 for s in SIGNALS}
        self.long: deque = deque(maxlen=24 * 60 + 60)
        self.episodes: dict[str, _Episode] = {}
        self.last_alert: dict[str, datetime] = {}
        self.pending_artifact: list[dict] = []
        self.history: list[dict] = []
        self._base_cache: tuple[dict, dict] | None = None
        self._long_cache: dict | None = None
        self._n = 0
        self._if_last = 0.0

    # ------------------------------------------------------------------ utilidades
    def _baseline(self) -> tuple[dict, dict] | None:
        """Mediana/MAD da janela recente, recalculada a cada 10 amostras (custo O(1) amortizado)."""
        if len(self.buf) < self.cfg.min_baseline:
            return None
        if self._base_cache is None or self._n % 10 == 0:
            arr = np.array(self.buf, float)
            med_v = np.median(arr, axis=0)
            mad_v = np.median(np.abs(arr - med_v), axis=0) * 1.4826
            floor = np.array([2, 3, 0.7, 1, 0.15])
            self._base_cache = (dict(zip(SIGNALS, med_v.tolist())), dict(zip(SIGNALS, np.maximum(mad_v, floor).tolist())))
        return self._base_cache

    def _long_median(self) -> dict | None:
        if len(self.long) < 12 * 60:
            return None
        if self._long_cache is None or self._n % 60 == 0:
            self._long_cache = dict(zip(SIGNALS, np.median(np.array(self.long, float), axis=0).tolist()))
        return self._long_cache

    def _is_artifact(self, sample: dict, base: tuple[dict, dict] | None) -> bool:
        """Queda abrupta de SpO2 sem resposta de FC/FR = provável sensor deslocado."""
        if base is None:
            return sample["spo2"] < 70
        med, mad = base
        drop = med["spo2"] - sample["spo2"]
        fc_ok = abs(sample["fc"] - med["fc"]) < 4 * mad["fc"] + 5
        fr_ok = abs(sample["fr"] - med["fr"]) < 4 * mad["fr"] + 3
        prev = self.history[-1]["spo2"] if self.history else sample["spo2"]
        abrupt = (prev - sample["spo2"]) >= self.cfg.artifact_spo2_drop or sample["spo2"] < 70
        in_run = bool(self.pending_artifact) and drop >= self.cfg.artifact_spo2_drop
        return (abrupt or in_run) and drop >= self.cfg.artifact_spo2_drop and fc_ok and fr_ok

    def _episode(self, key: str, active: bool, ts: datetime, persist: int, peak: float, detail: dict) -> _Episode | None:
        ep = self.episodes.get(key)
        if active:
            if ep is None:
                ep = self.episodes[key] = _Episode(key, ts)
            ep.count += 1
            ep.peak = max(ep.peak, peak)
            ep.detail = detail
            if ep.count >= persist and not ep.alerted:
                last = self.last_alert.get(key)
                if last is None or (ts - last).total_seconds() / 60 >= self.cfg.cooldown_min:
                    ep.alerted = True
                    self.last_alert[key] = ts
                    return ep
                ep.alerted = True
        elif ep is not None:
            del self.episodes[key]
        return None

    # ------------------------------------------------------------------ fluxo
    def update(self, sample: dict) -> list[Alert]:
        ts = pd.Timestamp(sample["timestamp"]).to_pydatetime()
        self._n += 1
        base = self._baseline()
        alerts: list[Alert] = []
        record = {"timestamp": ts, **{s: sample[s] for s in SIGNALS}, "artefato": False}

        if self._is_artifact(sample, base):
            self.pending_artifact.append(sample)
            record["artefato"] = True
            self.history.append(record)
            if len(self.pending_artifact) == 1:
                alerts.append(Alert(self.pid, "sinais_vitais", "artefato_sensor", "info",
                                    f"Queda abrupta de SpO₂ para {sample['spo2']}% sem alteração de FC/FR — "
                                    "provável sensor deslocado; alarme suprimido, verificar oxímetro.",
                                    ts.isoformat(timespec="seconds"), 0.1, {"spo2": sample["spo2"]}))
            if len(self.pending_artifact) <= self.cfg.artifact_max_min:
                return alerts
            # persistiu demais: deixa de ser tratado como artefato
            record["artefato"] = False
        else:
            self.pending_artifact.clear()

        score, parts = news2(sample["fc"], sample["pas"], sample["spo2"], sample["fr"], sample["temp"])
        record["news2"] = score
        tsi = ts.isoformat(timespec="seconds")

        # 1) NEWS2
        if "news2" in self.methods:
            red = [k for k, v in parts.items() if v == 3]
            level = "critica" if score >= 7 else "alta" if (score >= 5 or red) else None
            ep = self._episode("news2", level is not None, ts, self.cfg.news_persist, score, parts)
            if ep:
                vals = ", ".join(f"{LABELS[k]} {sample[k]}" for k, v in parts.items() if v > 0)
                alerts.append(Alert(self.pid, "sinais_vitais", "news2_elevado", level,
                                    f"NEWS2 = {score} por {ep.count} min ({vals}).", ep.start.isoformat(timespec="seconds"),
                                    min(1.0, score / 9), {"news2": score, "componentes": parts, "metodo": "NEWS2",
                                                          "detectado_em": tsi}))

        if base is not None:
            med, mad = base
            z = {s: (sample[s] - med[s]) / mad[s] for s in SIGNALS}
            record.update({f"z_{s}": round(z[s], 2) for s in SIGNALS})
            # 2) z-score robusto por sinal
            if "zscore" in self.methods:
                long_med = self._long_median()
                for s in SIGNALS:
                    # também precisa destoar da mediana de 24 h na mesma direção: evita alertar a
                    # *recuperação* após um evento (quando a janela curta ainda contém o evento)
                    lm_ok = long_med is None or (np.sign(sample[s] - long_med[s]) == np.sign(z[s])
                                                 and abs(sample[s] - long_med[s]) >= 2.5 * mad[s])
                    active = abs(z[s]) >= self.cfg.z_threshold and lm_ok
                    ep = self._episode(f"z_{s}", active, ts, self.cfg.z_persist, abs(z[s]), {"z": z[s]})
                    if ep:
                        direction = "elevação" if z[s] > 0 else "queda"
                        kind = _kind_from(s, z[s])
                        sev = "alta" if abs(z[s]) >= 6 or score >= 5 else "moderada"
                        alerts.append(Alert(self.pid, "sinais_vitais", kind, sev,
                                            f"{direction.capitalize()} súbita de {LABELS[s]}: {sample[s]} "
                                            f"(linha de base {med[s]:.0f}, z = {z[s]:+.1f}).",
                                            ep.start.isoformat(timespec="seconds"), min(1.0, abs(z[s]) / 8),
                                            {"sinal": s, "valor": sample[s], "linha_base": round(med[s], 1),
                                             "z": round(z[s], 2), "metodo": "z-score robusto", "detectado_em": tsi}))
            # 3) CUSUM sazonal (deriva lenta vs. mesmo horário do dia anterior)
            if "cusum" in self.methods and self._n % self.cfg.cusum_step == 0 and len(self.long) == self.long.maxlen:
                arr = np.array(self.long, float)
                cur = arr[-self.cfg.cusum_step:].mean(axis=0)
                ref = arr[:60].mean(axis=0)  # janela de 1 h centrada 24 h atrás
                drifting = []
                for k, s in enumerate(SIGNALS):
                    zr = (cur[k] - ref[k]) / CUSUM_SD[s]
                    self.cusum_pos[s] = max(0.0, self.cusum_pos[s] + zr - self.cfg.cusum_k)
                    self.cusum_neg[s] = max(0.0, self.cusum_neg[s] - zr - self.cfg.cusum_k)
                    # só conta deriva ainda ativa (passo atual na mesma direção do acumulador)
                    if self.cusum_pos[s] > self.cfg.cusum_h and zr > 1.0:
                        drifting.append((s, "↑"))
                    elif self.cusum_neg[s] > self.cfg.cusum_h and zr < -1.0:
                        drifting.append((s, "↓"))
                # deterioração = deriva em >= 2 sinais na direção clínica desfavorável
                bad = [(s, d) for s, d in drifting if (s, d) in {("fc", "↑"), ("fr", "↑"), ("temp", "↑"),
                                                                   ("pas", "↓"), ("spo2", "↓")}]
                ep = self._episode("cusum", len(bad) >= 2, ts, 1, len(bad), {"sinais": bad})
                if ep:
                    desc = ", ".join(f"{LABELS[s]}{d}" for s, d in bad)
                    kind = "deterioracao_febril" if ("temp", "↑") in bad else "deterioracao_progressiva"
                    alerts.append(Alert(self.pid, "sinais_vitais", kind, "alta",
                                        f"Tendência sustentada de piora (CUSUM) em {desc} vs. mesmo horário do dia anterior.",
                                        tsi, 0.7, {"sinais": [s for s, _ in bad], "metodo": "CUSUM",
                                                   "detectado_em": tsi}))
            # 4) Isolation Forest multivariado
            if "iforest" in self.methods and self.iforest is not None:
                # avalia a cada `if_every` amostras (o escore vale para o intervalo)
                if self._n % self.cfg.if_every == 0:
                    self._if_last = float(self.iforest.decision_function(self._if_features(sample, med))[0])
                sc = self._if_last
                record["if_score"] = round(sc, 4)
                ep = self._episode("iforest", sc < self.cfg.if_threshold, ts, self.cfg.if_persist, -sc,
                                   {"score": sc})
                if ep:
                    top = sorted(SIGNALS, key=lambda s: -abs(z[s]))[:3]
                    desc = ", ".join(f"{LABELS[s]} {sample[s]} (z {z[s]:+.1f})" for s in top)
                    alerts.append(Alert(self.pid, "sinais_vitais", "padrao_multivariado_anomalo", "moderada",
                                        f"Combinação incomum de sinais vitais (Isolation Forest): {desc}.",
                                        ep.start.isoformat(timespec="seconds"), min(1.0, -sc * 3),
                                        {"if_score": round(sc, 4), "metodo": "Isolation Forest", "detectado_em": tsi}))

        self.buf.append([sample[s] for s in SIGNALS])
        self.long.append([sample[s] for s in SIGNALS])
        self.history.append(record)
        return alerts

    def _if_features(self, sample: dict, med: dict) -> np.ndarray:
        raw = np.array([[sample[s] for s in SIGNALS] + [sample[s] - med[s] for s in SIGNALS]], float)
        if self.if_stats is not None:
            mu, sd = self.if_stats
            raw = (raw - mu) / sd
        return raw


def _kind_from(signal: str, z: float) -> str:
    return {("fc", True): "taquicardia", ("fc", False): "bradicardia", ("pas", True): "hipertensao",
            ("pas", False): "hipotensao", ("spo2", False): "dessaturacao", ("spo2", True): "spo2_elevada",
            ("fr", True): "taquipneia", ("fr", False): "bradipneia", ("temp", True): "febre",
            ("temp", False): "hipotermia"}[(signal, z > 0)]


def train_isolation_forest(normal: pd.DataFrame, seed: int = 42) -> tuple[IsolationForest, tuple[np.ndarray, np.ndarray]]:
    """Treina com períodos sem eventos (primeiras 24 h de cada paciente)."""
    feats = []
    for _pid, g in normal.groupby("patient_id"):
        med = g[SIGNALS].rolling(360, min_periods=60).median()
        x = np.hstack([g[SIGNALS].to_numpy(float), (g[SIGNALS] - med).to_numpy(float)])
        feats.append(x[~np.isnan(x).any(axis=1)])
    X = np.vstack(feats)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    model = IsolationForest(n_estimators=150, contamination="auto", random_state=seed)
    model.fit((X - mu) / sd)
    return model, (mu, sd)


def run_monitor(df: pd.DataFrame, iforest=None, if_stats=None, config: VitalsConfig | None = None,
                methods: set[str] | None = None) -> tuple[list[Alert], pd.DataFrame]:
    """Processa um DataFrame (vários pacientes) em ordem temporal, simulando o fluxo em tempo real."""
    monitors: dict[str, VitalSignsMonitor] = {}
    alerts: list[Alert] = []
    for row in df.sort_values("timestamp").to_dict("records"):
        pid = row["patient_id"]
        mon = monitors.get(pid)
        if mon is None:
            mon = monitors[pid] = VitalSignsMonitor(pid, config, iforest, if_stats, methods)
        alerts += mon.update(row)
    hist = pd.concat([pd.DataFrame(m.history).assign(patient_id=p) for p, m in monitors.items()], ignore_index=True)
    return alerts, hist
