"""Padrões de movimentação do paciente durante a internação.

Entrada: índice de atividade por minuto (sensor de leito/acelerômetro ou energia
de movimento extraída do vídeo). Detectores:

- **queda**: pico de aceleração seguido de ausência de movimento;
- **imobilidade prolongada**: sem reposicionamento por ≥ 2 h (risco de lesão por pressão);
- **agitação noturna**: atividade noturna muito acima da linha de base do paciente
  (sinal precoce de delirium);
- **declínio de mobilidade**: atividade das últimas 24 h ≪ primeiras 24 h.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..alerts import Alert


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def analyze_mobility(df: pd.DataFrame, fall_peak: float = 80, immobile_min: int = 120, immobile_level: float = 3.0,
                     night_z: float = 4.0, decline_ratio: float = 0.6) -> list[Alert]:
    alerts: list[Alert] = []
    for pid, g in df.sort_values("timestamp").groupby("patient_id"):
        ts = pd.to_datetime(g["timestamp"]).reset_index(drop=True)
        act = g["atividade"].to_numpy(float)
        hour = ts.dt.hour.to_numpy()

        # queda: pico seguido de >= 10 min quase sem movimento
        fall_idx = []
        for i in np.flatnonzero(act >= fall_peak):
            after = act[i + 1:i + 11]
            if len(after) == 10 and after.mean() < 1.0:
                fall_idx.append(i)
                alerts.append(Alert(pid, "mobilidade", "queda", "critica",
                                    f"Pico de aceleração ({act[i]:.0f}) seguido de {len(after)} min sem movimento — "
                                    "possível queda do leito.", ts[i].isoformat(), 1.0,
                                    {"pico": float(act[i]), "atividade_pos": round(float(after.mean()), 2)}))

        # imobilidade prolongada (máximo da janela móvel abaixo do nível de reposicionamento)
        rolling_max = pd.Series(act).rolling(immobile_min, min_periods=immobile_min).max().to_numpy()
        still = rolling_max < immobile_level
        for s, e in _runs(still):
            start = max(0, s - immobile_min + 1)
            if any(start <= f <= e for f in fall_idx):
                continue  # já coberto pelo alerta de queda
            dur = e - start + 1
            alerts.append(Alert(pid, "mobilidade", "imobilidade_prolongada", "alta",
                                f"Sem reposicionamento há {dur // 60} h {dur % 60:02d} min — risco de lesão por pressão/"
                                "rebaixamento de consciência.", ts[s].isoformat(), 0.7,
                                {"inicio": ts[start].isoformat(), "duracao_min": int(dur)}))

        # agitação noturna vs. 1ª noite (linha de base do paciente)
        night = (hour >= 23) | (hour < 6)
        first_night = night & (ts < ts[0] + timedelta(hours=24)).to_numpy()
        if first_night.sum() > 120:
            mu, sd = act[first_night].mean(), act[first_night].std() + 1e-6
            hourly = pd.Series(act).rolling(60, min_periods=60).mean().to_numpy()
            z = (hourly - mu) / (sd / np.sqrt(10))  # média de 60 min (autocorrelação ~ 6 min)
            agitated = night & (z > night_z) & (hourly > 10) & ~first_night
            for s, e in _runs(agitated):
                if e - s >= 15:
                    alerts.append(Alert(pid, "mobilidade", "agitacao_noturna", "moderada",
                                        f"Atividade noturna {hourly[s]:.0f} vs. linha de base {mu:.1f} "
                                        f"(z = {z[s]:.1f}) — avaliar delirium/dor.", ts[s].isoformat(), 0.6,
                                        {"atividade_media_1h": round(float(hourly[s]), 1), "linha_base": round(float(mu), 2)}))

        # declínio de mobilidade: últimas 12 h vs. mesmas horas do 1º dia (respeita o ritmo circadiano)
        if len(act) > 1440 + 720:
            for i in range(1440 + 720, len(act), 60):
                d = (i - 720) // 1440
                ref = act[i - 720 - 1440 * d:i - 1440 * d].sum()
                ratio = act[i - 720:i].sum() / max(ref, 1e-6)
                if ratio < decline_ratio:
                    alerts.append(Alert(pid, "mobilidade", "declinio_mobilidade", "moderada",
                                        f"Atividade das últimas 12 h = {ratio:.0%} do mesmo período na admissão.",
                                        ts[i].isoformat(), round(1 - ratio, 2), {"razao_12h": round(float(ratio), 3)}))
                    break
    return alerts
