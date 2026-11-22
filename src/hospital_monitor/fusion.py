"""Fusão multimodal (late fusion) de alertas em risco clínico do paciente.

Duas camadas complementares:

1. **Escore de risco contínuo** (noisy-OR com decaimento temporal): cada alerta
   contribui com um peso por gravidade que decai com meia-vida de 6 h. Sinais de
   modalidades diferentes no mesmo intervalo recebem bônus de concordância.
2. **Regras de correlação clínica**: padrões multimodais conhecidos (ex.: voz
   com dispneia + dessaturação) geram um alerta fundido, de maior gravidade e
   com a conduta recomendada — o que a equipe recebe primeiro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from .alerts import Alert

SEVERITY_WEIGHT = {"info": 0.0, "baixa": 0.05, "moderada": 0.15, "alta": 0.35, "critica": 0.6}
HALF_LIFE_H = 6.0
RISK_LEVELS = [(0.85, "crítico"), (0.6, "alto"), (0.3, "moderado"), (0.0, "baixo")]

VITAL_ANY = {"news2_elevado", "padrao_multivariado_anomalo", "deterioracao_progressiva", "deterioracao_febril",
             "taquicardia", "hipotensao",
             "hipertensao", "dessaturacao", "taquipneia", "febre", "bradicardia"}


@dataclass
class FusionRule:
    name: str
    title: str
    groups: list[set[tuple[str, str]]]  # todos os grupos devem ter ao menos um alerta
    window_h: float
    severity: str
    action: str


def _g(source: str, *types: str) -> set[tuple[str, str]]:
    return {(source, t) for t in types}


RULES = [
    FusionRule("suspeita_insuficiencia_respiratoria", "Suspeita de insuficiência respiratória",
               [_g("audio", "dificuldade_respiratoria", "fadiga"),
                _g("sinais_vitais", "dessaturacao", "taquipneia", "news2_elevado", "padrao_multivariado_anomalo",
                   "deterioracao_progressiva")],
               6, "critica", "Avaliar via aérea, gasometria e oxigenoterapia; rever sedação/opioides."),
    FusionRule("suspeita_avc", "Suspeita de AVC (protocolo AVC)",
               [_g("audio", "disartria"),
                _g("sinais_vitais", "hipertensao", "news2_elevado", "padrao_multivariado_anomalo")
                | _g("mobilidade", "imobilidade_prolongada") | _g("video", "assimetria_amplitude")],
               6, "critica", "Acionar protocolo AVC: escala NIHSS, TC de crânio imediata, glicemia; "
                             "revisar anticoagulação."),
    FusionRule("queda_com_repercussao", "Queda com repercussão clínica",
               [_g("video", "queda") | _g("mobilidade", "queda"),
                _g("sinais_vitais", "taquicardia", "hipotensao", "news2_elevado", "padrao_multivariado_anomalo")
                | _g("video", "imobilidade_pos_queda")],
               2, "critica", "Avaliação médica à beira-leito, investigar trauma craniano/fratura, "
                             "reavaliar risco de queda."),
    FusionRule("suspeita_sepse", "Suspeita de sepse / deterioração clínica",
               [_g("sinais_vitais", "deterioracao_febril", "febre"),
                _g("mobilidade", "declinio_mobilidade") | _g("prescricao", "escalonamento_terapeutico")],
               24, "critica", "Abrir protocolo sepse: lactato, hemoculturas, antibiótico na 1ª hora, "
                              "reavaliação hemodinâmica."),
    FusionRule("risco_medicamentoso_paciente_instavel", "Risco medicamentoso em paciente instável",
               [_g("prescricao", "interacao_medicamentosa", "dose_acima_maxima", "duplicidade_terapeutica",
                   "mudanca_abrupta_dose"),
                {("sinais_vitais", t) for t in VITAL_ANY} | _g("audio", "dificuldade_respiratoria", "disartria")],
               12, "alta", "Farmácia clínica e médico assistente devem revisar a prescrição imediatamente."),
]


def _ts(a: Alert) -> datetime:
    return datetime.fromisoformat(a.timestamp)


class FusionEngine:
    """Incremental: chame `add()` a cada alerta novo; retorna alertas fundidos gerados."""

    def __init__(self, rules: list[FusionRule] | None = None):
        self.rules = rules or RULES
        self.alerts: dict[str, list[Alert]] = {}
        self.fired: set[tuple[str, str]] = set()
        self.fused: list[Alert] = []

    def add(self, alert: Alert) -> list[Alert]:
        if alert.source == "fusao":
            return []
        self.alerts.setdefault(alert.patient_id, []).append(alert)
        out = []
        now = _ts(alert)
        for rule in self.rules:
            key = (alert.patient_id, rule.name)
            if key in self.fired:
                continue
            window = [a for a in self.alerts[alert.patient_id]
                      if now - timedelta(hours=rule.window_h) <= _ts(a) <= now and a.severity != "info"]
            best, support = [], []
            for group in rule.groups:
                hits = [a for a in window if (a.source, a.type) in group]
                if not hits:
                    break
                best.append(max(hits, key=lambda a: (a.rank, _ts(a))))
                support += [h for h in hits if h not in support]
            else:
                self.fired.add(key)
                # evidências: todos os alertas da janela que satisfazem algum grupo da regra
                support.sort(key=_ts)
                sources = sorted({a.source for a in support})
                evidence = [{"origem": a.source, "tipo": a.type, "gravidade": a.severity, "momento": a.timestamp,
                             "mensagem": a.message} for a in support]
                msg = (f"{rule.title}: evidências concordantes em {', '.join(sources)} — "
                       + "; ".join(f"[{a.source}] {a.message}" for a in best))
                fused = Alert(alert.patient_id, "fusao", rule.name, rule.severity, msg, now.isoformat(timespec="seconds"),
                              1.0, {"regra": rule.title, "conduta": rule.action, "evidencias": evidence,
                                    "modalidades": sources})
                self.fused.append(fused)
                out.append(fused)
        return out


def risk_timeline(alerts: list[Alert], patients: list[str], start: datetime, hours: int, step_min: int = 30) -> pd.DataFrame:
    """Escore de risco 0–1 por paciente ao longo do tempo (noisy-OR com decaimento)."""
    tau = HALF_LIFE_H / math.log(2)
    by_p: dict[str, list[Alert]] = {p: [] for p in patients}
    for a in alerts:
        if a.patient_id in by_p and a.source != "fusao":
            by_p[a.patient_id].append(a)
    rows = []
    for p in patients:
        evs = sorted(by_p[p], key=_ts)
        for k in range(0, hours * 60 + 1, step_min):
            t = start + timedelta(minutes=k)
            prod = 1.0
            recent_sources = set()
            for a in evs:
                dt = (t - _ts(a)).total_seconds() / 3600
                if dt < 0:
                    break
                w = SEVERITY_WEIGHT.get(a.severity, 0) * math.exp(-dt / tau)
                prod *= 1 - w
                if dt <= 6 and a.severity not in {"info", "baixa"}:
                    recent_sources.add(a.source)
            risk = 1 - prod
            bonus = 1 + 0.15 * max(0, len(recent_sources) - 1)
            risk = min(1.0, risk * bonus)
            rows.append({"timestamp": t, "patient_id": p, "risco": round(risk, 4),
                         "modalidades_6h": len(recent_sources), "nivel": risk_level(risk)})
    return pd.DataFrame(rows)


def risk_level(r: float) -> str:
    for thr, name in RISK_LEVELS:
        if r >= thr:
            return name
    return "baixo"
