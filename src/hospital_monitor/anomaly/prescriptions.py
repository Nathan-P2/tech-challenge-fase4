"""Detecção de anomalias na evolução de prescrições.

Regras farmacológicas (base de conhecimento simplificada) + estatística sobre o
histórico do próprio paciente:

- dose diária acima do máximo / dose fora da faixa usual (z-score log-dose);
- mudança abrupta de dose (≥ 3× ou ≤ 1/3 da anterior — padrão de erro de vírgula);
- duplicidade terapêutica (dois fármacos ativos da mesma classe);
- interações graves entre fármacos ativos;
- escalonamento terapêutico inesperado (vasopressor, carbapenêmico...);
- instabilidade terapêutica (muitas alterações em 24 h).
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from ..alerts import Alert

# medicamento: classe, faixa usual por dose, unidade, máximo diário
DRUGS = {
    "dipirona": ("analgesico", (500, 1000), "mg", 4000),
    "paracetamol": ("analgesico", (500, 1000), "mg", 4000),
    "tramadol": ("opioide", (50, 100), "mg", 400),
    "morfina": ("opioide", (2, 10), "mg", 60),
    "enoxaparina": ("anticoagulante", (40, 80), "mg", 200),
    "heparina": ("anticoagulante", (5000, 5000), "UI", 40000),
    "varfarina": ("anticoagulante", (2.5, 10), "mg", 15),
    "insulina_regular": ("insulina", (2, 20), "UI", 100),
    "ceftriaxona": ("antibiotico", (1000, 2000), "mg", 4000),
    "meropenem": ("antibiotico", (1000, 2000), "mg", 6000),
    "vancomicina": ("antibiotico", (1000, 1500), "mg", 4000),
    "furosemida": ("diuretico", (20, 40), "mg", 400),
    "noradrenalina": ("vasopressor", (4, 16), "mg", 64),
    "ondansetrona": ("antiemetico", (4, 8), "mg", 32),
    "midazolam": ("benzodiazepinico", (1, 5), "mg", 50),
    "omeprazol": ("ibp", (20, 40), "mg", 80),
    "losartana": ("anti_hipertensivo", (50, 100), "mg", 100),
    "aas": ("antiagregante", (100, 100), "mg", 300),
    "metformina": ("antidiabetico_oral", (500, 850), "mg", 2550),
    "cloreto_potassio": ("eletrolito", (600, 1200), "mg", 6000),
    "espironolactona": ("diuretico_poupador_k", (25, 50), "mg", 200),
}

INTERACTIONS = {
    frozenset({"morfina", "midazolam"}): ("alta", "depressão respiratória (opioide + benzodiazepínico)"),
    frozenset({"varfarina", "aas"}): ("alta", "risco aumentado de sangramento"),
    frozenset({"tramadol", "ondansetrona"}): ("moderada", "redução do efeito analgésico / risco serotoninérgico"),
    frozenset({"espironolactona", "cloreto_potassio"}): ("alta", "risco de hipercalemia"),
    frozenset({"enoxaparina", "aas"}): ("moderada", "risco aumentado de sangramento"),
}

ESCALATION = {
    "vasopressor": ("alta", "início de vasopressor — sugere instabilidade hemodinâmica"),
    "meropenem": ("moderada", "escalonamento para carbapenêmico — falha terapêutica/infecção grave?"),
    "vancomicina": ("moderada", "cobertura para Gram-positivos resistentes adicionada"),
}

# antibióticos ficam de fora: associações (ex.: meropenem + vancomicina) são terapêuticas
DUPLICATE_CLASSES = {"anticoagulante", "opioide", "ibp", "benzodiazepinico"}


def analyze_prescriptions(orders: pd.DataFrame, change_limit_24h: int = 4) -> tuple[list[Alert], pd.DataFrame]:
    alerts: list[Alert] = []
    rows = []
    for pid, g in orders.sort_values("timestamp").groupby("patient_id"):
        active: dict[str, dict] = {}
        changes: list[pd.Timestamp] = []
        for o in g.to_dict("records"):
            ts = pd.Timestamp(o["timestamp"])
            tsi = ts.isoformat(timespec="seconds")
            drug = o["medicamento"]
            info = DRUGS.get(drug)
            flags = []
            prev = active.get(drug)
            if o["acao"] == "suspensao":
                active.pop(drug, None)
                changes.append(ts)
                rows.append({**o, "flags": "suspensao"})
                continue
            daily = o["dose"] * o["vezes_dia"]
            if info:
                cls, (lo, hi), unit, max_day = info
                if daily > max_day:
                    sev = "critica" if daily >= 2 * max_day else "alta"
                    flags.append("dose_acima_maxima")
                    alerts.append(Alert(pid, "prescricao", "dose_acima_maxima", sev,
                                        f"{drug}: {o['dose']:g} {o['unidade']} {o['vezes_dia']}x/dia = {daily:g} "
                                        f"{unit}/dia (máximo {max_day:g}).", tsi, min(1.0, daily / max_day / 3),
                                        {"medicamento": drug, "dose_diaria": daily, "maximo": max_day}))
                mid = math.sqrt(lo * hi)
                spread = max(math.log(hi / lo) / 2, 0.25)
                z = math.log(max(o["dose"], 1e-9) / mid) / spread
                if abs(z) > 3 and "dose_acima_maxima" not in flags:
                    flags.append("dose_atipica")
                    alerts.append(Alert(pid, "prescricao", "dose_atipica", "moderada",
                                        f"{drug}: dose {o['dose']:g} {o['unidade']} fora da faixa usual "
                                        f"({lo:g}–{hi:g}); z(log-dose) = {z:+.1f}.", tsi, min(1.0, abs(z) / 6),
                                        {"medicamento": drug, "z_log_dose": round(z, 2)}))
            if prev is not None:
                ratio = o["dose"] / prev["dose"] if prev["dose"] else float("inf")
                if ratio >= 3 or ratio <= 1 / 3:
                    flags.append("mudanca_abrupta_dose")
                    decimal = abs(math.log10(ratio)) > 0.95 and abs(math.log10(ratio) - round(math.log10(ratio))) < 0.05
                    alerts.append(Alert(pid, "prescricao", "mudanca_abrupta_dose", "alta",
                                        f"{drug}: dose mudou de {prev['dose']:g} para {o['dose']:g} {o['unidade']} "
                                        f"({ratio:.1f}×)" + (" — padrão de erro de vírgula (10×)." if decimal else "."),
                                        tsi, min(1.0, abs(math.log(ratio)) / 3),
                                        {"medicamento": drug, "dose_anterior": prev["dose"], "dose_nova": o["dose"],
                                         "razao": round(ratio, 2), "suspeita_erro_virgula": decimal}))
            if info and prev is None:
                cls = info[0]
                same = [d for d in active if DRUGS.get(d, ("",))[0] == cls and d != drug]
                if same and cls in DUPLICATE_CLASSES:
                    flags.append("duplicidade_terapeutica")
                    alerts.append(Alert(pid, "prescricao", "duplicidade_terapeutica", "alta",
                                        f"{drug} prescrito com {', '.join(same)} ativo(s) — duplicidade de {cls}.",
                                        tsi, 0.8, {"medicamentos": [drug, *same], "classe": cls}))
                for other in active:
                    inter = INTERACTIONS.get(frozenset({drug, other}))
                    if inter:
                        flags.append("interacao")
                        alerts.append(Alert(pid, "prescricao", "interacao_medicamentosa", inter[0],
                                            f"{drug} + {other}: {inter[1]}.", tsi, 0.7,
                                            {"medicamentos": [drug, other]}))
                esc = ESCALATION.get(drug) or ESCALATION.get(cls)
                if esc and o["acao"] == "inicio" and ts - pd.Timestamp(g["timestamp"].min()) > timedelta(hours=6):
                    flags.append("escalonamento")
                    alerts.append(Alert(pid, "prescricao", "escalonamento_terapeutico", esc[0],
                                        f"{drug}: {esc[1]}.", tsi, 0.6, {"medicamento": drug, "classe": cls}))
            if prev is not None or ts - pd.Timestamp(g["timestamp"].min()) > timedelta(hours=6):
                changes.append(ts)
            recent = [c for c in changes if ts - c <= timedelta(hours=24)]
            if len(recent) >= change_limit_24h and len(recent) == change_limit_24h:
                alerts.append(Alert(pid, "prescricao", "instabilidade_terapeutica", "moderada",
                                    f"{len(recent)} alterações de prescrição em 24 h — tratamento instável.",
                                    tsi, 0.5, {"alteracoes_24h": len(recent)}))
            active[drug] = o
            rows.append({**o, "dose_diaria": daily, "flags": ",".join(flags)})
    return alerts, pd.DataFrame(rows)
