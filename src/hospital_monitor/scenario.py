"""Cenário integrado de demonstração: 6 pacientes internados por 72 h.

Cada paciente tem uma história clínica que se manifesta em várias modalidades,
com gabarito conhecido. O cenário é usado na demonstração ponta a ponta,
na fusão multimodal e no painel.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from .data.generators import generate_mobility, generate_vitals, prescription

START = datetime(2026, 11, 19, 0, 0)
HOURS = 72

PATIENTS = {
    "P-101": {"idade": 68, "leito": "UTI-03", "motivo": "Pós-operatório de artroplastia de joelho",
              "historia": "Evolui com insuficiência respiratória no 3º dia (dessaturação progressiva, fala cansada "
                          "e entrecortada, prescrição de opioide + benzodiazepínico)."},
    "P-102": {"idade": 74, "leito": "ENF-12", "motivo": "Fibrilação atrial anticoagulada",
              "historia": "Crise hipertensiva seguida de disartria e imobilidade — suspeita de AVC; "
                          "duplicidade de anticoagulantes na véspera."},
    "P-103": {"idade": 81, "leito": "ENF-07", "motivo": "Reabilitação pós-fratura de fêmur",
              "historia": "Queda durante a fisioterapia filmada, taquicardia reativa e agitação noturna (delirium)."},
    "P-104": {"idade": 59, "leito": "ENF-15", "motivo": "Pneumonia comunitária",
              "historia": "Deterioração lenta compatível com sepse: FC/Temp/FR sobem e PA cai ao longo de 16 h, "
                          "mobilidade cai e o antibiótico é escalonado."},
    "P-105": {"idade": 45, "leito": "ENF-02", "motivo": "Pós-operatório de colecistectomia",
              "historia": "Paciente estável (controle). Dois artefatos de oxímetro que não devem gerar alarme."},
    "P-106": {"idade": 63, "leito": "ENF-09", "motivo": "Diabetes descompensado",
              "historia": "Taquicardia paroxística breve e erro de prescrição de insulina (10 → 100 UI)."},
}

VITAL_EVENTS = {
    "P-101": [{"type": "dessaturacao", "start_h": 47.5, "dur_min": 300}],
    "P-102": [{"type": "crise_hipertensiva", "start_h": 29.0, "dur_min": 120}],
    "P-103": [{"type": "taquicardia", "start_h": 34.02, "dur_min": 25, "scale": 0.75}],
    "P-104": [{"type": "deterioracao_sepse", "start_h": 36.0, "dur_min": 960}],
    "P-105": [{"type": "artefato_sensor", "start_h": 20.0, "dur_min": 2},
              {"type": "artefato_sensor", "start_h": 44.3, "dur_min": 3}],
    "P-106": [{"type": "taquicardia", "start_h": 15.0, "dur_min": 25}],
}

MOBILITY_EVENTS = {
    "P-101": [{"type": "declinio_mobilidade", "start_h": 40, "dur_min": 600, "final": 0.35}],
    "P-102": [{"type": "imobilidade", "start_h": 31.0, "dur_min": 210}],
    "P-103": [{"type": "queda", "start_h": 34.0 + 30 / 3600, "dur_min": 20},
              {"type": "agitacao_noturna", "start_h": 49.0, "dur_min": 180}],
    "P-104": [{"type": "declinio_mobilidade", "start_h": 36, "dur_min": 900, "final": 0.3}],
    "P-105": [],
    "P-106": [],
}

# consultas gravadas: (paciente, condição do áudio sintético, hora)
AUDIO_SCHEDULE = [
    ("P-101", "baseline", 2.0), ("P-102", "baseline", 2.2), ("P-103", "baseline", 2.4), ("P-104", "baseline", 2.6),
    ("P-104", "controle", 24.0), ("P-102", "disartria", 30.0), ("P-103", "controle", 40.0),
    ("P-101", "fadiga_respiratoria", 49.0),
]

VIDEO_SCHEDULE = [("P-103", "sessao_fisioterapia.mp4", 34.0)]


def _orders() -> list[dict]:
    h = lambda x: START + timedelta(hours=x)  # noqa: E731
    o = []
    # P-101
    o += [prescription("P-101", h(1), "dipirona", 1000, "mg", 4, "IV"),
          prescription("P-101", h(1), "enoxaparina", 40, "mg", 1, "SC"),
          prescription("P-101", h(1), "omeprazol", 20, "mg", 1, "VO"),
          prescription("P-101", h(50.0), "furosemida", 40, "mg", 2, "IV"),
          prescription("P-101", h(50.5), "morfina", 4, "mg", 4, "IV"),
          prescription("P-101", h(50.6), "midazolam", 2, "mg", 3, "IV")]
    # P-102
    o += [prescription("P-102", h(1), "varfarina", 5, "mg", 1, "VO"),
          prescription("P-102", h(1), "losartana", 50, "mg", 1, "VO"),
          prescription("P-102", h(20.0), "enoxaparina", 80, "mg", 2, "SC"),
          prescription("P-102", h(21.0), "aas", 100, "mg", 1, "VO")]
    # P-103
    o += [prescription("P-103", h(1), "paracetamol", 750, "mg", 4, "VO"),
          prescription("P-103", h(1), "enoxaparina", 40, "mg", 1, "SC"),
          prescription("P-103", h(35.0), "tramadol", 50, "mg", 4, "IV"),
          prescription("P-103", h(35.1), "ondansetrona", 4, "mg", 3, "IV")]
    # P-104
    o += [prescription("P-104", h(1), "ceftriaxona", 2000, "mg", 1, "IV"),
          prescription("P-104", h(1), "dipirona", 1000, "mg", 4, "IV"),
          prescription("P-104", h(55.0), "ceftriaxona", 2000, "mg", 1, "IV", "suspensao"),
          prescription("P-104", h(55.0), "meropenem", 1000, "mg", 3, "IV"),
          prescription("P-104", h(55.2), "vancomicina", 1000, "mg", 2, "IV"),
          prescription("P-104", h(58.0), "noradrenalina", 8, "mg", 1, "IV")]
    # P-105 (controle: titulação normal)
    o += [prescription("P-105", h(1), "dipirona", 500, "mg", 4, "VO"),
          prescription("P-105", h(1), "omeprazol", 20, "mg", 1, "VO"),
          prescription("P-105", h(1), "enoxaparina", 40, "mg", 1, "SC"),
          prescription("P-105", h(30.0), "dipirona", 1000, "mg", 4, "VO", "ajuste")]
    # P-106
    o += [prescription("P-106", h(1), "insulina_regular", 10, "UI", 4, "SC"),
          prescription("P-106", h(1), "metformina", 850, "mg", 2, "VO"),
          prescription("P-106", h(40.0), "insulina_regular", 100, "UI", 4, "SC", "ajuste")]
    return o


# gabarito esperado de alertas de prescrição (tipo por paciente)
EXPECTED_PRESCRIPTION = {
    ("P-101", "interacao_medicamentosa"), ("P-102", "duplicidade_terapeutica"), ("P-102", "interacao_medicamentosa"),
    ("P-103", "interacao_medicamentosa"), ("P-104", "escalonamento_terapeutico"), ("P-106", "dose_acima_maxima"),
    ("P-106", "mudanca_abrupta_dose"), ("P-104", "instabilidade_terapeutica"),
}

EXPECTED_FUSION = {
    "P-101": "suspeita_insuficiencia_respiratoria", "P-102": "suspeita_avc", "P-103": "queda_com_repercussao",
    "P-104": "suspeita_sepse",
}


def build() -> dict:
    vitals, mobility, truth = [], [], []
    for k, pid in enumerate(PATIENTS):
        v, tv = generate_vitals(pid, START, HOURS, VITAL_EVENTS[pid], seed=200 + k)
        m, tm = generate_mobility(pid, START, HOURS, MOBILITY_EVENTS[pid], seed=300 + k)
        vitals.append(v)
        mobility.append(m)
        truth += tv + tm
    return {
        "inicio": START, "horas": HOURS, "pacientes": PATIENTS,
        "sinais_vitais": pd.concat(vitals, ignore_index=True),
        "mobilidade": pd.concat(mobility, ignore_index=True),
        "prescricoes": pd.DataFrame(_orders()),
        "gabarito": truth,
    }
