"""Indicadores vocais de fadiga, dificuldade respiratória e disartria.

Cada indicador combina desvios de biomarcadores em relação à **linha de base do
próprio paciente** (gravação da admissão). Sem linha de base, usa-se uma
referência populacional aproximada. Cada componente é normalizado para [0, 1]
entre um limiar de início e um de saturação, o que mantém o escore explicável:
o relatório mostra quanto cada biomarcador contribuiu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Referência populacional aproximada (adultos, fala lida/espontânea em pt-BR)
POPULATION_REFERENCE = {
    "taxa_fala_sil_s": 4.2, "taxa_articulacao_sil_s": 4.7, "razao_pausas": 0.10, "pausas_por_min": 13.0,
    "pausa_media_s": 0.40, "frase_media_s": 3.2, "f0_desvio_st": 2.7, "hnr_db": 8.0, "intensidade_db": -20.0,
    "energia_alta_freq": 0.10, "respiracoes_por_min": 0.0, "f0_media_hz": 160.0,
}

THRESHOLD = 0.5


@dataclass
class Component:
    name: str
    weight: float
    start: float
    full: float
    kind: str  # rel_drop | rel_rise | abs_drop | abs_rise | st_drop | abs_value


INDICATORS: dict[str, list[tuple[str, Component]]] = {
    "fadiga": [
        ("intensidade_db", Component("queda de intensidade (dB)", 0.30, 3.0, 8.0, "abs_drop")),
        ("taxa_fala_sil_s", Component("lentificação da fala", 0.30, 0.10, 0.30, "rel_drop")),
        ("f0_media_hz", Component("F0 mais grave (semitons)", 0.15, 0.5, 2.0, "st_drop")),
        ("pausas_por_min", Component("mais pausas por minuto", 0.25, 0.15, 0.60, "rel_rise")),
    ],
    "dificuldade_respiratoria": [
        ("pausas_por_min", Component("pausas frequentes", 0.25, 0.15, 0.60, "rel_rise")),
        ("frase_media_s", Component("frases curtas", 0.25, 0.10, 0.40, "rel_drop")),
        ("respiracoes_por_min", Component("inspirações audíveis/min", 0.25, 0.5, 3.0, "abs_value")),
        ("razao_pausas", Component("tempo em pausa", 0.25, 0.25, 1.00, "rel_rise")),
    ],
    "disartria": [
        ("f0_desvio_st", Component("instabilidade de F0", 0.40, 0.10, 0.40, "rel_rise")),
        ("taxa_articulacao_sil_s", Component("articulação lenta", 0.20, 0.08, 0.25, "rel_drop")),
        ("hnr_db", Component("voz mais ruidosa (HNR)", 0.20, 0.5, 2.5, "abs_drop")),
        ("energia_alta_freq", Component("perda de nitidez articulatória", 0.20, 0.10, 0.40, "rel_drop")),
    ],
}

# marcador obrigatório para cada indicador (evita confundir quadros com sobreposição acústica)
GATES = {"disartria": ("f0_desvio_st", 0.3), "fadiga": ("intensidade_db", 0.2)}

LABELS = {"fadiga": "Fadiga / cansaço vocal", "dificuldade_respiratoria": "Dificuldade respiratória",
          "disartria": "Disartria (fala arrastada/instável)"}


def _deviation(kind: str, value: float, ref: float) -> float:
    if kind == "rel_drop":
        return (ref - value) / ref if ref else 0.0
    if kind == "rel_rise":
        return (value - ref) / ref if ref else 0.0
    if kind == "abs_drop":
        return ref - value
    if kind == "abs_rise":
        return value - ref
    if kind == "st_drop":
        return 12 * math.log2(ref / value) if value > 0 and ref > 0 else 0.0
    if kind == "abs_value":
        return value
    raise ValueError(kind)


def score_indicators(features: dict, baseline: dict | None = None) -> dict:
    ref = baseline or POPULATION_REFERENCE
    out = {}
    for name, comps in INDICATORS.items():
        total, details = 0.0, []
        comp_scores = {}
        for feat, c in comps:
            dev = _deviation(c.kind, float(features[feat]), float(ref.get(feat, POPULATION_REFERENCE[feat])))
            sc = min(1.0, max(0.0, (dev - c.start) / (c.full - c.start)))
            comp_scores[feat] = sc
            total += c.weight * sc
            details.append({"biomarcador": c.name, "feature": feat, "valor": round(float(features[feat]), 3),
                            "referencia": round(float(ref.get(feat, POPULATION_REFERENCE[feat])), 3),
                            "desvio": round(dev, 3), "contribuicao": round(c.weight * sc, 3)})
        gate = GATES.get(name)
        gated = gate is not None and comp_scores.get(gate[0], 0) < gate[1]
        out[name] = {"rotulo": LABELS[name], "escore": round(total, 3),
                     "detectado": bool(total >= THRESHOLD and not gated), "componentes": details,
                     "referencia": "linha de base do paciente" if baseline else "referência populacional"}
    return out
