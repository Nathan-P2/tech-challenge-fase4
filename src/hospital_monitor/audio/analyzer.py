"""Pipeline de análise de áudio de consultas.

áudio → (Azure Speech to Text | Whisper) → transcrição
      → biomarcadores vocais → indicadores (fadiga, resp., disartria)
      → (Azure Text Analytics for Health | NLP local) → termos críticos + sentimento
      → alertas (com escalonamento quando voz e conteúdo concordam)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..alerts import SEVERITIES, Alert
from ..cloud.azure_language import analyze_text
from ..cloud.azure_speech import transcribe
from .features import extract_features, load_audio
from .voice_indicators import score_indicators

# termos que confirmam cada indicador acústico
CONCORDANCE = {
    "fadiga": {"fadiga", "fraqueza", "fatigue"},
    "dificuldade_respiratoria": {"dispneia", "dyspnea", "falta de ar", "dor torácica"},
    "disartria": {"disartria", "dysarthria", "paralisia facial", "déficit motor", "disfagia"},
}


def _bump(sev: str, n: int = 1) -> str:
    return SEVERITIES[min(len(SEVERITIES) - 1, SEVERITIES.index(sev) + n)]


def analyze_consultation(path: str | Path, patient_id: str, when: datetime, baseline: dict | None = None,
                         transcriber: str | None = None, prefer_azure: bool = True) -> dict:
    path = Path(path)
    y = load_audio(path)
    feats, series = extract_features(y)
    features = feats.to_dict()
    indicators = score_indicators(features, baseline)
    tr = transcribe(path, transcriber)
    nlp = analyze_text(tr.text, prefer_azure=prefer_azure)

    ts = when.isoformat(timespec="seconds")
    terms = [t for t in nlp["termos"] if not t["negado"] and t["gravidade"] > 0]
    term_labels = {t["normalizado"].lower() for t in terms}
    alerts: list[Alert] = []
    for name, ind in indicators.items():
        if not ind["detectado"]:
            continue
        sev = "alta" if ind["escore"] >= 0.8 else "moderada"
        confirmed = sorted(term_labels & CONCORDANCE[name])
        if confirmed:
            sev = _bump(sev)
        top = sorted(ind["componentes"], key=lambda c: -c["contribuicao"])[:3]
        evid = ", ".join(f"{c['biomarcador']} ({c['valor']} vs {c['referencia']})" for c in top)
        msg = f"{ind['rotulo']} na voz (escore {ind['escore']:.2f}): {evid}."
        if confirmed:
            msg += f" Relato do paciente confirma: {', '.join(confirmed)}."
        alerts.append(Alert(patient_id, "audio", name, sev, msg, ts, ind["escore"],
                            {"arquivo": path.name, "componentes": ind["componentes"], "termos_concordantes": confirmed}))
    if terms:
        max_sev = max(t["gravidade"] for t in terms)
        sev = {3: "alta", 2: "moderada", 1: "baixa"}[max_sev]
        listed = sorted({t["normalizado"] for t in terms}, key=lambda s: s)
        alerts.append(Alert(patient_id, "audio", "termos_criticos", sev,
                            f"Termos clínicos relatados na consulta: {', '.join(listed)}.", ts, max_sev / 3,
                            {"arquivo": path.name, "termos": terms, "motor": nlp["motor"]}))
    sent = nlp["sentimento"]
    if sent["sentimento"] == "negativo" and sent["scores"]["negativo"] >= 0.6:
        alerts.append(Alert(patient_id, "audio", "sentimento_negativo", "baixa",
                            f"Sentimento predominantemente negativo ({sent['scores']['negativo']:.0%}).", ts,
                            sent["scores"]["negativo"], {"arquivo": path.name}))

    return {
        "arquivo": path.name, "paciente": patient_id, "momento": ts,
        "transcricao": {"texto": tr.text, "motor": tr.engine, "latencia_s": tr.latency_s, "erro": tr.error,
                        "segmentos": tr.segments},
        "biomarcadores": features, "indicadores": indicators, "nlp": nlp,
        "alertas": [a.to_dict() for a in alerts], "_alerts": alerts, "_series": series,
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    from .clinical_nlp import normalize

    r, h = normalize(reference).split(), normalize(hypothesis).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def run_dataset(audio_dir: Path, out_dir: Path, start: datetime, transcriber: str | None = None) -> dict:
    """Analisa todas as consultas do manifesto usando a gravação de admissão como linha de base."""
    manifest = json.loads((audio_dir / "manifest.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines: dict[str, dict] = {}
    results, alerts = [], []
    for k, item in enumerate(manifest):
        from datetime import timedelta

        when = start + timedelta(hours=6 * (k % 4), minutes=7 * (k // 4))
        is_base = item["condicao"] == "baseline"
        res = analyze_consultation(audio_dir / item["arquivo"], item["paciente"], when,
                                   None if is_base else baselines.get(item["paciente"]), transcriber)
        if is_base:
            baselines[item["paciente"]] = res["biomarcadores"]
            res["alertas"] = [a for a in res["alertas"] if a["type"] == "termos_criticos"]
            res["_alerts"] = [a for a in res["_alerts"] if a.type == "termos_criticos"]
        res["condicao"] = item["condicao"]
        res["rotulos"] = item["rotulos"]
        res["wer"] = round(word_error_rate(item["transcricao_referencia"], res["transcricao"]["texto"]), 3)
        res["transcricao_referencia"] = item["transcricao_referencia"]
        alerts += res.pop("_alerts")
        res.pop("_series")
        results.append(res)

    evaluation = evaluate(results)
    payload = {"consultas": results, "avaliacao": evaluation}
    (out_dir / "audio_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                                               encoding="utf-8")
    payload["_alerts"] = alerts
    return payload


def evaluate(results: list[dict]) -> dict:
    """Matriz de confusão por indicador nas consultas de acompanhamento (exclui linhas de base)."""
    out = {}
    followups = [r for r in results if r["condicao"] != "baseline"]
    for ind in ("fadiga", "dificuldade_respiratoria", "disartria"):
        tp = fp = fn = tn = 0
        for r in followups:
            pred = r["indicadores"][ind]["detectado"]
            true = bool(r["rotulos"][ind])
            tp += pred and true
            fp += pred and not true
            fn += (not pred) and true
            tn += (not pred) and (not true)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 1.0
        out[ind] = {"vp": tp, "fp": fp, "fn": fn, "vn": tn, "precisao": round(prec, 3), "recall": round(rec, 3),
                    "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}
    wers = [r["wer"] for r in results]
    out["wer_medio"] = round(sum(wers) / len(wers), 3) if wers else None
    by_cond: dict[str, list[float]] = {}
    for r in results:
        by_cond.setdefault(r["condicao"], []).append(r["wer"])
    out["wer_por_condicao"] = {c: round(sum(v) / len(v), 3) for c, v in by_cond.items()}
    # termos críticos recuperados da transcrição automática vs. da transcrição de referência
    from .clinical_nlp import find_critical_terms

    ref_terms = asr_terms = 0
    for r in results:
        ref = {t["normalizado"] for t in find_critical_terms(r.get("transcricao_referencia", "")) if not t["negado"]}
        got = {t["normalizado"] for t in r["nlp"]["termos"] if not t["negado"]}
        ref_terms += len(ref)
        asr_terms += len(ref & got)
    out["recall_termos_criticos_asr"] = round(asr_terms / ref_terms, 3) if ref_terms else None
    out["motor_transcricao"] = sorted({r["transcricao"]["motor"] for r in results})
    out["motor_nlp"] = sorted({r["nlp"]["motor"] for r in results})
    return out
