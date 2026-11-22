"""NLP clínico local (fallback do Azure Text Analytics / Text Analytics for Health).

Detecta termos críticos em português com tratamento de negação ("nega dor",
"sem febre", "não tenho falta de ar") e estima o sentimento com um léxico.
É usado quando as credenciais do Azure não estão configuradas e também como
segunda opinião determinística nos testes.
"""

from __future__ import annotations

import re
import unicodedata

# termo -> (categoria, gravidade 1-3, rótulo normalizado)
CRITICAL_TERMS: dict[str, tuple[str, int, str]] = {
    "falta de ar": ("SintomaOuSinal", 3, "dispneia"),
    "dificuldade para respirar": ("SintomaOuSinal", 3, "dispneia"),
    "dispneia": ("SintomaOuSinal", 3, "dispneia"),
    "dor no peito": ("SintomaOuSinal", 3, "dor torácica"),
    "peito que aperta": ("SintomaOuSinal", 3, "dor torácica"),
    "aperto no peito": ("SintomaOuSinal", 3, "dor torácica"),
    "coracao fica acelerado": ("SintomaOuSinal", 2, "palpitação"),
    "coracao acelerado": ("SintomaOuSinal", 2, "palpitação"),
    "palpitacao": ("SintomaOuSinal", 2, "palpitação"),
    "fala esta enrolada": ("SintomaOuSinal", 3, "disartria"),
    "fala enrolada": ("SintomaOuSinal", 3, "disartria"),
    "boca torta": ("SintomaOuSinal", 3, "paralisia facial"),
    "braco direito fraco": ("SintomaOuSinal", 3, "déficit motor"),
    "braco esquerdo fraco": ("SintomaOuSinal", 3, "déficit motor"),
    "fraqueza": ("SintomaOuSinal", 2, "fraqueza"),
    "formigando": ("SintomaOuSinal", 2, "parestesia"),
    "formigamento": ("SintomaOuSinal", 2, "parestesia"),
    "visao embacada": ("SintomaOuSinal", 2, "alteração visual"),
    "dificuldade para engolir": ("SintomaOuSinal", 2, "disfagia"),
    "tonto": ("SintomaOuSinal", 2, "tontura"),
    "tontura": ("SintomaOuSinal", 2, "tontura"),
    "desmaio": ("SintomaOuSinal", 3, "síncope"),
    "desmaiei": ("SintomaOuSinal", 3, "síncope"),
    "confuso": ("SintomaOuSinal", 2, "confusão mental"),
    "muito cansado": ("SintomaOuSinal", 2, "fadiga"),
    "cansaco": ("SintomaOuSinal", 1, "fadiga"),
    "febre": ("SintomaOuSinal", 2, "febre"),
    "sangramento": ("SintomaOuSinal", 3, "sangramento"),
    "vomito": ("SintomaOuSinal", 2, "vômito"),
    "dor": ("SintomaOuSinal", 1, "dor"),
    "nao consegui dormir": ("SintomaOuSinal", 1, "insônia"),
    "medo de piorar": ("EstadoEmocional", 1, "ansiedade"),
    "medo": ("EstadoEmocional", 1, "ansiedade"),
    "dipirona": ("NomeMedicamento", 0, "dipirona"),
    "morfina": ("NomeMedicamento", 0, "morfina"),
    "heparina": ("NomeMedicamento", 0, "heparina"),
    "insulina": ("NomeMedicamento", 0, "insulina"),
}

NEGATIONS = {"nao", "nega", "sem", "nunca", "nenhum", "nenhuma", "ausencia"}
CLAUSE_BREAKS = {"|", "mas", "porem", "entretanto", "contudo", "embora"}

POSITIVE = {"bem", "melhor", "otimo", "animado", "confiante", "tranquilo", "feliz", "diminuiu", "consegui",
            "melhorou", "alivio", "boa", "bom", "sem dificuldade"}
NEGATIVE = {"cansado", "falta", "dor", "medo", "piorar", "aperta", "acelerado", "fraco", "tonto", "enrolada",
            "dificuldade", "embacada", "formigando", "ruim", "pior", "angustia", "nao consegui"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[,.;:!?]+", " | ", text)  # fronteira de oração (limita o escopo da negação)
    return re.sub(r"[^a-z0-9| ]+", " ", text)


def find_critical_terms(text: str) -> list[dict]:
    norm = normalize(text)
    tokens = norm.split()
    joined = " " + " ".join(tokens) + " "
    found, used = [], []
    for term in sorted(CRITICAL_TERMS, key=len, reverse=True):
        for m in re.finditer(rf" {re.escape(term)} ", joined):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in used):
                continue  # já coberto por termo mais longo
            used.append(span)
            prefix = joined[: m.start()].split()
            for k in range(len(prefix) - 1, -1, -1):  # escopo termina na pontuação ou em "mas/porém"
                if prefix[k] in CLAUSE_BREAKS:
                    prefix = prefix[k + 1:]
                    break
            negated = any(t in NEGATIONS for t in prefix[-4:]) and term != "nao consegui dormir"
            cat, sev, label = CRITICAL_TERMS[term]
            found.append({"texto": term, "categoria": cat, "normalizado": label, "gravidade": sev,
                          "negado": negated, "fonte": "lexico-local"})
    return found


def sentiment(text: str) -> dict:
    norm = normalize(text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    pos = sum(norm.count(f" {w} ") + norm.startswith(w + " ") for w in POSITIVE)
    neg = sum(norm.count(f" {w} ") + norm.startswith(w + " ") for w in NEGATIVE)
    total = pos + neg
    if total == 0:
        scores = {"positivo": 0.1, "neutro": 0.8, "negativo": 0.1}
    else:
        scores = {"positivo": round(pos / (total + 1), 3), "neutro": round(1 / (total + 1), 3),
                  "negativo": round(neg / (total + 1), 3)}
    label = max(scores, key=scores.get)
    per_sentence = []
    for s in sentences:
        n = normalize(s)
        p = sum(f" {w} " in f" {n} " for w in POSITIVE)
        q = sum(f" {w} " in f" {n} " for w in NEGATIVE)
        per_sentence.append({"texto": s.strip(), "sentimento": "negativo" if q > p else "positivo" if p > q else "neutro"})
    return {"sentimento": label, "scores": scores, "frases": per_sentence, "fonte": "lexico-local"}
