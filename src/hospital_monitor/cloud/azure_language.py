"""Azure AI Language (Text Analytics): sentimento, frases-chave e Text Analytics for Health.

Credenciais: `AZURE_LANGUAGE_ENDPOINT` e `AZURE_LANGUAGE_KEY`. Sem elas (ou em
caso de falha), usa o NLP clínico local, mantendo o mesmo formato de saída.
"""

from __future__ import annotations

import os

from ..audio import clinical_nlp

# Gravidade atribuída às categorias do Text Analytics for Health
HEALTH_SEVERITY = {
    "dispneia": 3, "dyspnea": 3, "falta de ar": 3, "dor no peito": 3, "dor torácica": 3, "chest pain": 3,
    "disartria": 3, "dysarthria": 3, "síncope": 3, "sangramento": 3, "palpitação": 2, "palpitations": 2,
    "tontura": 2, "dizziness": 2, "fadiga": 2, "fatigue": 2, "fraqueza": 2, "parestesia": 2, "disfagia": 2,
}


class AzureLanguageAnalyzer:
    name = "azure-text-analytics"

    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = endpoint or os.getenv("AZURE_LANGUAGE_ENDPOINT")
        self.key = key or os.getenv("AZURE_LANGUAGE_KEY")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.key)

    @property
    def client(self):
        if self._client is None:
            from azure.ai.textanalytics import TextAnalyticsClient
            from azure.core.credentials import AzureKeyCredential

            self._client = TextAnalyticsClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
        return self._client

    def sentiment(self, text: str) -> dict:
        doc = self.client.analyze_sentiment([text], language="pt-BR", show_opinion_mining=True)[0]
        if doc.is_error:
            raise RuntimeError(doc.error)
        sc = doc.confidence_scores
        return {
            "sentimento": {"positive": "positivo", "negative": "negativo", "neutral": "neutro", "mixed": "misto"}[doc.sentiment],
            "scores": {"positivo": sc.positive, "neutro": sc.neutral, "negativo": sc.negative},
            "frases": [{"texto": s.text, "sentimento": s.sentiment,
                        "opinioes": [{"alvo": o.target.text, "avaliacoes": [a.text for a in o.assessments]}
                                     for o in (s.mined_opinions or [])]} for s in doc.sentences],
            "fonte": self.name,
        }

    def key_phrases(self, text: str) -> list[str]:
        doc = self.client.extract_key_phrases([text], language="pt-BR")[0]
        return [] if doc.is_error else list(doc.key_phrases)

    def health_entities(self, text: str) -> list[dict]:
        poller = self.client.begin_analyze_healthcare_entities([text], language="pt")
        doc = next(iter(poller.result()))
        if doc.is_error:
            raise RuntimeError(doc.error)
        out = []
        for ent in doc.entities:
            assertion = ent.assertion
            negated = bool(assertion and assertion.certainty in {"negative", "negativePossible"})
            norm = (ent.normalized_text or ent.text).lower()
            sev = HEALTH_SEVERITY.get(norm, HEALTH_SEVERITY.get(ent.text.lower(), 1 if ent.category == "SymptomOrSign" else 0))
            out.append({"texto": ent.text, "categoria": ent.category, "normalizado": ent.normalized_text or ent.text,
                        "gravidade": sev, "negado": negated, "confianca": ent.confidence_score, "fonte": self.name})
        return out


def analyze_text(text: str, prefer_azure: bool = True) -> dict:
    """Retorna sentimento, termos críticos e frases-chave (Azure ou fallback local)."""
    azure = AzureLanguageAnalyzer()
    if prefer_azure and azure.available:
        try:
            return {"sentimento": azure.sentiment(text), "termos": azure.health_entities(text),
                    "frases_chave": azure.key_phrases(text), "motor": azure.name}
        except Exception as exc:
            local = _local(text)
            local["erro_azure"] = str(exc)
            return local
    return _local(text)


def _local(text: str) -> dict:
    return {"sentimento": clinical_nlp.sentiment(text), "termos": clinical_nlp.find_critical_terms(text),
            "frases_chave": [], "motor": "nlp-local"}
