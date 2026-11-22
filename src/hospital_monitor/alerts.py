"""Modelo de alerta e despacho para a equipe médica.

Os analisadores de cada modalidade produzem `Alert`s. O `AlertManager` aplica
deduplicação (janela de cooldown por paciente/origem/tipo), ordena por gravidade
e entrega aos *sinks* configurados: console, arquivo JSONL (trilha de auditoria)
e webhook (Microsoft Teams / Slack / Azure Logic Apps).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Protocol

SEVERITIES = ["info", "baixa", "moderada", "alta", "critica"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}
SEVERITY_LABEL = {"info": "INFO", "baixa": "BAIXA", "moderada": "MODERADA", "alta": "ALTA", "critica": "CRÍTICA"}
ACTION_BY_SEVERITY = {
    "info": "Registrar no prontuário.",
    "baixa": "Reavaliar na próxima ronda.",
    "moderada": "Enfermagem deve reavaliar o paciente em até 30 minutos.",
    "alta": "Avaliação médica em até 15 minutos.",
    "critica": "Acionar time de resposta rápida imediatamente.",
}


@dataclass
class Alert:
    patient_id: str
    source: str  # video | audio | sinais_vitais | prescricao | mobilidade | fusao
    type: str
    severity: str
    message: str
    timestamp: str
    score: float = 0.0
    evidence: dict = field(default_factory=dict)
    end: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def rank(self) -> int:
        return SEVERITY_RANK.get(self.severity, 0)

    @property
    def action(self) -> str:
        return ACTION_BY_SEVERITY.get(self.severity, "")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["acao_recomendada"] = self.action
        return data

    def format(self) -> str:
        return (f"[{SEVERITY_LABEL.get(self.severity, self.severity)}] {self.timestamp} | paciente {self.patient_id} | "
                f"{self.source}/{self.type}: {self.message} -> {self.action}")


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None:
        ...


class ConsoleSink:
    COLORS = {"critica": "\033[91m", "alta": "\033[93m", "moderada": "\033[96m"}

    def __init__(self, min_severity: str = "moderada"):
        self.min_rank = SEVERITY_RANK[min_severity]

    def send(self, alert: Alert) -> None:
        if alert.rank >= self.min_rank:
            color = self.COLORS.get(alert.severity, "")
            print(f"{color}{alert.format()}\033[0m")


class JsonlSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict(), ensure_ascii=False) + "\n")


class WebhookSink:
    """Envia alertas para um webhook (Teams/Slack/Logic Apps). URL via `ALERT_WEBHOOK_URL`."""

    def __init__(self, url: str | None = None, min_severity: str = "alta", timeout: float = 5.0):
        self.url = url or os.getenv("ALERT_WEBHOOK_URL")
        self.min_rank = SEVERITY_RANK[min_severity]
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        if not self.url or alert.rank < self.min_rank:
            return
        import urllib.request

        payload = {
            "text": f"**{SEVERITY_LABEL[alert.severity]}** · Paciente {alert.patient_id}\n\n{alert.message}\n\n"
                    f"Ação: {alert.action}",
            "alert": alert.to_dict(),
        }
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout)  # noqa: S310 - URL configurada pelo operador
        except Exception as exc:  # falha de rede não pode derrubar o monitoramento
            print(f"[webhook] falha ao enviar alerta {alert.id}: {exc}")


class AlertManager:
    def __init__(self, sinks: Iterable[AlertSink] = (), cooldown_minutes: float = 30.0):
        self.sinks = list(sinks)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self._last: dict[tuple[str, str, str], tuple[datetime, int]] = {}
        self.history: list[Alert] = []
        self.suppressed = 0

    def publish(self, alert: Alert) -> bool:
        """Entrega o alerta, exceto se for repetição sem escalonamento dentro do cooldown."""
        key = (alert.patient_id, alert.source, alert.type)
        ts = parse_ts(alert.timestamp)
        last = self._last.get(key)
        if last and ts - last[0] < self.cooldown and alert.rank <= last[1]:
            self.suppressed += 1
            return False
        self._last[key] = (ts, alert.rank)
        self.history.append(alert)
        for sink in self.sinks:
            sink.send(alert)
        return True

    def publish_many(self, alerts: Iterable[Alert]) -> int:
        ordered = sorted(alerts, key=lambda a: a.timestamp)
        return sum(self.publish(a) for a in ordered)

    def summary(self) -> dict:
        by_sev: dict[str, int] = {}
        by_src: dict[str, int] = {}
        for a in self.history:
            by_sev[a.severity] = by_sev.get(a.severity, 0) + 1
            by_src[a.source] = by_src.get(a.source, 0) + 1
        return {"total": len(self.history), "suprimidos": self.suppressed, "por_gravidade": by_sev, "por_origem": by_src}
