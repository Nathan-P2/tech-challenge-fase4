"""Transcrição de consultas: Azure Speech to Text com fallbacks.

Ordem de tentativa (configurável por `TRANSCRIBER`):
1. **Azure Speech to Text** (`AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION`):
   reconhecimento contínuo em pt-BR, saída detalhada com tempo por palavra.
2. **Whisper local** (faster-whisper), útil sem internet/credenciais.
3. **Transcrição de referência** (`<arquivo>.txt` ao lado do áudio).
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Transcript:
    text: str
    engine: str
    language: str = "pt-BR"
    segments: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    error: str | None = None


class AzureSpeechTranscriber:
    name = "azure-speech-to-text"

    def __init__(self, key: str | None = None, region: str | None = None, language: str = "pt-BR"):
        self.key = key or os.getenv("AZURE_SPEECH_KEY")
        self.region = region or os.getenv("AZURE_SPEECH_REGION")
        self.language = language

    @property
    def available(self) -> bool:
        return bool(self.key and self.region)

    def transcribe(self, path: str | Path, timeout: float = 300) -> Transcript:
        import azure.cognitiveservices.speech as speechsdk

        cfg = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        cfg.speech_recognition_language = self.language
        cfg.output_format = speechsdk.OutputFormat.Detailed
        cfg.request_word_level_timestamps()
        audio_cfg = speechsdk.audio.AudioConfig(filename=str(path))
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

        segments: list[dict] = []
        errors: list[str] = []
        done = threading.Event()

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detail = json.loads(evt.result.json)
                best = (detail.get("NBest") or [{}])[0]
                segments.append({
                    "texto": evt.result.text,
                    "inicio_s": detail.get("Offset", 0) / 1e7,
                    "duracao_s": detail.get("Duration", 0) / 1e7,
                    "confianca": best.get("Confidence"),
                    "palavras": [{"palavra": w["Word"], "inicio_s": w["Offset"] / 1e7, "duracao_s": w["Duration"] / 1e7}
                                 for w in best.get("Words", [])],
                })

        def on_canceled(evt):
            details = evt.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                errors.append(f"{details.error_code}: {details.error_details}")
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(lambda _evt: done.set())
        recognizer.canceled.connect(on_canceled)
        t0 = time.time()
        recognizer.start_continuous_recognition()
        done.wait(timeout)
        recognizer.stop_continuous_recognition()
        if errors and not segments:
            raise RuntimeError(errors[0])
        return Transcript(" ".join(s["texto"] for s in segments), self.name, self.language, segments,
                          round(time.time() - t0, 2), errors[0] if errors else None)


class WhisperTranscriber:
    name = "faster-whisper"
    _model = None

    def __init__(self, model_size: str | None = None):
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "small")

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def transcribe(self, path: str | Path) -> Transcript:
        from faster_whisper import WhisperModel

        if WhisperTranscriber._model is None or WhisperTranscriber._model[0] != self.model_size:
            WhisperTranscriber._model = (self.model_size, WhisperModel(self.model_size, device="cpu", compute_type="int8"))
        model = WhisperTranscriber._model[1]
        t0 = time.time()
        # decodifica com librosa (evita dependência de versão do PyAV) e envia o array a 16 kHz
        import librosa

        audio, _ = librosa.load(str(path), sr=16000, mono=True)
        segs, _info = model.transcribe(audio, language="pt", word_timestamps=True, vad_filter=False)
        segments = [{"texto": s.text.strip(), "inicio_s": s.start, "duracao_s": s.end - s.start,
                     "confianca": round(float(2.718281828 ** s.avg_logprob), 3),
                     "palavras": [{"palavra": w.word.strip(), "inicio_s": w.start, "duracao_s": w.end - w.start}
                                  for w in (s.words or [])]} for s in segs]
        return Transcript(" ".join(s["texto"] for s in segments), f"{self.name}-{self.model_size}", "pt-BR", segments,
                          round(time.time() - t0, 2))


class ReferenceTranscriber:
    name = "transcricao-referencia"
    available = True

    def transcribe(self, path: str | Path) -> Transcript:
        txt = Path(path).with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"Sem transcrição de referência para {path}")
        return Transcript(txt.read_text(encoding="utf-8").strip(), self.name)


_CACHE: dict[tuple, Transcript] = {}


def transcribe(path: str | Path, prefer: str | None = None) -> Transcript:
    """Transcreve usando o primeiro mecanismo disponível (com cache por arquivo na execução)."""
    prefer = (prefer or os.getenv("TRANSCRIBER", "auto")).lower()
    key = (str(Path(path).resolve()), Path(path).stat().st_mtime, prefer)
    if key not in _CACHE:
        _CACHE[key] = _transcribe(path, prefer)
    return _CACHE[key]


def _transcribe(path: str | Path, prefer: str) -> Transcript:
    chain = {
        "azure": [AzureSpeechTranscriber()],
        "whisper": [WhisperTranscriber()],
        "referencia": [ReferenceTranscriber()],
    }.get(prefer, [AzureSpeechTranscriber(), WhisperTranscriber(), ReferenceTranscriber()])
    if prefer not in {"referencia"}:
        chain = chain + [ReferenceTranscriber()]
    errors = []
    for engine in chain:
        if not engine.available:
            continue
        try:
            result = engine.transcribe(path)
            if result.text.strip():
                if errors:
                    result.error = "; ".join(errors)
                return result
        except Exception as exc:  # cai para o próximo mecanismo
            errors.append(f"{engine.name}: {exc}")
    raise RuntimeError("Nenhum mecanismo de transcrição disponível: " + "; ".join(errors))
