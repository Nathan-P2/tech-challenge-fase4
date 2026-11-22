"""Gera gravações sintéticas de consultas (pt-BR) com alterações vocais controladas.

Usa a síntese de voz do macOS (`say`) para produzir a fala base e aplica
transformações de sinal que reproduzem marcadores acústicos descritos na
literatura:

- **fadiga**: fala mais lenta, menor intensidade, F0 mais grave e monótono;
- **dificuldade respiratória**: frases curtas, pausas longas e frequentes com
  inspiração audível entre as frases;
- **disartria**: ritmo irregular (velocidade variável por trecho), articulação
  imprecisa (passa-baixa) e instabilidade de frequência (jitter).

Cada voz representa um paciente; a primeira gravação é a linha de base da
admissão, usada para comparar as consultas seguintes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 16000

VOICES = {
    "P-101": "Luciana",
    "P-102": "Flo (Portuguese (Brazil))",
    "P-103": "Eddy (Portuguese (Brazil))",
    "P-104": "Reed (Portuguese (Brazil))",
}

SCRIPTS = {
    "baseline": [
        "Bom dia, doutor. Meu nome está na pulseira e eu sei onde estou.",
        "Cheguei ontem para fazer a cirurgia do joelho e passei bem a noite.",
        "Estou tomando os remédios no horário certo e me alimentando bem.",
        "Hoje cedo caminhei pelo corredor com a fisioterapeuta sem dificuldade.",
    ],
    "controle": [
        "Boa tarde, doutor. Hoje eu estou me sentindo bem melhor.",
        "Dormi a noite inteira e a dor do joelho diminuiu bastante.",
        "Consegui fazer os exercícios e subir alguns degraus com ajuda.",
        "Estou animado e confiante para receber alta nos próximos dias.",
    ],
    "fadiga_respiratoria": [
        "Doutor, eu estou muito cansado.",
        "Desde ontem à noite sinto falta de ar,",
        "principalmente quando tento levantar da cama.",
        "Tenho uma dor no peito que aperta,",
        "e o coração fica acelerado.",
        "Não consegui dormir direito.",
        "Estou com medo de piorar.",
    ],
    "disartria": [
        "Doutor, desde hoje cedo a minha fala está enrolada.",
        "Sinto o braço direito fraco e formigando.",
        "Estou um pouco tonto e com a visão embaçada.",
        "Tenho dificuldade para engolir a água.",
    ],
}

LABELS = {
    "baseline": {"fadiga": 0, "dificuldade_respiratoria": 0, "disartria": 0},
    "controle": {"fadiga": 0, "dificuldade_respiratoria": 0, "disartria": 0},
    "fadiga_respiratoria": {"fadiga": 1, "dificuldade_respiratoria": 1, "disartria": 0},
    "disartria": {"fadiga": 0, "dificuldade_respiratoria": 0, "disartria": 1},
}

RATES = {"baseline": 180, "controle": 180, "fadiga_respiratoria": 135, "disartria": 125}


def _say(text: str, voice: str, rate: int) -> np.ndarray:
    if shutil.which("say") is None:
        raise RuntimeError("Geração de áudio sintético requer o comando `say` do macOS.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fala.wav"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(path), "--file-format=WAVE",
                        f"--data-format=LEI16@{SR}", text], check=True)
        audio, sr = sf.read(path, dtype="float32")
    assert sr == SR
    return _trim(audio)


def _trim(audio: np.ndarray, thr: float = 1e-3) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > thr)
    return audio[idx[0]: idx[-1] + 1] if len(idx) else audio


def _breath(duration: float, rng: np.random.Generator, level: float = 0.035) -> np.ndarray:
    """Ruído de inspiração: ruído rosa filtrado 300–2500 Hz com envelope suave."""
    n = int(duration * SR)
    white = rng.normal(0, 1, n)
    sos = butter(4, [300, 2500], btype="band", fs=SR, output="sos")
    noise = sosfilt(sos, white)
    env = np.sin(np.linspace(0, np.pi, n)) ** 1.5
    noise = noise / (np.abs(noise).max() + 1e-9)
    return (level * noise * env).astype(np.float32)


def _room_noise(n: int, rng: np.random.Generator, level: float = 0.0015) -> np.ndarray:
    return rng.normal(0, level, n).astype(np.float32)


def _pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(audio, sr=SR, n_steps=semitones)


def _dysarthric(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ritmo irregular + jitter de F0 + articulação imprecisa."""
    chunk = int(0.22 * SR)
    fade = int(0.01 * SR)
    out = np.zeros(0, np.float32)
    for start in range(0, len(audio), chunk):
        seg = audio[start:start + chunk + fade]
        if len(seg) < 4 * fade:
            out = np.concatenate([out, seg])
            continue
        # reamostragem com fator aleatório: altera velocidade e F0 juntos (ritmo irregular + jitter)
        rate = float(np.exp(rng.uniform(np.log(0.72), np.log(1.25))))
        n_new = max(4 * fade, int(len(seg) / rate))
        seg = np.interp(np.linspace(0, len(seg) - 1, n_new), np.arange(len(seg)), seg).astype(np.float32)
        if len(out) >= fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            out[-fade:] = out[-fade:] * (1 - ramp) + seg[:fade] * ramp
            seg = seg[fade:]
        out = np.concatenate([out, seg])
    sos = butter(4, 2200, btype="low", fs=SR, output="sos")  # consoantes menos nítidas
    out = sosfilt(sos, out)
    # voz áspera: ruído modulado pela própria envoltória
    env = np.abs(out)
    out = out + 0.25 * env * rng.normal(0, 1, len(out))
    return out.astype(np.float32)


def synthesize(condition: str, voice: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rate = RATES[condition]
    parts: list[np.ndarray] = [np.zeros(int(0.5 * SR), np.float32)]
    for phrase in SCRIPTS[condition]:
        speech = _say(phrase, voice, rate)
        if condition == "fadiga_respiratoria":
            speech = _pitch_shift(speech, -2.0)
            # menor intensidade e leve oscilação de amplitude (voz fraca/instável)
            t = np.arange(len(speech)) / SR
            speech = speech * 0.45 * (1 + 0.12 * np.sin(2 * np.pi * 5 * t))
            parts.append(speech.astype(np.float32))
            gap = float(rng.uniform(0.9, 1.4))
            pause = np.zeros(int(gap * SR), np.float32)
            b = _breath(min(0.55, gap - 0.2), rng)
            off = int(0.1 * SR)
            pause[off:off + len(b)] += b
            parts.append(pause)
        elif condition == "disartria":
            parts.append(_dysarthric(speech, rng))
            parts.append(np.zeros(int(rng.uniform(0.5, 0.8) * SR), np.float32))
        else:
            parts.append(speech)
            parts.append(np.zeros(int(rng.uniform(0.3, 0.45) * SR), np.float32))
    audio = np.concatenate(parts)
    audio = audio + _room_noise(len(audio), rng)
    peak = np.abs(audio).max()
    return (audio / peak * (0.45 if condition == "fadiga_respiratoria" else 0.9)).astype(np.float32)


def generate_consultations(out_dir: Path, voices: dict[str, str] | None = None) -> list[dict]:
    """Gera 4 gravações por paciente e um `manifest.json` com rótulos e transcrições."""
    out_dir.mkdir(parents=True, exist_ok=True)
    voices = voices or VOICES
    manifest = []
    for p_idx, (patient, voice) in enumerate(voices.items()):
        for c_idx, condition in enumerate(SCRIPTS):
            audio = synthesize(condition, voice, seed=100 * p_idx + c_idx)
            name = f"{patient}_{condition}.wav"
            sf.write(out_dir / name, audio, SR, subtype="PCM_16")
            transcript = " ".join(SCRIPTS[condition])
            (out_dir / name.replace(".wav", ".txt")).write_text(transcript, encoding="utf-8")
            manifest.append({"arquivo": name, "paciente": patient, "voz": voice, "condicao": condition,
                             "duracao_s": round(len(audio) / SR, 2), "rotulos": LABELS[condition],
                             "transcricao_referencia": transcript})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
