"""Extração de biomarcadores vocais.

Medidas calculadas (todas explicáveis e com significado clínico):

- **Temporais**: taxa de fala e de articulação (núcleos silábicos/s), razão de
  pausas, pausas por minuto, duração média das frases, irregularidade rítmica
  (coeficiente de variação dos intervalos entre sílabas).
- **Fonte glótica**: F0 média e variabilidade (semitons), jitter e shimmer
  (aproximação quadro a quadro), HNR (relação harmônico-ruído).
- **Energia/espectro**: intensidade média, centroide espectral, energia em
  alta frequência (nitidez articulatória).
- **Respiração**: inspirações audíveis detectadas nas pausas (ruído não vozeado
  com alta planicidade espectral).
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import find_peaks

SR = 16000
HOP = 160  # 10 ms
FRAME = 400  # 25 ms


@dataclass
class VoiceFeatures:
    duracao_s: float
    tempo_fala_s: float
    taxa_fala_sil_s: float
    taxa_articulacao_sil_s: float
    razao_pausas: float
    pausas_por_min: float
    pausa_media_s: float
    frase_media_s: float
    irregularidade_ritmo: float
    f0_media_hz: float
    f0_desvio_st: float
    jitter_pct: float
    shimmer_pct: float
    hnr_db: float
    intensidade_db: float
    centroide_hz: float
    energia_alta_freq: float
    respiracoes_audiveis: int
    respiracoes_por_min: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    runs = _runs(~mask)
    for s, e in runs:
        if s > 0 and e < len(mask) and e - s <= max_gap:
            out[s:e] = True
    return out


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask):
        if e - s < min_len:
            out[s:e] = False
    return out


def load_audio(path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


def extract_features(y: np.ndarray, sr: int = SR) -> tuple[VoiceFeatures, dict]:
    import librosa

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = y - y.mean()
    duration = len(y) / SR

    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=True)[0]
    db = 20 * np.log10(rms + 1e-10)
    peak_db = np.percentile(db, 99)
    floor_db = np.percentile(db, 10)
    thr = max(floor_db + 10, peak_db - 30)
    speech = db > thr
    speech = _fill_gaps(speech, max_gap=15)  # oclusivas e micro-silêncios < 150 ms
    speech = _drop_short(speech, min_len=6)  # rajadas < 60 ms

    # F0 (pYIN) e vozeamento
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced, vprob = librosa.pyin(y, fmin=65, fmax=450, sr=SR, frame_length=1024, hop_length=HOP)
    n = min(len(f0), len(speech))
    f0, voiced, speech, rms, db = f0[:n], voiced[:n], speech[:n], rms[:n], db[:n]
    flat = librosa.feature.spectral_flatness(y=y, n_fft=512, hop_length=HOP)[0][:n]

    # segmentos sem vozeamento e curtos são inspirações audíveis, não fala
    breaths = 0
    breath_mask = np.zeros(n, bool)
    for s, e in _runs(speech):
        dur = (e - s) * HOP / SR
        if dur < 1.0 and voiced[s:e].mean() < 0.15 and np.median(flat[s:e]) > 0.01:
            speech[s:e] = False
            breath_mask[s:e] = True
            breaths += 1
    segs = _runs(speech)

    # pausas entre frases (exclui silêncio inicial e final)
    pauses = []
    for (s1, e1), (s2, _e2) in zip(segs[:-1], segs[1:]):
        gap = (s2 - e1) * HOP / SR
        if gap >= 0.25:
            pauses.append((e1, s2, gap))
    active_start = segs[0][0] if segs else 0
    active_end = segs[-1][1] if segs else len(speech)
    active_s = max((active_end - active_start) * HOP / SR, 1e-6)
    speech_s = float(speech.sum() * HOP / SR)

    # frases = segmentos de fala separados por pausas >= 250 ms
    phrases, cur = [], None
    for s, e in segs:
        if cur is None:
            cur = [s, e]
        elif (s - cur[1]) * HOP / SR < 0.25:
            cur[1] = e
        else:
            phrases.append(cur)
            cur = [s, e]
    if cur:
        phrases.append(cur)
    phrase_lens = [(e - s) * HOP / SR for s, e in phrases]

    speech_f = speech
    vmask = voiced & speech_f & np.isfinite(f0)
    f0v = f0[vmask]
    if len(f0v) > 5:
        st = 12 * np.log2(f0v / np.median(f0v))
        f0_std = float(np.std(st))
        f0_mean = float(np.median(f0v))
        periods = 1.0 / f0
        # jitter/shimmer em quadros vozeados consecutivos
        cons = vmask[1:] & vmask[:-1]
        dp = np.abs(np.diff(periods))[cons]
        jitter = float(np.mean(dp) / np.mean(periods[vmask]) * 100) if dp.size else 0.0
        amp = rms[:n]
        da = np.abs(np.diff(amp))[cons]
        shimmer = float(np.mean(da) / np.mean(amp[vmask]) * 100) if da.size else 0.0
    else:
        f0_std = f0_mean = jitter = shimmer = 0.0

    # HNR por autocorrelação nos quadros vozeados
    frames = librosa.util.frame(np.pad(y, (FRAME // 2, FRAME // 2)), frame_length=FRAME, hop_length=HOP)
    frames = frames[:, :n]
    win = np.hanning(FRAME)
    win_ac = np.correlate(win, win, mode="full")[FRAME - 1:]
    win_ac = win_ac / win_ac[0]
    hnrs = []
    lag_min, lag_max = int(SR / 450), int(SR / 65)
    idx = np.flatnonzero(vmask)
    for i in idx[:: max(1, len(idx) // 400)]:
        fr = (frames[:, i] - frames[:, i].mean()) * win
        ac = np.correlate(fr, fr, mode="full")[FRAME - 1:]
        if ac[0] <= 0:
            continue
        # correção do viés da janela (Boersma, 1993)
        hi = min(lag_max, FRAME // 2)
        r = float(np.max(ac[lag_min:hi] / win_ac[lag_min:hi]) / ac[0])
        r = min(max(r, 1e-4), 0.9999)
        hnrs.append(10 * np.log10(r / (1 - r)))
    hnr = float(np.median(hnrs)) if hnrs else 0.0

    # núcleos silábicos: picos da energia em 300–3000 Hz dentro de fala vozeada
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    band = S[(freqs >= 300) & (freqs <= 3000)].sum(axis=0)
    band_db = 10 * np.log10(band + 1e-10)[:n]
    band_db = np.convolve(band_db, np.ones(5) / 5, mode="same")
    peaks, _ = find_peaks(band_db, prominence=3.0, distance=8)
    peaks = [p for p in peaks if speech_f[p] and (voiced[max(0, p - 3):p + 4].any())]
    n_syl = len(peaks)
    intervals = []
    for s, e in phrases:
        pk = [p for p in peaks if s <= p < e]
        intervals += list(np.diff(pk) * HOP / SR)
    intervals = np.array(intervals)
    rhythm_cv = float(intervals.std() / intervals.mean()) if len(intervals) > 3 else 0.0

    # espectro durante a fala
    sp = speech[: S.shape[1]]
    S_sp = S[:, : len(sp)][:, sp] if sp.any() else S
    total = S_sp.sum(axis=0) + 1e-10
    centroid = float(np.mean((freqs[:, None] * S_sp).sum(axis=0) / total))
    high = float(np.mean(S_sp[freqs >= 3000].sum(axis=0) / total))
    intensity = float(np.mean(db[speech])) if speech.any() else float(np.mean(db))


    feats = VoiceFeatures(
        duracao_s=duration,
        tempo_fala_s=speech_s,
        taxa_fala_sil_s=n_syl / active_s,
        taxa_articulacao_sil_s=n_syl / max(speech_s, 1e-6),
        razao_pausas=float(sum(p[2] for p in pauses) / active_s),
        pausas_por_min=len(pauses) / active_s * 60,
        pausa_media_s=float(np.mean([p[2] for p in pauses])) if pauses else 0.0,
        frase_media_s=float(np.mean(phrase_lens)) if phrase_lens else 0.0,
        irregularidade_ritmo=rhythm_cv,
        f0_media_hz=f0_mean,
        f0_desvio_st=f0_std,
        jitter_pct=jitter,
        shimmer_pct=shimmer,
        hnr_db=hnr,
        intensidade_db=intensity,
        centroide_hz=centroid,
        energia_alta_freq=high,
        respiracoes_audiveis=int(breaths),
    )
    feats.respiracoes_por_min = breaths / active_s * 60
    series = {"t": np.arange(n) * HOP / SR, "db": db[:n], "f0": f0, "fala": speech_f, "silabas": np.array(peaks),
              "respiracoes": [(s * HOP / SR, e * HOP / SR) for s, e in _runs(breath_mask)],
              "pausas": [(s * HOP / SR, e * HOP / SR) for s, e, _ in pauses]}
    return feats, series
