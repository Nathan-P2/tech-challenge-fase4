"""Análise de vídeo clínico: postura, amplitude, tremor, equilíbrio, queda e zonas críticas.

Fluxo:
1. Cada quadro passa pelo backend de pose (YOLOv8-pose ou OpenPose) e,
   a cada N quadros, pelo detector de objetos YOLOv8.
2. Um modelo de fundo (quadro de referência da sala vazia ou MOG2) isola o
   paciente e mede energia de movimento — isso mantém o monitoramento mesmo
   quando o detector de pose perde o paciente deitado no chão.
3. Séries temporais de ângulos articulares alimentam detectores de desvio:
   assimetria/perda de amplitude por repetição, tremor (análise espectral),
   desequilíbrio (inclinação do tronco), queda, imobilidade e violação de zonas.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from ..alerts import Alert
from .keypoints import KP, SKELETON, angle_between

COCO_RELEVANT = {"person", "bed", "chair", "couch", "bottle", "cup", "scissors", "knife", "cell phone", "laptop", "tv"}


@dataclass
class VideoConfig:
    asym_deg: float = 30.0           # diferença L/R de amplitude por repetição
    rom_drop: float = 0.25           # queda de amplitude vs. repetições iniciais
    exercise_min_peak: float = 100.0  # pico mínimo para considerar repetição de abdução
    tremor_band: tuple[float, float] = (3.5, 12.0)
    tremor_window_s: float = 2.0
    tremor_min_amp: float = 0.025    # RMS do resíduo / comprimento do tronco
    tremor_peak_ratio: float = 0.5   # concentração espectral no pico
    tremor_max_amp: float = 0.15
    tremor_min_s: float = 3.0
    tilt_deg: float = 12.0
    tilt_min_s: float = 0.6
    fall_tilt_deg: float = 55.0
    fall_window_s: float = 1.6
    immobile_s: float = 4.0
    motion_thresh: float = 0.015     # pixels alterados / área da silhueta (janela de 0,2 s)
    immobile_upright_s: float = 60.0
    det_every: int = 10
    det_conf: float = 0.4


@dataclass
class VideoEvent:
    type: str
    start: float
    end: float
    severity: str
    description: str
    metrics: dict = field(default_factory=dict)


SEVERITY = {
    "queda": "critica",
    "imobilidade_pos_queda": "critica",
    "paciente_no_chao": "alta",
    "desequilibrio": "moderada",
    "tremor": "moderada",
    "assimetria_amplitude": "moderada",
    "perda_amplitude": "baixa",
    "saida_zona_segura": "moderada",
    "imobilidade": "baixa",
}


def _ffmpeg_h264(src: Path, dst: Path) -> Path:
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True)
        src.unlink(missing_ok=True)
        return dst
    except (FileNotFoundError, subprocess.CalledProcessError):
        return src


class VideoAnalyzer:
    def __init__(self, pose_backend, detector=None, config: VideoConfig | None = None,
                 zones: list[dict] | None = None, reference_frame: np.ndarray | None = None):
        self.pose = pose_backend
        self.detector = detector
        self.cfg = config or VideoConfig()
        self.zones = zones or []
        self.reference = reference_frame

    # ------------------------------------------------------------------ extração
    def extract(self, video_path: str | Path, annotate_to: Path | None = None) -> tuple[pd.DataFrame, np.ndarray, list]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        ref_gray = cv2.GaussianBlur(cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY), (5, 5), 0) if self.reference is not None else None
        rows, kps, detections, frames = [], [], [], []
        lag = max(1, int(round(fps * 0.2)))  # compara com o quadro de 0,2 s atrás
        history: list[np.ndarray] = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if ref_gray is not None:
                fg = (cv2.absdiff(gray, ref_gray) > 28).astype(np.uint8) * 255
                fg[:32] = 0  # ignora a faixa com o carimbo de tempo da câmera
            else:
                fg = mog.apply(frame, learningRate=0.001)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            n, _, stats, cents = cv2.connectedComponentsWithStats(fg)
            blob = None
            if n > 1:
                j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[j, cv2.CC_STAT_AREA] > 800:
                    blob = stats[j]
            motion = 0.0
            if len(history) >= lag:
                diff = cv2.absdiff(gray, history[-lag])
                diff[:32] = 0
                # pixels alterados normalizados pela área da silhueta (independe da distância à câmera)
                area = float(blob[cv2.CC_STAT_AREA]) if blob is not None else 0.05 * gray.size
                changed = cv2.morphologyEx((diff > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                motion = float(changed.sum()) / max(area, 1000.0)
            history.append(gray)
            if len(history) > lag:
                history.pop(0)

            kp, conf = self.pose(frame, i)
            row = {"frame": i, "t": t, "motion": motion, "pose": kp is not None,
                   "pose_conf": float(conf.mean()) if conf is not None else 0.0}
            if blob is not None:
                x, y, w, h, area = blob
                row.update(blob_x=x, blob_y=y, blob_w=w, blob_h=h, blob_area=area, blob_aspect=w / max(h, 1))
            if kp is not None:
                row.update(self._pose_features(kp, conf))
                kps.append(kp)
            else:
                kps.append(np.full((17, 2), np.nan, np.float32))
            if self.detector is not None and i % self.cfg.det_every == 0:
                res = self.detector(frame, verbose=False, conf=self.cfg.det_conf)[0]
                for c, s, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist()):
                    label = res.names[int(c)]
                    if label in COCO_RELEVANT:
                        detections.append({"t": round(t, 2), "classe": label, "conf": round(s, 3),
                                           "bbox": [round(v) for v in b]})
            rows.append(row)
            if annotate_to is not None:
                frames.append(frame)
            i += 1
        cap.release()
        df = pd.DataFrame(rows)
        df.attrs["fps"] = fps
        self._frames = frames
        return df, np.stack(kps) if kps else np.zeros((0, 17, 2)), detections

    @staticmethod
    def _pose_features(kp: np.ndarray, conf: np.ndarray) -> dict:
        mid_sho = (kp[KP.L_SHOULDER] + kp[KP.R_SHOULDER]) / 2
        mid_hip = (kp[KP.L_HIP] + kp[KP.R_HIP]) / 2
        torso = mid_sho - mid_hip
        torso_len = float(np.linalg.norm(torso)) + 1e-6
        # inclinação do tronco em relação à vertical (imagem: y cresce para baixo)
        tilt = math.degrees(math.atan2(torso[0], -torso[1]))
        f = {
            "abd_esq": float(angle_between(kp[KP.L_HIP], kp[KP.L_SHOULDER], kp[KP.L_ELBOW])),
            "abd_dir": float(angle_between(kp[KP.R_HIP], kp[KP.R_SHOULDER], kp[KP.R_ELBOW])),
            "cotovelo_esq": float(angle_between(kp[KP.L_SHOULDER], kp[KP.L_ELBOW], kp[KP.L_WRIST])),
            "cotovelo_dir": float(angle_between(kp[KP.R_SHOULDER], kp[KP.R_ELBOW], kp[KP.R_WRIST])),
            "joelho_esq": float(angle_between(kp[KP.L_HIP], kp[KP.L_KNEE], kp[KP.L_ANKLE])),
            "joelho_dir": float(angle_between(kp[KP.R_HIP], kp[KP.R_KNEE], kp[KP.R_ANKLE])),
            "inclinacao_tronco": tilt,
            "tronco_px": torso_len,
            "cabeca_y": float(kp[KP.NOSE][1]),
            "quadril_y": float(mid_hip[1]),
            "pes_x": float((kp[KP.L_ANKLE][0] + kp[KP.R_ANKLE][0]) / 2),
            "pes_y": float((kp[KP.L_ANKLE][1] + kp[KP.R_ANKLE][1]) / 2),
        }
        for side, idx in (("esq", KP.L_WRIST), ("dir", KP.R_WRIST)):
            f[f"punho_{side}_x"] = float(kp[idx][0])
            f[f"punho_{side}_y"] = float(kp[idx][1])
            f[f"punho_{side}_conf"] = float(conf[idx])
        return f

    # ------------------------------------------------------------------ detectores
    def detect(self, df: pd.DataFrame) -> tuple[list[VideoEvent], dict]:
        fps = df.attrs.get("fps", 25.0)
        events: list[VideoEvent] = []
        summary: dict = {}
        pose = df[df["pose"]].copy()
        if len(pose) > 10:
            ev, reps = self._rom_events(pose, fps)
            events += ev
            summary["repeticoes_abducao"] = reps
            events += self._tremor_events(pose, fps)
            events += self._tilt_events(pose, fps)
            summary["agachamentos"] = self._count_squats(pose, fps)
        fall_events = self._fall_events(df, fps)
        # inclinação que termina em queda faz parte da própria queda
        events = [e for e in events if not (e.type == "desequilibrio" and
                  any(e.start <= f.end and e.end >= f.start for f in fall_events))]
        events += fall_events
        events += self._immobility_events(df, fps, fall_events)
        events += self._zone_events(df, fps)
        events.sort(key=lambda e: e.start)
        summary["quadros"] = int(len(df))
        summary["quadros_com_pose"] = int(df["pose"].sum())
        summary["duracao_s"] = round(float(df["t"].iloc[-1]), 2) if len(df) else 0
        return events, summary

    def _rom_events(self, pose: pd.DataFrame, fps: float) -> tuple[list[VideoEvent], list[dict]]:
        cfg = self.cfg
        t = pose["t"].to_numpy()
        both = np.fmax(pose["abd_esq"].to_numpy(), pose["abd_dir"].to_numpy())
        win = max(5, int(fps * 0.4) | 1)
        both_s = savgol_filter(both, win, 2) if len(both) > win else both
        peaks, _ = find_peaks(both_s, prominence=40, distance=int(fps * 1.5), height=cfg.exercise_min_peak)
        reps = []
        for n, p in enumerate(peaks):
            lo, hi = max(0, p - int(fps * 0.5)), min(len(t), p + int(fps * 0.5))
            reps.append({"rep": n + 1, "t": round(float(t[p]), 2),
                         "esq": round(float(np.nanmax(pose["abd_esq"].iloc[lo:hi])), 1),
                         "dir": round(float(np.nanmax(pose["abd_dir"].iloc[lo:hi])), 1)})
        events = []
        if len(reps) < 2:
            return events, reps
        base_l = float(np.median([r["esq"] for r in reps[:2]]))
        base_r = float(np.median([r["dir"] for r in reps[:2]]))
        flagged = []
        for r in reps:
            diff = r["esq"] - r["dir"]
            drop_l = 1 - r["esq"] / base_l
            drop_r = 1 - r["dir"] / base_r
            kind = None
            if abs(diff) >= cfg.asym_deg:
                kind = "assimetria_amplitude"
            elif max(drop_l, drop_r) >= cfg.rom_drop:
                kind = "perda_amplitude"
            r["desvio"] = kind
            if kind:
                flagged.append((r, kind, diff, drop_l, drop_r))
        period = float(np.median(np.diff([r["t"] for r in reps]))) if len(reps) > 1 else 3.0
        for r, kind, diff, dl, dr in flagged:
            start, end = r["t"] - period / 2, r["t"] + period / 2
            side = "esquerdo" if r["esq"] < r["dir"] else "direito"
            desc = (f"Repetição {r['rep']}: abdução esq {r['esq']:.0f}° vs dir {r['dir']:.0f}° "
                    f"(Δ {abs(diff):.0f}°); membro {side} abaixo da amplitude inicial "
                    f"({max(dl, dr) * 100:.0f}% de perda).")
            if events and events[-1].type == kind and start - events[-1].end < period:
                events[-1].end = end
                events[-1].metrics["repeticoes"].append(r["rep"])
                events[-1].description = events[-1].description.split(" | ")[0] + f" | última: {desc}"
            else:
                events.append(VideoEvent(kind, start, end, SEVERITY[kind], desc,
                                         {"repeticoes": [r["rep"]], "amplitude_base_esq": base_l,
                                          "amplitude_base_dir": base_r}))
        return events, reps

    def _tremor_events(self, pose: pd.DataFrame, fps: float) -> list[VideoEvent]:
        cfg = self.cfg
        events = []
        win = int(cfg.tremor_window_s * fps)
        step = max(1, win // 4)
        torso = pose["tronco_px"].rolling(25, min_periods=1).median().to_numpy()
        t = pose["t"].to_numpy()
        sg = max(5, int(fps * 0.32) | 1)
        for side in ("esq", "dir"):
            x = pose[f"punho_{side}_x"].to_numpy()
            y = pose[f"punho_{side}_y"].to_numpy()
            if len(x) < win + sg:
                continue
            wconf = pose[f"punho_{side}_conf"].to_numpy()
            rx = (x - savgol_filter(x, sg, 2)) / torso
            ry = (y - savgol_filter(y, sg, 2)) / torso
            flags = []
            for s in range(0, len(rx) - win, step):
                seg = rx[s:s + win] + 1j * ry[s:s + win]
                if t[s + win - 1] - t[s] > cfg.tremor_window_s * 1.5:  # lacuna na pose
                    continue
                if np.median(wconf[s:s + win]) < 0.5:  # punho mal localizado pelo modelo
                    continue
                spec = np.abs(np.fft.fft(seg - seg.mean())) ** 2
                freqs = np.abs(np.fft.fftfreq(len(seg), 1 / fps))
                band = (freqs >= 0.5)
                tb = (freqs >= cfg.tremor_band[0]) & (freqs <= cfg.tremor_band[1])
                if spec[band].sum() == 0 or not tb.any():
                    continue
                f_peak = freqs[tb][np.argmax(spec[tb])]
                near = band & (np.abs(freqs - f_peak) <= 1.0)
                ratio = spec[near].sum() / spec[band].sum()
                rms = float(np.sqrt(np.mean(np.abs(seg - seg.mean()) ** 2)))
                # amplitude muito alta = troca/salto de keypoint, não tremor fisiológico
                if cfg.tremor_min_amp <= rms <= cfg.tremor_max_amp and ratio >= cfg.tremor_peak_ratio:
                    # marca o centro da janela como início para não antecipar o evento
                    flags.append((t[s + win // 2], t[s + win - 1], f_peak, rms, ratio))
            for f in flags:
                if events and events[-1].metrics["lado"] == side and f[0] <= events[-1].end:
                    e = events[-1]
                    e.end = max(e.end, f[1])
                    e.metrics["freq_hz"].append(round(float(f[2]), 2))
                    e.metrics["amp_rel"] = max(e.metrics["amp_rel"], round(f[3], 4))
                else:
                    events.append(VideoEvent("tremor", f[0], f[1], SEVERITY["tremor"], "",
                                             {"lado": side, "freq_hz": [round(float(f[2]), 2)], "amp_rel": round(f[3], 4)}))
        events = [e for e in events if e.end - e.start >= cfg.tremor_min_s]
        for e in events:
            freq = float(np.median(e.metrics["freq_hz"]))
            e.metrics["freq_hz"] = round(freq, 2)
            lado = "direito" if e.metrics["lado"] == "dir" else "esquerdo"
            e.description = (f"Oscilação rítmica de {freq:.1f} Hz no punho {lado} "
                             f"(amplitude {e.metrics['amp_rel'] * 100:.1f}% do tronco) — compatível com tremor.")
        return events

    def _tilt_events(self, pose: pd.DataFrame, fps: float) -> list[VideoEvent]:
        cfg = self.cfg
        t = pose["t"].to_numpy()
        tilt = pose["inclinacao_tronco"].rolling(int(fps * 0.4) | 1, center=True, min_periods=1).median().to_numpy()
        mask = (np.abs(tilt) >= cfg.tilt_deg) & (np.abs(tilt) < cfg.fall_tilt_deg)
        events = []
        for s, e in _runs(mask):
            if t[e] - t[s] >= cfg.tilt_min_s:
                peak = float(np.max(np.abs(tilt[s:e + 1])))
                events.append(VideoEvent("desequilibrio", float(t[s]), float(t[e]), SEVERITY["desequilibrio"],
                                         f"Inclinação lateral do tronco de até {peak:.0f}° por {t[e] - t[s]:.1f}s "
                                         "— perda de controle postural.", {"inclinacao_max": round(peak, 1)}))
        return _merge(events, gap=1.0)

    def _count_squats(self, pose: pd.DataFrame, fps: float) -> int:
        knee = pose[["joelho_esq", "joelho_dir"]].mean(axis=1).to_numpy()
        peaks, _ = find_peaks(-knee, prominence=25, distance=int(fps * 1.5))
        return int(len(peaks))

    def _fall_events(self, df: pd.DataFrame, fps: float) -> list[VideoEvent]:
        """Queda = transição rápida de postura vertical para horizontal.

        Usa três sinais independentes: inclinação do tronco (pose), razão de aspecto
        da silhueta (fundo) e perda da pose logo após inclinação acentuada.
        """
        cfg = self.cfg
        t = df["t"].to_numpy()
        tilt = df.get("inclinacao_tronco", pd.Series(np.nan, index=df.index)).to_numpy()
        aspect = df.get("blob_aspect", pd.Series(np.nan, index=df.index)).to_numpy()
        has_pose = df["pose"].to_numpy()
        win = int(cfg.fall_window_s * fps)
        events = []
        i = 0
        upright_aspect = np.nanmedian(aspect[: int(fps * 3)]) if np.isfinite(aspect[: int(fps * 3)]).any() else 0.6
        while i < len(df):
            start_ok = (np.isfinite(tilt[i]) and abs(tilt[i]) >= 25) or (np.isfinite(aspect[i]) and aspect[i] > upright_aspect * 1.6)
            if not start_ok:
                i += 1
                continue
            j_end = min(len(df) - 1, i + win)
            horizontal = None
            for j in range(i, j_end + 1):
                lying_pose = np.isfinite(tilt[j]) and abs(tilt[j]) >= cfg.fall_tilt_deg
                lying_blob = np.isfinite(aspect[j]) and aspect[j] >= 1.5
                lost_pose = not has_pose[j] and lying_blob
                if lying_pose or lying_blob or lost_pose:
                    horizontal = j
                    break
            if horizontal is not None:
                # recua até o início da inclinação (≥ 20°) para marcar o início da queda
                s = i
                while s > 0 and np.isfinite(tilt[s - 1]) and abs(tilt[s - 1]) >= 20:
                    s -= 1
                evidence = {"inclinacao_max": round(float(np.nanmax(np.abs(tilt[s:horizontal + 1]))), 1)
                            if np.isfinite(tilt[s:horizontal + 1]).any() else None,
                            "aspecto_silhueta": round(float(aspect[horizontal]), 2) if np.isfinite(aspect[horizontal]) else None,
                            "pose_perdida": bool(not has_pose[horizontal])}
                events.append(VideoEvent("queda", float(t[s]), float(t[horizontal]), SEVERITY["queda"],
                                         f"Transição vertical→horizontal em {t[horizontal] - t[s]:.1f}s; "
                                         "silhueta passou a horizontal — provável queda.", evidence))
                # pula até o paciente voltar a ficar em pé
                k = horizontal
                while k < len(df) and np.isfinite(aspect[k]) and aspect[k] >= 1.2:
                    k += 1
                i = k + int(fps * 2)
            else:
                i += 1
        return events

    def _immobility_events(self, df: pd.DataFrame, fps: float, falls: list[VideoEvent]) -> list[VideoEvent]:
        cfg = self.cfg
        t = df["t"].to_numpy()
        motion = pd.Series(df["motion"].to_numpy()).rolling(int(fps), min_periods=1).median().to_numpy()
        aspect = df.get("blob_aspect", pd.Series(np.nan, index=df.index)).to_numpy()
        still = motion < cfg.motion_thresh
        still[: int(fps * 0.5)] = False
        events = []
        for s, e in _runs(still):
            dur = t[e] - t[s]
            lying = np.nanmedian(aspect[s:e + 1]) >= 1.2 if np.isfinite(aspect[s:e + 1]).any() else False
            # parado em pé/sentado é normal; só alerta se deitado ou se a imobilidade for longa
            if dur < cfg.immobile_s or (not lying and dur < cfg.immobile_upright_s):
                continue
            after_fall = any(f.end - 1.0 <= t[s] <= f.end + 5.0 for f in falls)
            kind = "imobilidade_pos_queda" if after_fall else "imobilidade"
            desc = (f"Sem movimento detectável por {dur:.1f}s" +
                    (" após a queda — possível perda de consciência ou lesão." if after_fall else "."))
            events.append(VideoEvent(kind, float(t[s]), float(t[e]), SEVERITY[kind], desc,
                                     {"duracao_s": round(dur, 1), "energia_movimento": round(float(motion[s:e + 1].mean()), 5)}))
        return events

    def _zone_events(self, df: pd.DataFrame, fps: float) -> list[VideoEvent]:
        events = []
        t = df["t"].to_numpy()
        for zone in self.zones:
            x1, y1, x2, y2 = zone["bbox"]
            if zone.get("tipo") == "proibida":
                # silhueta (ou cabeça) dentro de zona proibida, ex.: piso
                top = df.get("blob_y", pd.Series(np.nan, index=df.index)).to_numpy()
                head = df.get("cabeca_y", pd.Series(np.nan, index=df.index)).to_numpy()
                ref = np.where(np.isfinite(head), head, top)
                mask = np.isfinite(ref) & (ref >= y1) & (ref <= y2)
                kind = "paciente_no_chao"
                text = f"Cabeça/tronco do paciente dentro da zona '{zone['nome']}'"
            else:
                feet = df.get("pes_x", pd.Series(np.nan, index=df.index)).to_numpy()
                mask = np.isfinite(feet) & ((feet < x1) | (feet > x2))
                kind = "saida_zona_segura"
                text = f"Pés fora da zona segura '{zone['nome']}'"
            for s, e in _runs(mask):
                if t[e] - t[s] >= 1.0:
                    events.append(VideoEvent(kind, float(t[s]), float(t[e]), SEVERITY[kind],
                                             f"{text} por {t[e] - t[s]:.1f}s.", {"zona": zone["nome"]}))
        return _merge(events, gap=1.0)

    # ------------------------------------------------------------------ saída
    def write_annotated(self, df: pd.DataFrame, kps: np.ndarray, events: list[VideoEvent], out_path: Path) -> Path:
        fps = df.attrs.get("fps", 25.0)
        frames = getattr(self, "_frames", [])
        if not frames:
            return out_path
        h, w = frames[0].shape[:2]
        tmp = out_path.with_suffix(".tmp.mp4")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        colors = {"critica": (40, 40, 220), "alta": (30, 120, 240), "moderada": (20, 170, 230), "baixa": (160, 160, 60)}
        for i, frame in enumerate(frames):
            img = frame.copy()
            t = i / fps
            kp = kps[i]
            if np.isfinite(kp).all():
                for a, b in SKELETON:
                    cv2.line(img, tuple(np.int32(kp[a])), tuple(np.int32(kp[b])), (0, 220, 120), 2, cv2.LINE_AA)
                for p in kp:
                    cv2.circle(img, tuple(np.int32(p)), 3, (0, 90, 255), -1, cv2.LINE_AA)
                row = df.iloc[i]
                cv2.putText(img, f"abd E {row['abd_esq']:.0f}  D {row['abd_dir']:.0f}  tronco {row['inclinacao_tronco']:+.0f}",
                            (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
            elif "blob_x" in df.columns and np.isfinite(df.iloc[i].get("blob_x", np.nan)):
                r = df.iloc[i]
                cv2.rectangle(img, (int(r.blob_x), int(r.blob_y)), (int(r.blob_x + r.blob_w), int(r.blob_y + r.blob_h)),
                              (0, 140, 255), 2)
                cv2.putText(img, "pose perdida - rastreando silhueta", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 90, 200), 1, cv2.LINE_AA)
            for zone in self.zones:
                x1, y1, x2, y2 = zone["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 0) if zone.get("tipo") != "proibida" else (60, 60, 200), 1)
            active = [e for e in events if e.start <= t <= e.end + 1.0]
            for k, e in enumerate(active[:3]):
                y = 36 + 28 * k
                cv2.rectangle(img, (w - 330, y - 18), (w - 8, y + 6), colors.get(e.severity, (90, 90, 90)), -1)
                cv2.putText(img, f"ALERTA: {e.type.replace('_', ' ')}", (w - 322, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        return _ffmpeg_h264(tmp, out_path)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge(events: list[VideoEvent], gap: float) -> list[VideoEvent]:
    out: list[VideoEvent] = []
    for e in sorted(events, key=lambda e: (e.type, e.start)):
        if out and out[-1].type == e.type and e.start - out[-1].end <= gap:
            out[-1].end = max(out[-1].end, e.end)
        else:
            out.append(e)
    return sorted(out, key=lambda e: e.start)


# ---------------------------------------------------------------------- avaliação
GT_TO_PRED = {"assimetria_amplitude": {"assimetria_amplitude", "perda_amplitude"}, "tremor": {"tremor"},
              "desequilibrio": {"desequilibrio"}, "queda": {"queda"},
              "imobilidade": {"imobilidade_pos_queda", "imobilidade"}}


def evaluate_events(events: list[VideoEvent], expected: list[dict], tolerance: float = 1.0) -> dict:
    results, matched = [], set()
    for gt in expected:
        kinds = GT_TO_PRED.get(gt["tipo"], {gt["tipo"]})
        hits = [k for k, e in enumerate(events) if e.type in kinds
                and e.start <= gt["fim"] + tolerance and e.end >= gt["inicio"] - tolerance]
        matched.update(hits)
        latency = min((events[k].start - gt["inicio"] for k in hits), default=None)
        results.append({"evento": gt["tipo"], "detectado": bool(hits),
                        "latencia_s": round(latency, 2) if latency is not None else None})
    auxiliary = {"paciente_no_chao", "saida_zona_segura"}  # eventos derivados (não estão no gabarito)
    fp = [e.type for k, e in enumerate(events) if k not in matched and e.type not in auxiliary]
    tp = sum(r["detectado"] for r in results)
    return {"eventos": results, "recall": round(tp / max(len(expected), 1), 3), "falsos_positivos": fp,
            "precisao": round((len(events) - len(fp) - sum(e.type in auxiliary for e in events)) /
                              max(len(events) - sum(e.type in auxiliary for e in events), 1), 3)}


def pose_accuracy(pred: np.ndarray, truth: np.ndarray, alpha: float = 0.2) -> dict:
    """PCK@alpha: keypoint correto se erro < alpha × comprimento do tronco verdadeiro."""
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    valid = np.isfinite(pred).all(axis=(1, 2))
    if not valid.any():
        return {"pck": 0.0, "erro_mediano_px": None, "cobertura": 0.0}
    p, g = pred[valid], truth[valid]
    torso = np.linalg.norm((g[:, KP.L_SHOULDER] + g[:, KP.R_SHOULDER]) / 2 - (g[:, KP.L_HIP] + g[:, KP.R_HIP]) / 2, axis=1)
    err = np.linalg.norm(p - g, axis=2)
    return {"pck": round(float((err < alpha * torso[:, None]).mean()), 3),
            "erro_mediano_px": round(float(np.median(err)), 2),
            "cobertura": round(float(valid.mean()), 3)}


def events_to_alerts(events: list[VideoEvent], patient_id: str, session_start: datetime, video_name: str) -> list[Alert]:
    alerts = []
    for e in events:
        ts = session_start + timedelta(seconds=e.start)
        alerts.append(Alert(patient_id=patient_id, source="video", type=e.type, severity=e.severity,
                            message=e.description, timestamp=ts.isoformat(timespec="seconds"),
                            end=(session_start + timedelta(seconds=e.end)).isoformat(timespec="seconds"),
                            score=1.0, evidence={"video": video_name, "inicio_s": round(e.start, 2),
                                                 "fim_s": round(e.end, 2), **e.metrics}))
    return alerts


def analyze_video(video_path: Path, out_dir: Path, backend, detector=None, zones=None, reference=None,
                  config: VideoConfig | None = None, truth_keypoints: np.ndarray | None = None,
                  expected_events: list[dict] | None = None, annotate: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzer = VideoAnalyzer(backend, detector, config, zones, reference)
    df, kps, detections = analyzer.extract(video_path, annotate_to=out_dir if annotate else None)
    events, summary = analyzer.detect(df)
    df.to_csv(out_dir / "video_frames.csv", index=False)
    result = {"video": str(video_path.name), "backend_pose": getattr(backend, "name", "?"),
              "resumo": summary, "eventos": [asdict(e) for e in events],
              "objetos_detectados": _summarize_detections(detections)}
    if truth_keypoints is not None:
        result["precisao_pose"] = pose_accuracy(kps, truth_keypoints)
    if expected_events is not None:
        result["avaliacao"] = evaluate_events(events, expected_events)
    if annotate:
        result["video_anotado"] = analyzer.write_annotated(df, kps, events, out_dir / "video_anotado.mp4").name
    (out_dir / "video_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    (out_dir / "video_report.md").write_text(render_video_report(result))
    result["_df"] = df
    result["_events"] = events
    return result


def _summarize_detections(dets: list[dict]) -> dict:
    out: dict = {}
    for d in dets:
        c = out.setdefault(d["classe"], {"ocorrencias": 0, "conf_max": 0.0})
        c["ocorrencias"] += 1
        c["conf_max"] = max(c["conf_max"], d["conf"])
    return out


def render_video_report(result: dict) -> str:
    """Relatório automático de desvios do procedimento (Markdown)."""
    s = result["resumo"]
    lines = [f"# Relatório automático — {result['video']}", "",
             f"- Backend de pose: **{result['backend_pose']}**",
             f"- Duração: {s.get('duracao_s')} s · quadros: {s.get('quadros')} · com pose: {s.get('quadros_com_pose')}",
             f"- Repetições de abdução: {len(s.get('repeticoes_abducao', []))} · agachamentos: {s.get('agachamentos', 0)}", ""]
    reps = s.get("repeticoes_abducao", [])
    if reps:
        lines += ["## Amplitude por repetição", "", "| Rep | t (s) | Esq (°) | Dir (°) | Desvio |", "|---|---|---|---|---|"]
        lines += [f"| {r['rep']} | {r['t']} | {r['esq']} | {r['dir']} | {r.get('desvio') or '—'} |" for r in reps]
        lines.append("")
    lines += ["## Desvios detectados", ""]
    if not result["eventos"]:
        lines.append("Nenhum desvio relevante.")
    for e in result["eventos"]:
        lines.append(f"- **{e['type']}** ({e['severity']}) {e['start']:.1f}s–{e['end']:.1f}s: {e['description']}")
    if result.get("avaliacao"):
        ev = result["avaliacao"]
        lines += ["", "## Avaliação contra o gabarito", "", f"Recall {ev['recall']:.0%} · precisão {ev['precisao']:.0%}", ""]
        lines += [f"- {r['evento']}: {'detectado' if r['detectado'] else 'NÃO detectado'}"
                  + (f" (latência {r['latencia_s']:+.1f}s)" if r["latencia_s"] is not None else "") for r in ev["eventos"]]
    if result.get("precisao_pose"):
        p = result["precisao_pose"]
        lines += ["", f"Precisão da pose: PCK@0.2 = {p['pck']:.1%}, erro mediano {p['erro_mediano_px']} px, "
                      f"cobertura {p['cobertura']:.0%}."]
    if result.get("objetos_detectados"):
        lines += ["", "## Objetos (YOLOv8)", ""]
        lines += [f"- {k}: {v['ocorrencias']} quadros (conf. máx {v['conf_max']:.2f})" for k, v in result["objetos_detectados"].items()]
    return "\n".join(lines) + "\n"
