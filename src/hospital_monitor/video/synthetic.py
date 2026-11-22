"""Gerador de sessão de fisioterapia sintética.

Renderiza um paciente (manequim 2D) executando abdução de ombros e agachamentos,
com anomalias injetadas em instantes conhecidos. Além do vídeo `.mp4`, o gerador
salva os keypoints verdadeiros no formato COCO-17 (o mesmo esqueleto usado pelo
OpenPose/YOLOv8-pose) e a lista de eventos esperados, permitindo medir a precisão
do pipeline de vídeo de forma objetiva.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .keypoints import KP

FPS = 25
WIDTH, HEIGHT = 640, 480
GROUND_Y = 440
MAT = (150, 400, 490, 452)  # x1, y1, x2, y2 do tapete de exercício (zona segura)

SKIN = (150, 180, 225)
SKIN_DARK = (120, 150, 200)
SHIRT = (170, 110, 40)
PANTS = (60, 60, 70)
HAIR = (35, 45, 60)


@dataclass
class Segment:
    """Trecho do roteiro da sessão."""

    name: str
    start: float
    end: float


@dataclass
class SessionScript:
    duration: float = 38.0
    segments: list[Segment] = field(default_factory=lambda: [
        Segment("repouso", 0.0, 3.0),
        Segment("abducao_ombros", 3.0, 21.0),
        Segment("agachamento", 21.0, 30.0),
        Segment("queda", 30.0, 31.0),
        Segment("imobilidade_pos_queda", 31.0, 38.0),
    ])
    # Anomalias esperadas (ground truth) usadas na avaliação.
    events: list[dict] = field(default_factory=lambda: [
        {"tipo": "assimetria_amplitude", "inicio": 12.0, "fim": 21.0,
         "descricao": "Braço esquerdo atinge ~95° contra ~160° do direito (compensação)."},
        {"tipo": "tremor", "inicio": 15.0, "fim": 21.0,
         "descricao": "Tremor de ~6 Hz no membro superior direito."},
        {"tipo": "desequilibrio", "inicio": 27.0, "fim": 30.0,
         "descricao": "Inclinação lateral do tronco no último agachamento."},
        {"tipo": "queda", "inicio": 30.0, "fim": 31.0,
         "descricao": "Queda lateral para fora do tapete."},
        {"tipo": "imobilidade", "inicio": 31.0, "fim": 38.0,
         "descricao": "Paciente imóvel no chão após a queda."},
    ])


def _smooth_cycle(phase: float) -> float:
    """0→1→0 suave para uma repetição (phase em [0, 1])."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * phase)


def _rotate(points: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return (points - pivot) @ rot.T + pivot


def pose_at(t: float, normal: bool = False) -> tuple[np.ndarray, dict]:
    """Retorna keypoints COCO-17 (coordenadas de imagem) e o estado do roteiro em t.

    `normal=True` gera a sessão de controle (mesmo roteiro, sem anomalias).
    """
    rng = np.random.default_rng(int(t * 1000))
    squat = 0.0
    left_abd = right_abd = 8.0  # graus a partir do braço pendente
    trunk_lean = 0.0
    fall = 0.0
    drift_x = 0.0
    tremor = 0.0
    sway = 2.0 * math.sin(2 * math.pi * 0.25 * t)

    if 3.0 <= t < 21.0:
        rep = int((t - 3.0) // 3.0)
        phase = ((t - 3.0) % 3.0) / 3.0
        lift = _smooth_cycle(phase)
        right_abd = 8 + 152 * lift
        left_rom = 152 if (rep < 3 or normal) else 87  # reps 4-6: amplitude reduzida à esquerda
        left_abd = 8 + left_rom * lift
        if t >= 15.0 and not normal:
            tremor = 1.0
    elif 21.0 <= t < 30.0:
        rep = int((t - 21.0) // 3.0)
        phase = ((t - 21.0) % 3.0) / 3.0
        squat = _smooth_cycle(phase)
        left_abd = right_abd = 8 + 70 * squat  # braços à frente para equilíbrio
        if rep == 2 and not normal:
            trunk_lean = math.radians(22) * squat + math.radians(3) * math.sin(2 * math.pi * 1.5 * t)
    elif normal:
        pass  # sessão de controle: termina em repouso, em pé
    elif 30.0 <= t < 31.0:
        p = (t - 30.0) / 1.0
        fall = (math.pi / 2) * min(1.0, p ** 2 * 1.4)
        drift_x = -90 * p
        left_abd, right_abd = 40 - 20 * p, 30 - 10 * p
    elif t >= 31.0:
        fall = math.pi / 2
        drift_x = -90
        left_abd, right_abd = 20, 20
        sway = 0.0

    # Coordenadas locais: origem no meio dos tornozelos, y para cima.
    hip_y = 165 - 62 * squat
    knee_out = 14 * squat
    pts = np.zeros((17, 2))
    pts[KP.L_ANKLE] = (22, 0)
    pts[KP.R_ANKLE] = (-22, 0)
    pts[KP.L_KNEE] = (24 + knee_out, 82 - 4 * squat)
    pts[KP.R_KNEE] = (-24 - knee_out, 82 - 4 * squat)
    pts[KP.L_HIP] = (22, hip_y)
    pts[KP.R_HIP] = (-22, hip_y)
    hip_center = np.array([0.0, hip_y])

    upper = np.zeros((17, 2))
    sy = hip_y + 112
    upper[KP.L_SHOULDER] = (36, sy)
    upper[KP.R_SHOULDER] = (-36, sy)
    head_c = np.array([0.0, sy + 42])
    upper[KP.NOSE] = head_c + (0, -4)
    upper[KP.L_EYE] = head_c + (8, 3)
    upper[KP.R_EYE] = head_c + (-8, 3)
    upper[KP.L_EAR] = head_c + (19, -1)
    upper[KP.R_EAR] = head_c + (-19, -1)

    def arm(sho, abd_deg, side):
        a = math.radians(abd_deg)
        d1 = np.array([side * math.sin(a), -math.cos(a)])
        a2 = a + math.radians(10)
        d2 = np.array([side * math.sin(a2), -math.cos(a2)])
        elbow = sho + 64 * d1
        wrist = elbow + 58 * d2
        return elbow, wrist

    upper[KP.L_ELBOW], upper[KP.L_WRIST] = arm(upper[KP.L_SHOULDER], left_abd, +1)
    upper[KP.R_ELBOW], upper[KP.R_WRIST] = arm(upper[KP.R_SHOULDER], right_abd, -1)
    if tremor:
        amp = 6.0
        osc = amp * math.sin(2 * math.pi * 6.0 * t)
        upper[KP.R_WRIST] += (osc, 0.6 * osc)
        upper[KP.R_ELBOW] += (0.4 * osc, 0.2 * osc)

    upper_idx = [KP.NOSE, KP.L_EYE, KP.R_EYE, KP.L_EAR, KP.R_EAR, KP.L_SHOULDER,
                 KP.R_SHOULDER, KP.L_ELBOW, KP.R_ELBOW, KP.L_WRIST, KP.R_WRIST]
    if trunk_lean:
        upper[upper_idx] = _rotate(upper[upper_idx], -trunk_lean, hip_center)
    pts[upper_idx] = upper[upper_idx]

    pts[:, 0] += sway
    if fall:
        pts = _rotate(pts, -fall, np.array([30.0, 0.0]))
        # mantém o tronco apoiado no chão e nenhum membro abaixo do piso
        core = [KP.L_HIP, KP.R_HIP, KP.L_SHOULDER, KP.R_SHOULDER, KP.L_ANKLE, KP.R_ANKLE]
        pts[:, 1] -= max(0.0, pts[core, 1].min() - 12 * (fall / (math.pi / 2)))
        pts[:, 1] = np.maximum(pts[:, 1], 0.0)
    pts[:, 0] += drift_x

    img = np.empty_like(pts)
    img[:, 0] = WIDTH / 2 + pts[:, 0]
    img[:, 1] = GROUND_Y - pts[:, 1]
    img += rng.normal(0, 0.1, img.shape)
    state = {"squat": squat, "left_abd": left_abd, "right_abd": right_abd,
             "trunk_lean": math.degrees(trunk_lean), "fall": math.degrees(fall)}
    return img, state


def _background() -> np.ndarray:
    bg = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for y in range(HEIGHT):
        shade = 225 - int(25 * y / HEIGHT)
        bg[y, :] = (shade, shade - 5, shade - 15)
    bg[GROUND_Y - 30:, :] = (150, 160, 170)  # piso
    cv2.rectangle(bg, (0, GROUND_Y - 32), (WIDTH, GROUND_Y - 28), (120, 125, 130), -1)
    x1, y1, x2, y2 = MAT
    cv2.rectangle(bg, (x1, y1), (x2, y2), (90, 140, 70), -1)  # tapete verde
    # barra paralela ao fundo
    cv2.rectangle(bg, (30, 230), (40, GROUND_Y - 30), (110, 110, 110), -1)
    cv2.rectangle(bg, (30, 230), (120, 238), (110, 110, 110), -1)
    cv2.putText(bg, "Sala de Fisioterapia 2", (380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
    return bg


def _limb(img, a, b, color, thick):
    cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color, thick, cv2.LINE_AA)
    for p in (a, b):
        cv2.circle(img, tuple(np.int32(p)), thick // 2, color, -1, cv2.LINE_AA)


def render(kp: np.ndarray, background: np.ndarray, t: float) -> np.ndarray:
    img = background.copy()
    # sombra
    feet = kp[[KP.L_ANKLE, KP.R_ANKLE]].mean(axis=0)
    cv2.ellipse(img, (int(feet[0]), GROUND_Y + 2), (55, 7), 0, 0, 360, (110, 120, 125), -1, cv2.LINE_AA)
    # pernas
    for hip, knee, ank in ((KP.L_HIP, KP.L_KNEE, KP.L_ANKLE), (KP.R_HIP, KP.R_KNEE, KP.R_ANKLE)):
        _limb(img, kp[hip], kp[knee], PANTS, 26)
        _limb(img, kp[knee], kp[ank], PANTS, 22)
        foot = kp[ank] + (kp[ank] - kp[knee]) / np.linalg.norm(kp[ank] - kp[knee] + 1e-6) * 6
        _limb(img, kp[ank], foot, (40, 40, 40), 18)
    # tronco
    ls, rs, lh, rh = kp[KP.L_SHOULDER], kp[KP.R_SHOULDER], kp[KP.L_HIP], kp[KP.R_HIP]
    out_s = (ls - rs) * 0.12
    out_h = (lh - rh) * 0.25
    torso = np.array([ls + out_s, rs - out_s, rh - out_h, lh + out_h], np.int32)
    cv2.fillConvexPoly(img, torso, SHIRT, cv2.LINE_AA)
    # pescoço e cabeça
    neck = (ls + rs) / 2
    head = kp[[KP.L_EAR, KP.R_EAR]].mean(axis=0)
    _limb(img, neck, head, SKIN_DARK, 18)
    up = head - neck
    up = up / (np.linalg.norm(up) + 1e-6)
    angle = math.degrees(math.atan2(up[1], up[0])) + 90
    cv2.ellipse(img, tuple(np.int32(head)), (20, 25), angle, 0, 360, SKIN, -1, cv2.LINE_AA)
    cv2.ellipse(img, tuple(np.int32(head + up * 9)), (21, 17), angle, 180, 360, HAIR, -1, cv2.LINE_AA)
    for eye in (KP.L_EYE, KP.R_EYE):
        cv2.circle(img, tuple(np.int32(kp[eye])), 3, (40, 30, 30), -1, cv2.LINE_AA)
    mouth = kp[KP.NOSE] - up * 9
    cv2.circle(img, tuple(np.int32(mouth)), 3, (80, 80, 160), -1, cv2.LINE_AA)
    # braços: manga + antebraço
    for sho, elb, wri in ((KP.L_SHOULDER, KP.L_ELBOW, KP.L_WRIST), (KP.R_SHOULDER, KP.R_ELBOW, KP.R_WRIST)):
        mid = kp[sho] + (kp[elb] - kp[sho]) * 0.55
        _limb(img, kp[sho], mid, SHIRT, 22)
        _limb(img, mid, kp[elb], SKIN, 17)
        _limb(img, kp[elb], kp[wri], SKIN, 15)
        cv2.circle(img, tuple(np.int32(kp[wri])), 10, SKIN_DARK, -1, cv2.LINE_AA)
    cv2.putText(img, f"CAM-04  t={t:05.1f}s", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def generate_session(out_dir: Path, script: SessionScript | None = None, fps: int = FPS, normal: bool = False) -> dict:
    """Gera o vídeo da sessão, keypoints verdadeiros e eventos esperados.

    `normal=True` gera `sessao_controle.mp4`, sem anomalias, para medir falsos positivos.
    """
    script = script or SessionScript()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "sessao_controle" if normal else "sessao_fisioterapia"
    video_path = out_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    bg = _background()
    cv2.imwrite(str(out_dir / "sala_vazia.png"), bg)  # quadro de referência da câmera fixa
    n = int(script.duration * fps)
    keypoints = np.zeros((n, 17, 2), np.float32)
    for i in range(n):
        t = i / fps
        kp, _ = pose_at(t, normal)
        keypoints[i] = kp
        writer.write(render(kp, bg, t))
    writer.release()
    np.save(out_dir / f"{stem}_keypoints.npy", keypoints)
    meta = {
        "video": video_path.name,
        "fps": fps,
        "frames": n,
        "resolucao": [WIDTH, HEIGHT],
        "referencia": "sala_vazia.png",
        "zonas": [
            {"nome": "tapete_exercicio", "tipo": "segura", "bbox": list(MAT)},
            {"nome": "piso", "tipo": "proibida", "bbox": [0, 330, WIDTH, HEIGHT]},
        ],
        "segmentos": [s.__dict__ for s in script.segments],
        "eventos_esperados": [] if normal else script.events,
    }
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
