"""Convenções de keypoints COCO-17 (compatível com OpenPose COCO e YOLOv8-pose)."""

from __future__ import annotations

import numpy as np


class KP:
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16


NAMES = [
    "nariz", "olho_esq", "olho_dir", "orelha_esq", "orelha_dir",
    "ombro_esq", "ombro_dir", "cotovelo_esq", "cotovelo_dir", "punho_esq", "punho_dir",
    "quadril_esq", "quadril_dir", "joelho_esq", "joelho_dir", "tornozelo_esq", "tornozelo_dir",
]

SKELETON = [
    (KP.L_SHOULDER, KP.R_SHOULDER), (KP.L_SHOULDER, KP.L_ELBOW), (KP.L_ELBOW, KP.L_WRIST),
    (KP.R_SHOULDER, KP.R_ELBOW), (KP.R_ELBOW, KP.R_WRIST), (KP.L_SHOULDER, KP.L_HIP),
    (KP.R_SHOULDER, KP.R_HIP), (KP.L_HIP, KP.R_HIP), (KP.L_HIP, KP.L_KNEE),
    (KP.L_KNEE, KP.L_ANKLE), (KP.R_HIP, KP.R_KNEE), (KP.R_KNEE, KP.R_ANKLE),
    (KP.NOSE, KP.L_EYE), (KP.NOSE, KP.R_EYE), (KP.L_EYE, KP.L_EAR), (KP.R_EYE, KP.R_EAR),
]

# Mapeamento OpenPose COCO-18 (com pescoço no índice 1) -> COCO-17.
OPENPOSE18_TO_COCO17 = {0: 0, 15: 2, 14: 1, 17: 4, 16: 3, 5: 5, 2: 6, 6: 7, 3: 8,
                        7: 9, 4: 10, 11: 11, 8: 12, 12: 13, 9: 14, 13: 15, 10: 16}


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo ABC em graus (vetorizado sobre a primeira dimensão)."""
    ba = a - b
    bc = c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
