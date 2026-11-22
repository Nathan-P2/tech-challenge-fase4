"""Backends de estimativa de pose.

Todos retornam keypoints no formato COCO-17 `(17, 2)` e confiança `(17,)`.

- `YoloPoseBackend`: YOLOv8-pose (Ultralytics), padrão do projeto.
- `OpenPoseBackend`: OpenPose COCO-18 via OpenCV DNN (requer os pesos Caffe em
  `models/openpose/`; ver README). Convertido para COCO-17.
- `GroundTruthBackend`: keypoints conhecidos do vídeo sintético, com ruído
  opcional. Usado em testes e como referência de avaliação.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .keypoints import OPENPOSE18_TO_COCO17


class PoseBackend(Protocol):
    name: str

    def __call__(self, frame: np.ndarray, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        ...


class YoloPoseBackend:
    name = "yolov8-pose"

    def __init__(self, weights: str = "yolov8n-pose.pt", conf: float = 0.25, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def __call__(self, frame, index):
        result = self.model(frame, verbose=False, conf=self.conf, device=self.device)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return None, None
        best = int(result.boxes.conf.argmax())
        xy = result.keypoints.xy[best].cpu().numpy()
        conf = result.keypoints.conf
        conf = conf[best].cpu().numpy() if conf is not None else np.ones(17)
        return xy.astype(np.float32), conf.astype(np.float32)


class OpenPoseBackend:
    """OpenPose (CMU) COCO via `cv2.dnn`.

    Arquivos esperados (download oficial do projeto OpenPose):
    `models/openpose/pose_deploy_linevec.prototxt` e
    `models/openpose/pose_iter_440000.caffemodel`.
    Implementação single-person: máximo de cada mapa de calor.
    """

    name = "openpose-coco"

    def __init__(self, model_dir: str | Path = "models/openpose", in_size: int = 368, threshold: float = 0.1):
        import cv2

        model_dir = Path(model_dir)
        proto = model_dir / "pose_deploy_linevec.prototxt"
        weights = model_dir / "pose_iter_440000.caffemodel"
        if not proto.exists() or not weights.exists():
            raise FileNotFoundError(
                f"Pesos do OpenPose não encontrados em {model_dir}. Rode scripts/download_openpose.sh "
                "ou use --pose-backend yolo."
            )
        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.in_size = in_size
        self.threshold = threshold

    def __call__(self, frame, index):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.in_size, self.in_size), (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 57, H, W): 18 keypoints + fundo + PAFs
        xy = np.zeros((17, 2), np.float32)
        conf = np.zeros(17, np.float32)
        for op_idx, coco_idx in OPENPOSE18_TO_COCO17.items():
            heat = out[0, op_idx]
            _, prob, _, point = cv2.minMaxLoc(heat)
            xy[coco_idx] = (w * point[0] / out.shape[3], h * point[1] / out.shape[2])
            conf[coco_idx] = prob
        if (conf > self.threshold).sum() < 5:
            return None, None
        return xy, conf


class GroundTruthBackend:
    name = "ground-truth"

    def __init__(self, keypoints: np.ndarray, noise_px: float = 0.0, seed: int = 0, hide_lying: bool = False):
        self.keypoints = keypoints
        self.noise = noise_px
        self.rng = np.random.default_rng(seed)
        self.hide_lying = hide_lying

    def __call__(self, frame, index):
        if index >= len(self.keypoints):
            return None, None
        kp = self.keypoints[index].copy()
        if self.hide_lying:
            span = kp.max(0) - kp.min(0)
            if span[0] > 1.3 * span[1]:
                return None, None
        if self.noise:
            kp += self.rng.normal(0, self.noise, kp.shape)
        return kp.astype(np.float32), np.ones(17, np.float32)


def make_backend(name: str, **kwargs) -> PoseBackend:
    name = name.lower()
    if name in {"yolo", "yolov8", "yolov8-pose"}:
        return YoloPoseBackend(**kwargs)
    if name == "openpose":
        return OpenPoseBackend(**kwargs)
    if name in {"gt", "ground-truth", "groundtruth"}:
        return GroundTruthBackend(**kwargs)
    raise ValueError(f"Backend de pose desconhecido: {name}")
