"""
PapillaDetector — YOLOv8-based papilla detection with AccessState axis mapping.

Axis mapping (matches ercp_access project convention):
    e_x = (cx_px - W/2) / (W/2)      right positive
    e_y = -(cy_px - H/2) / (H/2)     up positive (y-axis flipped)
    scale = (box_w * box_h) / (W * H) normalized area
    conf  = YOLO detection confidence

No-detection returns: (0.0, 0.0, 0.0, 0.0)

Single-class detector (class 0 = papilla).
When multiple detections exist, the highest-confidence detection is used.
"""
from __future__ import annotations

import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")


class PapillaDetector:
    """
    Wraps a trained YOLOv8 model for single-class papilla detection.

    Parameters
    ----------
    weights : str
        Path to trained YOLOv8 .pt weights file.
    conf_thresh : float
        Minimum confidence threshold for detection (default 0.25).
    device : str
        Inference device: 'cuda', 'cpu', or '0', '1', etc. (default 'cuda').
    """

    def __init__(
        self,
        weights: str,
        conf_thresh: float = 0.25,
        device: str = "cuda",
    ):
        from ultralytics import YOLO
        self._model = YOLO(weights)
        self._conf_thresh = conf_thresh
        self._device = device

    def detect(
        self,
        rgb: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Run detection on a single RGB frame.

        Parameters
        ----------
        rgb : np.ndarray
            Image in CHW float32 [0,1] OR HWC uint8/float32.
            Accepted shapes: (3, H, W) or (H, W, 3).

        Returns
        -------
        (e_x, e_y, scale, conf) : tuple of float
            All normalized. Returns (0.0, 0.0, 0.0, 0.0) if no detection.

        Axis mapping:
            e_x   = (cx_px - W/2) / (W/2)    right positive
            e_y   = -(cy_px - H/2) / (H/2)   up positive
            scale = (box_w * box_h) / (W * H)
            conf  = YOLO confidence score
        """
        img = self._preprocess(rgb)

        results = self._model.predict(
            img,
            conf=self._conf_thresh,
            device=self._device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Pick highest-confidence detection
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))

        # xywhn: normalized [cx, cy, w, h] in [0,1]
        xywhn = boxes.xywhn.cpu().numpy()[best_idx]  # shape (4,)
        cx_n, cy_n, bw_n, bh_n = xywhn

        # Image dimensions (used for pixel-space axis mapping)
        # Since we work in normalized coords, H=W=1 effectively
        # e_x: rightward positive; center is 0.5 in normalized coords
        e_x = (cx_n - 0.5) / 0.5          # maps [0,1] -> [-1,1]
        e_y = -((cy_n - 0.5) / 0.5)       # flipped: up is positive

        scale = float(bw_n * bh_n)         # normalized bbox area
        conf  = float(confidences[best_idx])

        return float(e_x), float(e_y), float(scale), float(conf)

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert input to HWC uint8 for YOLO.

        Accepts:
            - CHW float32 [0,1]  -> HWC uint8 [0,255]
            - HWC float32 [0,1]  -> HWC uint8 [0,255]
            - HWC uint8           -> pass through
        """
        arr = np.asarray(rgb)

        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[0] < arr.shape[1]:
            # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

        return arr

    def __repr__(self) -> str:
        return (
            f"PapillaDetector(conf_thresh={self._conf_thresh}, device={self._device!r})"
        )
