"""Face Detection module for real-time video stream using OpenCV.

Provides offline, zero-download face detection using OpenCV's bundled Haar Cascade,
with optional YuNet DNN detection support when ONNX weights are placed in models/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces in video frames using OpenCV."""

    def __init__(
        self,
        method: str = "haar",
        min_face_size: Tuple[int, int] = (40, 40),
        yunet_model_path: Optional[str | Path] = None,
    ):
        """Args:

        method: 'haar' (default bundled) or 'yunet' (OpenCV DNN)
        min_face_size: Minimum width and height of faces to consider.
        yunet_model_path: Optional path to YuNet ONNX model.
        """
        self.method = method.lower()
        self.min_face_size = min_face_size
        self.cascade: Optional[cv2.CascadeClassifier] = None
        self.yunet = None

        if self.method == "haar":
            self._init_haar()
        elif self.method == "yunet":
            if yunet_model_path and Path(yunet_model_path).exists():
                self._init_yunet(yunet_model_path)
            else:
                logger.warning("YuNet model path not found, falling back to bundled Haar Cascade.")
                self.method = "haar"
                self._init_haar()
        else:
            self._init_haar()

    def _init_haar(self) -> None:
        """Initialize OpenCV bundled Haar Cascade classifier."""
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(haar_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {haar_path}")
        logger.info("Initialized OpenCV Haar Cascade face detector.")

    def _init_yunet(self, model_path: str | Path) -> None:
        """Initialize OpenCV YuNet DNN face detector."""
        try:
            # OpenCV 4.5.4+ includes FaceDetectorYN
            self.yunet = cv2.FaceDetectorYN.create(
                model=str(model_path),
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            logger.info("Initialized OpenCV YuNet DNN face detector.")
        except Exception as e:
            logger.warning(f"Failed to initialize YuNet ({e}). Falling back to Haar Cascade.")
            self.method = "haar"
            self._init_haar()

    def detect_faces(
        self,
        frame: np.ndarray,
        expand_ratio: float = 0.15,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a BGR image frame.

        Args:
            frame: OpenCV BGR image (uint8).
            expand_ratio: Padding ratio to expand bbox for full face/mask context.

        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        if self.method == "haar" and self.cascade is not None:
            # Downsample 2x for 4x faster detection on CPU
            scale = 0.5
            small_w = max(1, int(w * scale))
            small_h = max(1, int(h * scale))
            small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            scaled_min_size = (
                max(20, int(self.min_face_size[0] * scale)),
                max(20, int(self.min_face_size[1] * scale)),
            )

            raw_boxes = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=scaled_min_size,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (sbx, sby, sbw, sbh) in raw_boxes:
                # Scale back up to original frame dimensions
                bx = int(sbx / scale)
                by = int(sby / scale)
                bw = int(sbw / scale)
                bh = int(sbh / scale)

                # Add padding to encompass chin and forehead
                pad_x = int(bw * expand_ratio)
                pad_y = int(bh * expand_ratio)

                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)

                boxes.append((x1, y1, x2 - x1, y2 - y1))

        elif self.method == "yunet" and self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is not None:
                for face in faces:
                    bx, by, bw, bh = face[0:4].astype(int)
                    pad_x = int(bw * expand_ratio)
                    pad_y = int(bh * expand_ratio)
                    x1 = max(0, bx - pad_x)
                    y1 = max(0, by - pad_y)
                    x2 = min(w, bx + bw + pad_x)
                    y2 = min(h, by + bh + pad_y)
                    boxes.append((x1, y1, x2 - x1, y2 - y1))

        return boxes
