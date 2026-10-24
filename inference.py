"""Real-time Face Mask Detection and Inference Pipeline.

Connects OpenCV face detector with MobileNetV2 classifier, applies prediction smoothing,
draws bounding boxes and labels, computes live FPS, and optionally logs privacy-friendly events.
"""

from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from detector import FaceDetector
from models.model_builder import load_checkpoint, DEFAULT_CLASS_NAMES

logger = logging.getLogger(__name__)

# BGR color constants
COLOR_GREEN = (0, 255, 0)       # With Mask
COLOR_RED = (0, 0, 255)         # Without Mask
COLOR_ORANGE = (0, 165, 255)     # Incorrect Mask
COLOR_GRAY = (128, 128, 128)     # Low confidence / unclassified

CLASS_COLORS = {
    0: COLOR_ORANGE,  # mask_weared_incorrect
    1: COLOR_GREEN,   # with_mask
    2: COLOR_RED,     # without_mask
}

DISPLAY_NAMES = {
    0: "Incorrect Mask",
    1: "With Mask",
    2: "Without Mask",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PredictionSmoother:
    """Smooth predictions across frames for detected face boxes to prevent flicker."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Tracks list of dicts: {'box': (x, y, w, h), 'history': deque([(class_idx, conf), ...]), 'last_seen': time}
        self.tracks: List[Dict[str, Any]] = []

    def update(
        self,
        detected_boxes: List[Tuple[int, int, int, int]],
        raw_predictions: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """Match detected boxes with existing tracks and return smoothed predictions."""
        now = time.time()
        smoothed_results: List[Tuple[int, float]] = []

        # Remove stale tracks older than 1.0 second
        self.tracks = [t for t in self.tracks if (now - t["last_seen"]) < 1.0]

        matched_track_indices = set()

        for box, (c_idx, conf) in zip(detected_boxes, raw_predictions):
            bx, by, bw, bh = box
            b_cx, b_cy = bx + bw / 2, by + bh / 2

            best_match_idx = None
            best_dist = float("inf")

            for idx, track in enumerate(self.tracks):
                if idx in matched_track_indices:
                    continue
                tx, ty, tw, th = track["box"]
                t_cx, t_cy = tx + tw / 2, ty + th / 2
                dist = np.hypot(b_cx - t_cx, b_cy - t_cy)
                # Match if centers are within half the box dimension
                max_allowable_dist = max(bw, bh) * 0.75
                if dist < max_allowable_dist and dist < best_dist:
                    best_dist = dist
                    best_match_idx = idx

            if best_match_idx is not None:
                matched_track_indices.add(best_match_idx)
                track = self.tracks[best_match_idx]
                track["box"] = box
                track["last_seen"] = now
                track["history"].append((c_idx, conf))
            else:
                new_track = {
                    "box": box,
                    "history": deque([(c_idx, conf)], maxlen=self.window_size),
                    "last_seen": now,
                }
                self.tracks.append(new_track)
                track = new_track

            # Compute smoothed prediction: weighted average confidence across classes
            class_scores: Dict[int, List[float]] = {}
            for hist_c, hist_conf in track["history"]:
                class_scores.setdefault(hist_c, []).append(hist_conf)

            # Pick class with highest average confidence * occurrence count
            best_c = c_idx
            best_avg_conf = conf
            best_weight = -1.0

            for c_key, conf_list in class_scores.items():
                weight = len(conf_list) * (sum(conf_list) / len(conf_list))
                if weight > best_weight:
                    best_weight = weight
                    best_c = c_key
                    best_avg_conf = sum(conf_list) / len(conf_list)

            smoothed_results.append((best_c, best_avg_conf))

        return smoothed_results


class MaskInferencePipeline:
    """Real-time face detection and mask classification inference engine."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        confidence_threshold: float = 0.60,
        smoothing_window: int = 5,
        face_detector_method: str = "haar",
        log_file: Optional[str | Path] = "outputs/logs/detection_events.csv",
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.detector = FaceDetector(method=face_detector_method)
        self.smoother = PredictionSmoother(window_size=smoothing_window)
        self.log_file = Path(log_file) if log_file else None

        self.model: Optional[torch.nn.Module] = None
        self.metadata: Dict[str, Any] = {}
        self.is_model_loaded = False

        self.preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # FPS calculation state
        self.prev_frame_time = 0.0
        self.fps = 0.0

        # Privacy logging toggle
        self.enable_logging = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: str | Path) -> bool:
        """Load trained MobileNetV2 checkpoint."""
        try:
            model, meta = load_checkpoint(checkpoint_path, device=self.device)
            self.model = model
            self.metadata = meta
            self.is_model_loaded = True
            logger.info(f"Loaded classifier model from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            self.model = None
            self.is_model_loaded = False
            return False

    def log_event(self, class_name: str, confidence: float) -> None:
        """Privacy-friendly logger: Saves ONLY timestamp, class, and confidence."""
        if not self.enable_logging or self.log_file is None:
            return

        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.log_file.exists()
            with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "predicted_class", "confidence"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    class_name,
                    f"{confidence:.4f}",
                ])
        except Exception as e:
            logger.warning(f"Failed to write privacy detection log: {e}")

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """Process a single video frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            Tuple of (annotated_frame, list_of_detections, fps)
        """
        if frame is None or frame.size == 0:
            return frame, [], 0.0

        # FPS calculation
        curr_time = time.time()
        if self.prev_frame_time > 0:
            dt = curr_time - self.prev_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
        self.prev_frame_time = curr_time

        output_frame = frame.copy()
        h, w = frame.shape[:2]

        # 1. Detect faces
        faces = self.detector.detect_faces(frame)
        detections: List[Dict[str, Any]] = []

        if not self.is_model_loaded or self.model is None:
            # Draw detected faces with warning that model is not loaded
            for (x, y, fw, fh) in faces:
                cv2.rectangle(output_frame, (x, y), (x + fw, y + fh), COLOR_GRAY, 2)
                cv2.putText(
                    output_frame,
                    "Model not loaded",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_GRAY,
                    2,
                )
            self._draw_overlay_header(output_frame, "No classifier model loaded - train or select model first")
            return output_frame, detections, self.fps

        if not faces:
            self._draw_fps(output_frame)
            return output_frame, detections, self.fps

        # 2. Extract crops and classify
        raw_preds: List[Tuple[int, float]] = []
        valid_crops: List[Tuple[int, int, int, int]] = []

        for (x, y, fw, fh) in faces:
            crop = frame[y:y + fh, x:x + fw]
            if crop.size == 0:
                continue

            try:
                # Convert BGR to RGB PIL image for standard PyTorch preprocessing
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                input_tensor = self.preprocess_transform(pil_crop).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, dim=1)
                    raw_preds.append((int(pred_idx.item()), float(conf.item())))
                    valid_crops.append((x, y, fw, fh))
            except Exception as e:
                logger.error(f"Inference error on crop: {e}")

        # 3. Apply rolling window smoothing
        smoothed = self.smoother.update(valid_crops, raw_preds)

        # 4. Draw overlays
        for (x, y, fw, fh), (c_idx, conf) in zip(valid_crops, smoothed):
            if conf < self.confidence_threshold:
                # Below threshold: display in gray as uncertain
                label_text = f"Uncertain ({conf * 100:.1f}%)"
                color = COLOR_GRAY
            else:
                label_text = f"{DISPLAY_NAMES.get(c_idx, 'Unknown')} ({conf * 100:.1f}%)"
                color = CLASS_COLORS.get(c_idx, COLOR_GRAY)
                # Log event if enabled
                self.log_event(DISPLAY_NAMES.get(c_idx, "Unknown"), conf)

            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + fw, y + fh), color, 2)

            # Draw badge background for text readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(0, y - text_h - 10)
            badge_y2 = y
            badge_x1 = x
            badge_x2 = min(w, x + text_w + 10)

            cv2.rectangle(output_frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # Choose text color (black for light background, white for red)
            text_color = (0, 0, 0) if c_idx != 2 else (255, 255, 255)
            cv2.putText(
                output_frame,
                label_text,
                (badge_x1 + 5, badge_y2 - 5),
                font,
                font_scale,
                text_color,
                thickness,
            )

            detections.append({
                "box": (x, y, fw, fh),
                "class_idx": c_idx,
                "class_name": DISPLAY_NAMES.get(c_idx, "Unknown"),
                "confidence": conf,
            })

        self._draw_fps(output_frame)
        return output_frame, detections, self.fps

    def _draw_fps(self, frame: np.ndarray) -> None:
        """Draw FPS counter in top-right corner."""
        fps_text = f"FPS: {self.fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps_text, (frame.shape[1] - 130, 30), font, 0.7, (0, 255, 255), 2)

    def _draw_overlay_header(self, frame: np.ndarray, msg: str) -> None:
        """Draw top banner message."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(frame, msg, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
