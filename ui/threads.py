"""Background worker threads for training and camera capture to keep PySide6 UI responsive."""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from train import train_model
from inference import MaskInferencePipeline

logger = logging.getLogger(__name__)


class TrainingWorker(QThread):
    """Executes model training in a background thread."""

    progress_updated = Signal(dict)
    log_received = Signal(str)
    training_finished = Signal(dict)
    training_failed = Signal(str)

    def __init__(
        self,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 4,
        img_size: int = 224,
        checkpoint_dir: str = "checkpoints",
        model_save_name: str = "mobilenet_v2_mask_best.pt",
        resume_checkpoint: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.img_size = img_size
        self.checkpoint_dir = checkpoint_dir
        self.model_save_name = model_save_name
        self.resume_checkpoint = resume_checkpoint
        self._is_stopped = False

    def stop(self) -> None:
        """Signal thread to stop training loop."""
        self._is_stopped = True

    def run(self) -> None:
        try:
            self._is_stopped = False
            results = train_model(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                early_stopping_patience=self.early_stopping_patience,
                img_size=self.img_size,
                checkpoint_dir=self.checkpoint_dir,
                model_save_name=self.model_save_name,
                resume_checkpoint=self.resume_checkpoint,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
                stop_requested=lambda: self._is_stopped,
            )
            self.training_finished.emit(results)
        except Exception as e:
            logger.exception("Training worker thread encountered an error.")
            self.training_failed.emit(str(e))

    def _on_progress(self, data: Dict[str, Any]) -> None:
        self.progress_updated.emit(data)

    def _on_log(self, msg: str) -> None:
        self.log_received.emit(msg)


class CameraWorker(QThread):
    """Captures and processes live webcam frames in background thread."""

    frame_ready = Signal(QImage, float, int)  # (qimage, fps, detected_faces_count)
    camera_error = Signal(str)

    def __init__(
        self,
        pipeline: MaskInferencePipeline,
        camera_index: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.pipeline = pipeline
        self.camera_index = camera_index
        self._running = False
        self.cap: Optional[cv2.VideoCapture] = None

    def set_camera_index(self, index: int) -> None:
        self.camera_index = index

    def stop(self) -> None:
        self._running = False
        self.wait(1000)

    def run(self) -> None:
        self._running = True
        logger.info(f"Opening camera index {self.camera_index}...")

        # Open video capture with DirectShow backend for faster start on Windows
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Fallback to MSMF backend
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            msg = f"Could not open camera at index {self.camera_index}."
            if self.camera_index != 0:
                msg += "\nPlease switch the Camera dropdown to 'Camera 0 (Default Webcam)'."
            else:
                msg += "\nPlease verify webcam privacy permissions in Windows Settings (Settings > Privacy & security > Camera)."
            self.camera_error.emit(msg)
            self._running = False
            return

        # Attempt setting standard resolution (640x480 for smooth FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Standard 30 FPS target pacing to prevent 100% CPU lock
        target_frame_time = 1.0 / 30.0

        while self._running:
            loop_start = time.perf_counter()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            try:
                # Mirror frame horizontally for natural webcam feel
                frame = cv2.flip(frame, 1)

                # Process frame through inference pipeline
                annotated_frame, detections, fps = self.pipeline.process_frame(frame)

                # Convert OpenCV BGR frame to QImage for PySide6
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

                self.frame_ready.emit(q_img, fps, len(detections))
            except Exception as e:
                logger.warning(f"Error in camera loop: {e}")

            # Smart pacing sleep to avoid 100% CPU saturation
            elapsed = time.perf_counter() - loop_start
            sleep_needed = target_frame_time - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)
            else:
                time.sleep(0.002)

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera worker thread terminated cleanly.")
