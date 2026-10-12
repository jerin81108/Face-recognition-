"""
Face Detector using OpenCV YuNet Deep Learning Model with Haar Cascade Fallback.
Handles multi-resolution input, facial landmarks extraction, and bounding boxes.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class FaceDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.65,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ):
        """
        Initialize YuNet face detector.
        :param model_path: Path to face_detection_yunet_2023mar.onnx
        :param conf_threshold: Confidence threshold for face detection [0.0 - 1.0]
        :param nms_threshold: Non-maximum suppression threshold
        :param top_k: Maximum number of faces to detect
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.detector = None
        self.fallback_cascade = None
        self.last_input_size = None

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_detection_yunet_2023mar.onnx")

        self.model_path = model_path
        self._initialize_detector()

    def _initialize_detector(self):
        """Load YuNet model or fallback to OpenCV Haar Cascade."""
        if os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                # Initial dummy size (320, 320)
                self.detector = cv2.FaceDetectorYN.create(
                    model=self.model_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=self.top_k,
                )
                self.last_input_size = (320, 320)
                print(f"[FaceDetector] Loaded YuNet deep learning detector from {self.model_path}")
                return
            except Exception as e:
                print(f"[FaceDetector] Warning: Failed to initialize YuNet ({e}). Loading fallback.")

        # Fallback to Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.fallback_cascade = cv2.CascadeClassifier(cascade_path)
            print("[FaceDetector] Loaded Haar Cascade fallback detector.")
        else:
            print("[FaceDetector] Error: No face detector could be loaded.")

    def set_confidence(self, conf: float):
        """Update detection confidence threshold dynamically."""
        self.conf_threshold = max(0.1, min(0.99, conf))
        if self.detector is not None:
            self.detector.setScoreThreshold(self.conf_threshold)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image (BGR format).
        Returns list of dicts:
        {
            "bbox": [x, y, w, h],
            "landmarks": [[x, y], ...], # 5 keypoints: right eye, left eye, nose, right mouth, left mouth
            "confidence": float,
            "raw": np.ndarray (row vector for SFace alignment)
        }
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if self.detector is not None:
            # Update input size if changed
            if self.last_input_size != (w, h):
                self.detector.setInputSize((w, h))
                self.last_input_size = (w, h)

            try:
                ret, raw_faces = self.detector.detect(image)
                if ret != 1 or raw_faces is None or len(raw_faces) == 0:
                    return []

                results = []
                for face_row in raw_faces:
                    # YuNet face row format:
                    # [0:4]: x, y, w, h
                    # [4:14]: 5 landmarks (re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y)
                    # [14]: confidence score
                    x, y, box_w, box_h = face_row[0:4]
                    conf = float(face_row[14])

                    # Clamp bbox to image boundaries
                    x1 = max(0, int(round(x)))
                    y1 = max(0, int(round(y)))
                    x2 = min(w, int(round(x + box_w)))
                    y2 = min(h, int(round(y + box_h)))
                    clamped_w = max(1, x2 - x1)
                    clamped_h = max(1, y2 - y1)

                    landmarks = []
                    for idx in range(4, 14, 2):
                        lx = int(round(face_row[idx]))
                        ly = int(round(face_row[idx + 1]))
                        landmarks.append([lx, ly])

                    results.append({
                        "bbox": [x1, y1, clamped_w, clamped_h],
                        "landmarks": landmarks,
                        "confidence": round(conf, 4),
                        "raw": face_row,
                    })

                return results
            except Exception as e:
                print(f"[FaceDetector] Detection error: {e}")

        # Fallback using Haar Cascade
        if self.fallback_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.fallback_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            results = []
            for (x, y, box_w, box_h) in faces:
                # Estimate dummy landmarks for compatibility
                re = [int(x + box_w * 0.3), int(y + box_h * 0.4)]
                le = [int(x + box_w * 0.7), int(y + box_h * 0.4)]
                nt = [int(x + box_w * 0.5), int(y + box_h * 0.6)]
                rcm = [int(x + box_w * 0.35), int(y + box_h * 0.8)]
                lcm = [int(x + box_w * 0.65), int(y + box_h * 0.8)]
                raw_dummy = np.array([
                    x, y, box_w, box_h,
                    re[0], re[1], le[0], le[1], nt[0], nt[1],
                    rcm[0], rcm[1], lcm[0], lcm[1],
                    0.85
                ], dtype=np.float32)

                results.append({
                    "bbox": [int(x), int(y), int(box_w), int(box_h)],
                    "landmarks": [re, le, nt, rcm, lcm],
                    "confidence": 0.85,
                    "raw": raw_dummy,
                })
            return results

        return []
