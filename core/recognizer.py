"""
Face Recognizer using OpenCV SFace Deep Learning Model.
Extracts 128-D facial feature embeddings from aligned face crops and calculates similarity.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any


class FaceRecognizer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        cosine_threshold: float = 0.42,
    ):
        """
        Initialize SFace face recognizer.
        :param model_path: Path to face_recognition_sface_2021dec.onnx
        :param cosine_threshold: Cosine similarity cutoff for identity match (default 0.42)
        """
        self.cosine_threshold = cosine_threshold
        self.recognizer = None

        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_recognition_sface_2021dec.onnx")

        self.model_path = model_path
        self._initialize_recognizer()

    def _initialize_recognizer(self):
        """Load SFace deep learning recognizer."""
        if os.path.exists(self.model_path) and hasattr(cv2, "FaceRecognizerSF"):
            try:
                self.recognizer = cv2.FaceRecognizerSF.create(
                    model=self.model_path,
                    config="",
                )
                print(f"[FaceRecognizer] Loaded SFace deep learning model from {self.model_path}")
                return
            except Exception as e:
                print(f"[FaceRecognizer] Warning: SFace initialization failed: {e}")

        print("[FaceRecognizer] Warning: SFace model not found or unavailable. Fallback mode enabled.")

    def set_threshold(self, threshold: float):
        """Set cosine similarity threshold."""
        self.cosine_threshold = max(0.1, min(0.95, threshold))

    def align_and_crop(self, image: np.ndarray, face_raw: np.ndarray) -> Optional[np.ndarray]:
        """Align face according to 5 landmarks and crop to standard 112x112 frame."""
        if image is None or face_raw is None:
            return None

        if self.recognizer is not None:
            try:
                return self.recognizer.alignCrop(image, face_raw)
            except Exception as e:
                print(f"[FaceRecognizer] Alignment error: {e}")

        # Fallback simple crop if alignCrop fails
        x, y, w, h = [int(v) for v in face_raw[0:4]]
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (112, 112))
        return None

    def extract_feature(self, image: np.ndarray, face_raw: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 128-D embedding vector from detected face.
        """
        aligned_face = self.align_and_crop(image, face_raw)
        if aligned_face is None:
            return None

        if self.recognizer is not None:
            try:
                feature = self.recognizer.feature(aligned_face)
                # Ensure 1D flattened float32 vector
                feature = feature.flatten().astype(np.float32)
                # L2 normalize
                norm = np.linalg.norm(feature)
                if norm > 1e-6:
                    feature = feature / norm
                return feature
            except Exception as e:
                print(f"[FaceRecognizer] Feature extraction error: {e}")

        # Fallback simple grayscale resized histogram/feature
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 8)).flatten().astype(np.float32)
        norm = np.linalg.norm(resized)
        if norm > 1e-6:
            resized = resized / norm
        return resized

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature embeddings [-1.0 to 1.0].
        """
        if feat1 is None or feat2 is None:
            return 0.0

        f1 = feat1.flatten().astype(np.float32)
        f2 = feat2.flatten().astype(np.float32)

        if len(f1) != len(f2):
            return 0.0

        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        # Cosine similarity via dot product of normalized vectors
        similarity = float(np.dot(f1, f2) / (norm1 * norm2))
        return round(similarity, 4)

    def similarity_to_confidence(self, similarity: float) -> float:
        """
        Convert cosine similarity into intuitive percentage confidence [0% - 100%].
        If similarity == 1.0 -> 100%
        If similarity == threshold -> 50%
        If similarity < threshold -> scaled between 0% and 49%
        """
        thresh = self.cosine_threshold
        if similarity >= thresh:
            # Map [thresh, 1.0] -> [50.0, 100.0]
            if 1.0 - thresh > 1e-6:
                conf = 50.0 + ((similarity - thresh) / (1.0 - thresh)) * 50.0
            else:
                conf = 100.0
        else:
            # Map [0.0, thresh] -> [0.0, 49.9]
            if thresh > 1e-6:
                conf = max(0.0, (similarity / thresh) * 49.9)
            else:
                conf = 0.0
        return round(float(np.clip(conf, 0.0, 100.0)), 1)
