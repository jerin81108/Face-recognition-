"""
Face Recognition Pipeline & Attendance Logger.
Coordinates detection, feature extraction, database lookup, visualization, and event logging.
"""

import os
import time
import json
import csv
from datetime import datetime
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .database import FaceDatabase
from .visualizer import Visualizer


class RecognitionPipeline:
    def __init__(
        self,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
        database: Optional[FaceDatabase] = None,
        log_dir: Optional[str] = None,
        log_cooldown_seconds: float = 8.0,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_dir = log_dir or os.path.join(base_dir, "data", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.json_log_path = os.path.join(self.log_dir, "detections.json")
        self.csv_log_path = os.path.join(self.log_dir, "attendance.csv")

        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()
        self.database = database or FaceDatabase(detector=self.detector, recognizer=self.recognizer)

        self.log_cooldown_seconds = log_cooldown_seconds
        self.last_logged_times: Dict[str, float] = {}

        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Date", "Time", "Person", "Confidence", "Similarity", "Status"])

    def process_image(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        annotate: bool = True,
        show_landmarks: bool = True,
        log_events: bool = True,
    ) -> Dict[str, Any]:
        """
        Full pipeline: Detect -> Recognize -> Log -> Annotate.
        :param image: BGR numpy image
        :param threshold: Optional override for cosine similarity cutoff
        :param annotate: Whether to generate visual overlay
        :param show_landmarks: Whether to show landmark dots
        :param log_events: Whether to record detection events to log files
        """
        start_time = time.time()
        if image is None or image.size == 0:
            return {
                "faces": [],
                "count": 0,
                "latency_ms": 0.0,
                "annotated_image": None,
            }

        # 1. Detect faces
        detected_faces = self.detector.detect(image)
        results = []

        now = time.time()
        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for face in detected_faces:
            raw_face = face["raw"]
            bbox = face["bbox"]
            landmarks = face["landmarks"]
            det_conf = face["confidence"]

            # 2. Extract 128-D feature embedding
            feature = self.recognizer.extract_feature(image, raw_face)

            # 3. Match against database
            name, similarity, confidence, is_known = self.database.find_match(feature, threshold=threshold)

            # Prepare thumbnail crop for log
            x, y, w, h = bbox
            h_img, w_img = image.shape[:2]
            crop = image[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]
            thumb_b64 = ""
            if crop.size > 0:
                thumb_b64 = Visualizer.to_base64_jpeg(crop, quality=70)

            face_res = {
                "name": name,
                "is_known": is_known,
                "confidence": confidence,
                "similarity": similarity,
                "detection_confidence": det_conf,
                "bbox": bbox,
                "landmarks": landmarks,
                "thumbnail": thumb_b64,
                "timestamp": dt_str,
            }
            results.append(face_res)

            # 4. Log event (with cooldown for repeated detection of same identity)
            if log_events and is_known:
                last_time = self.last_logged_times.get(name, 0)
                if (now - last_time) >= self.log_cooldown_seconds:
                    self._record_log_entry(face_res)
                    self.last_logged_times[name] = now

        # 5. Annotate image
        annotated_img = None
        if annotate:
            annotated_img = Visualizer.annotate_frame(image, results, show_landmarks=show_landmarks)

        latency_ms = round((time.time() - start_time) * 1000, 1)

        return {
            "faces": results,
            "count": len(results),
            "latency_ms": latency_ms,
            "annotated_image": annotated_img,
        }

    def _record_log_entry(self, entry: Dict[str, Any]):
        """Persist detection record to CSV and JSON."""
        dt = datetime.now()
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M:%S")
        ts_full = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Append CSV
        try:
            with open(self.csv_log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ts_full,
                    date_str,
                    time_str,
                    entry["name"],
                    f"{entry['confidence']:.1f}%",
                    f"{entry['similarity']:.3f}",
                    "Verified" if entry["is_known"] else "Unknown",
                ])
        except Exception as e:
            print(f"[Pipeline] CSV log error: {e}")

        # Append JSON
        try:
            logs = []
            if os.path.exists(self.json_log_path):
                try:
                    with open(self.json_log_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except Exception:
                    logs = []

            log_item = {
                "timestamp": ts_full,
                "date": date_str,
                "time": time_str,
                "name": entry["name"],
                "confidence": entry["confidence"],
                "similarity": entry["similarity"],
                "is_known": entry["is_known"],
                "thumbnail": entry.get("thumbnail", ""),
            }
            logs.insert(0, log_item)
            # Cap at 1000 entries
            logs = logs[:1000]

            with open(self.json_log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[Pipeline] JSON log error: {e}")

    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve latest detection logs."""
        if os.path.exists(self.json_log_path):
            try:
                with open(self.json_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data[:limit]
            except Exception as e:
                print(f"[Pipeline] Read logs error: {e}")
        return []

    def clear_logs(self):
        """Clear detection logs."""
        if os.path.exists(self.json_log_path):
            with open(self.json_log_path, "w", encoding="utf-8") as f:
                json.dump([], f)
        with open(self.csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Date", "Time", "Person", "Confidence", "Similarity", "Status"])
        self.last_logged_times.clear()
