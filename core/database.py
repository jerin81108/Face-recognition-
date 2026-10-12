"""
Face Database Manager.
Maintains known identity roster, extracts embeddings, manages persistent caching,
and performs vector nearest-neighbor matching against query features.
"""

import os
import glob
import pickle
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .detector import FaceDetector
from .recognizer import FaceRecognizer


class FaceDatabase:
    def __init__(
        self,
        db_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        detector: Optional[FaceDetector] = None,
        recognizer: Optional[FaceRecognizer] = None,
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_dir = db_dir or os.path.join(base_dir, "data", "known_faces")
        self.cache_path = cache_path or os.path.join(base_dir, "data", "embeddings_cache.pkl")
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        self.detector = detector or FaceDetector()
        self.recognizer = recognizer or FaceRecognizer()

        # In-memory storage:
        # {person_name: [{"path": str, "mtime": float, "feature": np.ndarray}, ...]}
        self.records: Dict[str, List[Dict[str, Any]]] = {}
        self.load_database()

    def _load_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                print(f"[FaceDatabase] Cache read error: {e}. Will rebuild cache.")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.records, f)
        except Exception as e:
            print(f"[FaceDatabase] Cache write error: {e}")

    def load_database(self) -> int:
        """
        Scan db_dir, detect new/modified photos, compute embeddings, and update cache.
        Returns total number of registered individuals.
        """
        cache = self._load_cache()
        updated_records: Dict[str, List[Dict[str, Any]]] = {}

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        items = os.listdir(self.db_dir)

        # 1. Directory-per-person format: data/known_faces/<Person_Name>/...
        for item in items:
            item_path = os.path.join(self.db_dir, item)
            if os.path.isdir(item_path):
                person_name = item.strip()
                updated_records[person_name] = []
                cached_items = {r["path"]: r for r in cache.get(person_name, [])}

                for file_name in os.listdir(item_path):
                    _, ext = os.path.splitext(file_name)
                    if ext.lower() not in valid_exts:
                        continue

                    file_path = os.path.join(item_path, file_name)
                    mtime = os.path.getmtime(file_path)

                    # Check if already cached with matching mtime
                    if file_path in cached_items and abs(cached_items[file_path].get("mtime", 0) - mtime) < 1e-4:
                        updated_records[person_name].append(cached_items[file_path])
                    else:
                        feat = self._extract_feature_from_file(file_path)
                        if feat is not None:
                            entry = {"path": file_path, "mtime": mtime, "feature": feat}
                            updated_records[person_name].append(entry)

            # 2. Single-image format: data/known_faces/<Person_Name>.jpg
            elif os.path.isfile(item_path):
                name, ext = os.path.splitext(item)
                if ext.lower() in valid_exts:
                    person_name = name.strip()
                    if person_name not in updated_records:
                        updated_records[person_name] = []

                    mtime = os.path.getmtime(item_path)
                    cached_items = {r["path"]: r for r in cache.get(person_name, [])}

                    if item_path in cached_items and abs(cached_items[item_path].get("mtime", 0) - mtime) < 1e-4:
                        updated_records[person_name].append(cached_items[item_path])
                    else:
                        feat = self._extract_feature_from_file(item_path)
                        if feat is not None:
                            entry = {"path": item_path, "mtime": mtime, "feature": feat}
                            updated_records[person_name].append(entry)

        # Prune empty persons
        self.records = {k: v for k, v in updated_records.items() if len(v) > 0}
        self._save_cache()
        print(f"[FaceDatabase] Loaded {len(self.records)} persons with {sum(len(v) for v in self.records.values())} total face samples.")
        return len(self.records)

    def _extract_feature_from_file(self, file_path: str) -> Optional[np.ndarray]:
        """Read image from file, detect face, and extract embedding."""
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
            faces = self.detector.detect(img)
            if not faces:
                print(f"[FaceDatabase] Warning: No face detected in {file_path}")
                return None

            # If multiple faces detected in registration image, pick the largest one
            largest_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            feature = self.recognizer.extract_feature(img, largest_face["raw"])
            return feature
        except Exception as e:
            print(f"[FaceDatabase] Error processing {file_path}: {e}")
            return None

    def add_person(self, name: str, image_bytes_or_mat, filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Enroll a new person or add photo to existing person.
        :param name: Person's full name
        :param image_bytes_or_mat: Image as numpy array (BGR) or raw bytes
        :param filename: Optional specific filename
        :return: (success: bool, message: str)
        """
        clean_name = name.strip()
        if not clean_name:
            return False, "Person name cannot be empty."

        person_dir = os.path.join(self.db_dir, clean_name)
        os.makedirs(person_dir, exist_ok=True)

        if isinstance(image_bytes_or_mat, bytes):
            nparr = np.frombuffer(image_bytes_or_mat, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_bytes_or_mat, np.ndarray):
            img = image_bytes_or_mat
        else:
            return False, "Unsupported image format."

        if img is None or img.size == 0:
            return False, "Could not decode image data."

        # Verify a face is detectable
        faces = self.detector.detect(img)
        if not faces:
            return False, "No face was detected in the provided image. Please ensure good lighting and front-facing angle."

        largest_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
        feat = self.recognizer.extract_feature(img, largest_face["raw"])
        if feat is None:
            return False, "Failed to extract facial features from the image."

        # Save to file
        if not filename:
            filename = f"sample_{int(time.time() * 1000)}.jpg"
        target_path = os.path.join(person_dir, filename)
        cv2.imwrite(target_path, img)
        mtime = os.path.getmtime(target_path)

        if clean_name not in self.records:
            self.records[clean_name] = []

        self.records[clean_name].append({
            "path": target_path,
            "mtime": mtime,
            "feature": feat
        })
        self._save_cache()

        return True, f"Successfully enrolled '{clean_name}' with 1 new sample."

    def delete_person(self, name: str) -> bool:
        """Delete an enrolled person and all their photos."""
        clean_name = name.strip()
        if clean_name in self.records:
            # Delete files and directory
            person_dir = os.path.join(self.db_dir, clean_name)
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir, ignore_errors=True)

            # Also check single file
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                single_path = os.path.join(self.db_dir, f"{clean_name}{ext}")
                if os.path.exists(single_path):
                    os.remove(single_path)

            del self.records[clean_name]
            self._save_cache()
            return True
        return False

    def get_persons(self) -> List[Dict[str, Any]]:
        """Return list of enrolled persons with metadata."""
        output = []
        for name, samples in sorted(self.records.items()):
            latest_mtime = max((s.get("mtime", 0) for s in samples), default=0)
            sample_path = samples[0]["path"] if samples else None
            output.append({
                "name": name,
                "samples_count": len(samples),
                "sample_path": sample_path,
                "updated_at": latest_mtime,
            })
        return output

    def find_match(self, query_feature: np.ndarray, threshold: Optional[float] = None) -> Tuple[str, float, float, bool]:
        """
        Compare query_feature against all enrolled persons.
        :param query_feature: 128-D vector
        :param threshold: Cosine cutoff (defaults to recognizer threshold)
        :return: (name: str, similarity: float, confidence: float, is_known: bool)
        """
        if query_feature is None or not self.records:
            return "Unknown", 0.0, 0.0, False

        cutoff = threshold if threshold is not None else self.recognizer.cosine_threshold

        best_name = "Unknown"
        best_similarity = -1.0

        for name, samples in self.records.items():
            for sample in samples:
                cached_feat = sample.get("feature")
                if cached_feat is None:
                    continue
                sim = self.recognizer.compute_similarity(query_feature, cached_feat)
                if sim > best_similarity:
                    best_similarity = sim
                    best_name = name

        is_known = best_similarity >= cutoff
        confidence = self.recognizer.similarity_to_confidence(best_similarity)

        if not is_known:
            return "Unknown", best_similarity, confidence, False

        return best_name, best_similarity, confidence, True
