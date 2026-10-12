"""
Face Recognition System Core Package.
Provides deep learning face detection, feature extraction, database matching, and annotation.
"""

from .detector import FaceDetector
from .recognizer import FaceRecognizer
from .database import FaceDatabase
from .visualizer import Visualizer
from .pipeline import RecognitionPipeline

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "FaceDatabase",
    "Visualizer",
    "RecognitionPipeline",
]
