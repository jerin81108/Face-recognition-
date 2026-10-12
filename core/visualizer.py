"""
High-aesthetic visualizer for face detection and recognition overlays.
Renders biometric HUD targeting brackets, gradient name badges, confidence meters,
and facial landmarks onto images and video frames.
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Tuple


class Visualizer:
    # Modern Color Palette in BGR
    COLOR_KNOWN_PRIMARY = (110, 231, 183)      # Emerald Mint #6ee7b7
    COLOR_KNOWN_ACCENT = (14, 165, 233)       # Sky Cyan #0ea5e9
    COLOR_UNKNOWN_PRIMARY = (99, 102, 241)    # Coral Red/Indigo #6366f1
    COLOR_UNKNOWN_ACCENT = (79, 70, 229)     # Deep Coral
    COLOR_LANDMARK = (250, 204, 21)           # Amber Glow #facc15
    COLOR_BG_DARK = (24, 24, 27)              # Zinc 900 #18181b
    COLOR_TEXT_WHITE = (244, 244, 245)        # Zinc 100

    @staticmethod
    def draw_hud_box(img: np.ndarray, bbox: List[int], color: Tuple[int, int, int], corner_len: int = 14, thickness: int = 2):
        """Draws a sleek cyber-style HUD bounding box with accentuated corners."""
        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        # 1. Soft bounding box rectangle
        cv2.rectangle(img, (x, y), (x2, y2), color, 1, cv2.LINE_AA)

        # 2. Accentuated corner brackets
        c_len = min(corner_len, w // 3, h // 3)
        c_thick = thickness + 1

        # Top-Left
        cv2.line(img, (x, y), (x + c_len, y), color, c_thick, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + c_len), color, c_thick, cv2.LINE_AA)

        # Top-Right
        cv2.line(img, (x2, y), (x2 - c_len, y), color, c_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y), (x2, y + c_len), color, c_thick, cv2.LINE_AA)

        # Bottom-Left
        cv2.line(img, (x, y2), (x + c_len, y2), color, c_thick, cv2.LINE_AA)
        cv2.line(img, (x, y2), (x, y2 - c_len), color, c_thick, cv2.LINE_AA)

        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - c_len, y2), color, c_thick, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), color, c_thick, cv2.LINE_AA)

    @staticmethod
    def draw_badge(img: np.ndarray, text: str, subtext: str, bbox: List[int], color: Tuple[int, int, int]):
        """Draws a modern pill badge with name and confidence percentage."""
        x, y, w, h = bbox
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale_title = 0.52
        font_scale_sub = 0.40

        title_size, _ = cv2.getTextSize(text, font, font_scale_title, 1)
        sub_size, _ = cv2.getTextSize(subtext, font, font_scale_sub, 1)

        badge_w = max(title_size[0] + sub_size[0] + 32, 110)
        badge_h = 28

        # Badge position: above bbox if space allows, otherwise inside/below
        badge_x = x
        badge_y = y - badge_h - 6
        if badge_y < 10:
            badge_y = y + 8

        # Keep badge within image bounds
        img_h, img_w = img.shape[:2]
        if badge_x + badge_w > img_w - 6:
            badge_x = max(6, img_w - badge_w - 6)

        # Background badge pill with alpha blend
        overlay = img.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), Visualizer.COLOR_BG_DARK, -1)
        # Border for badge
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), color, 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        # Status indicator circle
        dot_center = (badge_x + 12, badge_y + badge_h // 2)
        cv2.circle(img, dot_center, 4, color, -1, cv2.LINE_AA)

        # Title Text (Name)
        text_origin = (badge_x + 22, badge_y + 19)
        cv2.putText(img, text, text_origin, font, font_scale_title, Visualizer.COLOR_TEXT_WHITE, 1, cv2.LINE_AA)

        # Subtext (Confidence / Score)
        if subtext:
            sub_origin = (badge_x + 26 + title_size[0], badge_y + 19)
            cv2.putText(img, subtext, sub_origin, font, font_scale_sub, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_landmarks(img: np.ndarray, landmarks: List[List[int]]):
        """Draws subtle facial landmarks keypoints."""
        for pt in landmarks:
            if len(pt) == 2:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 2, Visualizer.COLOR_LANDMARK, -1, cv2.LINE_AA)

    @classmethod
    def annotate_frame(
        cls,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_landmarks: bool = True,
    ) -> np.ndarray:
        """
        Draw all face detection & recognition results on a copy of the image.
        """
        canvas = image.copy()

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            name = det.get("name", "Unknown")
            confidence = det.get("confidence", 0.0)
            is_known = det.get("is_known", False)
            landmarks = det.get("landmarks", [])

            color = cls.COLOR_KNOWN_PRIMARY if is_known else cls.COLOR_UNKNOWN_PRIMARY

            # Draw HUD Box
            cls.draw_hud_box(canvas, bbox, color)

            # Draw Landmarks
            if show_landmarks and landmarks:
                cls.draw_landmarks(canvas, landmarks)

            # Draw Badge
            subtext = f"{confidence:.1f}%" if confidence > 0 else ""
            cls.draw_badge(canvas, name, subtext, bbox, color)

        return canvas

    @staticmethod
    def to_base64_jpeg(image: np.ndarray, quality: int = 85) -> str:
        """Convert OpenCV image to base64 data URI."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        ret, buffer = cv2.imencode(".jpg", image, encode_params)
        if not ret:
            return ""
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
