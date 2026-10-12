"""
Face Recognition System - Flask Web Application Backend.
Provides high-performance REST APIs for real-time video stream recognition,
image upload analysis, person enrollment, roster management, and attendance logs.
"""

import os
import sys
import io
import time
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response

from core.pipeline import RecognitionPipeline
from core.visualizer import Visualizer

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

# Initialize Pipeline
pipeline = RecognitionPipeline()


def decode_image_from_request(req) -> np.ndarray:
    """Helper to extract OpenCV BGR image from either multipart file or base64 JSON payload."""
    if "image" in req.files:
        file = req.files["image"]
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    data = req.get_json(silent=True)
    if data and "image" in data:
        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return None


@app.route("/")
def index():
    """Render dashboard web UI."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Return system and model readiness status."""
    persons = pipeline.database.get_persons()
    total_samples = sum(p["samples_count"] for p in persons)
    return jsonify({
        "status": "online",
        "engine": "OpenCV YuNet + SFace Deep Learning",
        "detector_model": os.path.basename(pipeline.detector.model_path),
        "recognizer_model": os.path.basename(pipeline.recognizer.model_path),
        "persons_count": len(persons),
        "samples_count": total_samples,
        "cosine_threshold": pipeline.recognizer.cosine_threshold,
        "detector_confidence": pipeline.detector.conf_threshold,
    })


@app.route("/api/persons", methods=["GET"])
def get_persons():
    """Retrieve list of enrolled persons."""
    persons = pipeline.database.get_persons()
    output = []
    for p in persons:
        # Generate thumbnail data uri if sample exists
        thumb_uri = ""
        if p["sample_path"] and os.path.exists(p["sample_path"]):
            try:
                img = cv2.imread(p["sample_path"])
                if img is not None:
                    # Resize thumbnail to max 200px
                    h, w = img.shape[:2]
                    scale = 200 / max(h, w)
                    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
                    thumb_uri = Visualizer.to_base64_jpeg(thumb, quality=75)
            except Exception:
                pass

        output.append({
            "name": p["name"],
            "samples_count": p["samples_count"],
            "thumbnail": thumb_uri,
            "updated_at": p["updated_at"],
        })
    return jsonify({"persons": output, "count": len(output)})


@app.route("/api/persons", methods=["POST"])
def enroll_person():
    """
    Enroll a new person or add photos to existing person.
    Accepts:
      - Multipart: name + file upload(s)
      - JSON: name + base64 image
    """
    name = ""
    img = None

    if request.is_json:
        data = request.get_json()
        name = data.get("name", "").strip()
        img_b64 = data.get("image", "")
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid image base64 data: {e}"}), 400
    else:
        name = request.form.get("name", "").strip()
        if "photos" in request.files:
            file = request.files["photos"]
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif "image" in request.files:
            file = request.files["image"]
            nparr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if not name:
        return jsonify({"success": False, "error": "Person name is required."}), 400

    if img is None:
        return jsonify({"success": False, "error": "No valid image provided."}), 400

    success, msg = pipeline.database.add_person(name, img)
    if not success:
        return jsonify({"success": False, "error": msg}), 400

    return jsonify({"success": True, "message": msg, "name": name})


@app.route("/api/persons/<name>", methods=["DELETE"])
def delete_person(name):
    """Delete an enrolled person."""
    deleted = pipeline.database.delete_person(name)
    if deleted:
        return jsonify({"success": True, "message": f"Person '{name}' deleted successfully."})
    return jsonify({"success": False, "error": f"Person '{name}' not found."}), 404


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """
    Analyze image for faces, compute embeddings, match identities, and return annotated result.
    """
    img = decode_image_from_request(request)
    if img is None:
        return jsonify({"success": False, "error": "No image payload detected."}), 400

    # Optional query/body parameters
    threshold = None
    show_landmarks = True
    log_events = True

    if request.is_json:
        data = request.get_json()
        if "threshold" in data:
            try:
                threshold = float(data["threshold"])
            except Exception:
                pass
        show_landmarks = data.get("show_landmarks", True)
        log_events = data.get("log_events", True)
    else:
        if "threshold" in request.form:
            try:
                threshold = float(request.form["threshold"])
            except Exception:
                pass
        show_landmarks = request.form.get("show_landmarks", "true").lower() == "true"
        log_events = request.form.get("log_events", "true").lower() == "true"

    result = pipeline.process_image(
        img,
        threshold=threshold,
        annotate=True,
        show_landmarks=show_landmarks,
        log_events=log_events,
    )

    annotated_b64 = ""
    if result["annotated_image"] is not None:
        annotated_b64 = Visualizer.to_base64_jpeg(result["annotated_image"], quality=85)

    detected_names = [f["name"] for f in result["faces"]]
    verified_count = sum(1 for f in result["faces"] if f["is_known"])
    unknown_count = result["count"] - verified_count

    return jsonify({
        "success": True,
        "count": result["count"],
        "verified_count": verified_count,
        "unknown_count": unknown_count,
        "latency_ms": result["latency_ms"],
        "faces": result["faces"],
        "detected_names": detected_names,
        "annotated_image": annotated_b64,
    })


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Get detection / attendance history."""
    limit = request.args.get("limit", 100, type=int)
    logs = pipeline.get_logs(limit=limit)
    return jsonify({"logs": logs, "count": len(logs)})


@app.route("/api/logs/export", methods=["GET"])
def export_logs():
    """Export attendance logs as downloadable CSV."""
    if os.path.exists(pipeline.csv_log_path):
        return send_file(
            pipeline.csv_log_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"attendance_export_{int(time.time())}.csv",
        )
    return jsonify({"error": "No logs found."}), 404


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    """Clear all logged events."""
    pipeline.clear_logs()
    return jsonify({"success": True, "message": "Detection logs cleared."})


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update runtime thresholds."""
    data = request.get_json(silent=True) or {}
    if "cosine_threshold" in data:
        pipeline.recognizer.set_threshold(float(data["cosine_threshold"]))
    if "detector_confidence" in data:
        pipeline.detector.set_confidence(float(data["detector_confidence"]))

    return jsonify({
        "success": True,
        "cosine_threshold": pipeline.recognizer.cosine_threshold,
        "detector_confidence": pipeline.detector.conf_threshold,
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """List preloaded sample test images."""
    samples = []
    if os.path.exists(TEST_IMAGES_DIR):
        for f in os.listdir(TEST_IMAGES_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not f.endswith("_annotated.jpg"):
                samples.append({
                    "filename": f,
                    "title": f.replace("_", " ").replace(".jpg", "").replace(".png", "").title(),
                    "url": f"/api/samples/{f}",
                })
    return jsonify({"samples": samples})


@app.route("/api/samples/<filename>", methods=["GET"])
def get_sample_file(filename):
    """Serve a sample test image."""
    return send_from_directory(TEST_IMAGES_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================")
    print(f"  Face Recognition Web Dashboard Running On:")
    print(f"  http://127.0.0.1:{port}")
    print(f"==================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
