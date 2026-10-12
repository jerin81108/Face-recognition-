"""
Model Downloader for OpenCV YuNet and SFace ONNX models.
Downloads official OpenCV Zoo weights for deep learning face detection and recognition.
"""

import os
import sys
import urllib.request
import time

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ]
}


def download_with_progress(url: str, dest_path: str):
    print(f"Downloading: {url} -> {dest_path}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    temp_path = dest_path + ".tmp"
    with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
        total_size = response.getheader("Content-Length")
        total_size = int(total_size) if total_size else None

        bytes_read = 0
        chunk_size = 64 * 1024
        start_time = time.time()

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            bytes_read += len(chunk)
            out_file.write(chunk)

            if total_size:
                pct = bytes_read / total_size * 100
                speed = bytes_read / (1024 * max(0.01, (time.time() - start_time)))
                print(f"\r  Progress: {pct:5.1f}% ({bytes_read / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB) @ {speed:.1f} KB/s", end="", flush=True)
            else:
                print(f"\r  Downloaded {bytes_read / 1024:.1f} KB", end="", flush=True)

    print()
    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"  Successfully saved to {dest_path} (Size: {os.path.getsize(dest_path):,} bytes)")


def ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_ready = True

    for filename, urls in MODELS.items():
        target = os.path.join(MODELS_DIR, filename)
        # Check if file exists and has plausible size (> 10KB)
        if os.path.exists(target) and os.path.getsize(target) > 50000:
            print(f"[OK] Model already exists: {filename} ({os.path.getsize(target):,} bytes)")
            continue

        downloaded = False
        for url in urls:
            try:
                print(f"[FETCH] Attempting to download {filename} from {url}...")
                download_with_progress(url, target)
                if os.path.exists(target) and os.path.getsize(target) > 50000:
                    downloaded = True
                    break
                else:
                    print(f"[WARN] File downloaded was too small ({os.path.getsize(target)} bytes), trying mirror...")
            except Exception as e:
                print(f"[WARN] Failed from {url}: {e}")

        if not downloaded:
            print(f"[ERROR] Could not download {filename} from any mirror.")
            all_ready = False

    return all_ready


if __name__ == "__main__":
    success = ensure_models()
    if success:
        print("\nAll models verified and ready!")
    else:
        print("\nSome models failed to download.")
        sys.exit(1)
