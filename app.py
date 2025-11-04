import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator
import contextlib
import cv2
from flask import Flask, render_template, request, send_from_directory, url_for, jsonify
from ultralytics import YOLO
import torch
from ultralytics.nn.tasks import DetectionModel

# === PyTorch 2.6 fix for YOLO checkpoints ============================
torch.serialization.add_safe_globals([DetectionModel])
# =====================================================================

# Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes

app = Flask(__name__)

# --- Context manager for YOLO model load ---
@contextlib.contextmanager
def _safe_torch_load_context() -> Iterator[None]:
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please place best.pt in the models directory."
        )
    with _safe_torch_load_context():
        return YOLO(str(model_path))


model = load_model(MODEL_PATH)

# --- Utility directories ---
def ensure_directories():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()

# --- Run YOLO inference and return annotated image + summary text ---
def run_inference(image_path: Path) -> tuple[Path, dict]:
    """Run YOLOv8 inference and return both annotated image and text results."""
    results = model.predict(source=str(image_path), conf=0.25, imgsz=640, verbose=False)
    result = results[0]
    rendered = result.plot()  # image with drawn boxes
    names = result.names

    # Collect textual detection summary
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = names[cls_id]
        conf = float(box.conf)
        detections.append({"class": cls_name, "confidence": round(conf, 2)})

    # Save rendered image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_path = RESULT_DIR / f"result_{timestamp}.jpg"
    cv2.imwrite(str(output_path), rendered)

    # Count detections by class for simple summary
    summary = {}
    for d in detections:
        summary[d["class"]] = summary.get(d["class"], 0) + 1

    return output_path, {"detections": detections, "summary": summary}


# --- Background cleanup ---
def cleanup_worker():
    while True:
        cutoff = time.time() - CLEANUP_INTERVAL_SECONDS
        for folder in (UPLOAD_DIR, RESULT_DIR):
            for file_path in folder.glob("*"):
                try:
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                except FileNotFoundError:
                    continue
        time.sleep(CLEANUP_INTERVAL_SECONDS)


threading.Thread(target=cleanup_worker, daemon=True).start()


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    ensure_directories()
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"upload_{timestamp}.jpg"
    upload_path = UPLOAD_DIR / filename
    file.save(upload_path)

    result_path, results_data = run_inference(upload_path)

    result_url = url_for("static", filename=f"results/{result_path.name}")
    results_data["result_url"] = result_url

    return jsonify(results_data)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
