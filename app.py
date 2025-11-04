import contextlib
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import torch
from flask import Flask, render_template, request, send_from_directory, url_for
from ultralytics import YOLO

# Paths used across the app
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "models" / "best.pt"
CLEANUP_INTERVAL_SECONDS = 600  # 10 minutes


app = Flask(__name__)


@contextlib.contextmanager
def _safe_torch_load_context() -> contextlib.Iterator[None]:
    """Temporarily allow YOLO checkpoints that require full pickle deserialization."""

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        # Ultralytics internally calls torch.load without specifying weights_only.
        # Explicitly disabling the safe-only mode restores compatibility with
        # checkpoints created before PyTorch 2.6 tightened defaults.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]


def load_model(model_path: Path) -> YOLO:
    """Load the YOLOv8 model once when the application starts."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. Please place best.pt in the models directory."
        )
    # YOLO automatically detects device (CPU/GPU). No extra logic needed here.
    with _safe_torch_load_context():
        return YOLO(str(model_path))


model = load_model(MODEL_PATH)


def ensure_directories() -> None:
    """Guarantee the upload and results folders exist before handling requests."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute the Intersection over Union (IoU) between two bounding boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def run_inference(image_path: Path) -> Path:
    """Run YOLOv8 inference and annotate helmet usage."""
    image = cv2.imread(str(image_path))
    results = model(str(image_path))
    result = results[0]
    names: Dict[int, str] = result.names

    people: List[Tuple[int, int, int, int]] = []
    helmets: List[Tuple[int, int, int, int]] = []

    for box in result.boxes:
        cls = int(box.cls)
        name = names.get(cls, "")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if "person" in name.lower():
            people.append((x1, y1, x2, y2))
        elif "helmet" in name.lower():
            helmets.append((x1, y1, x2, y2))

    # Track which person boxes have a helmet nearby (using IoU threshold).
    helmet_matches = [False] * len(people)
    for i, person_box in enumerate(people):
        for helmet_box in helmets:
            if iou(person_box, helmet_box) > 0.1:
                helmet_matches[i] = True
                break

    # Draw the annotations on the image. Helmets get green, people without helmets get red.
    for helmet_box in helmets:
        x1, y1, x2, y2 = helmet_box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(image, "HELMET", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    for person_box, has_helmet in zip(people, helmet_matches):
        x1, y1, x2, y2 = person_box
        if has_helmet:
            color = (0, 180, 255)
            label = "HELMET ON"
        else:
            color = (0, 0, 255)
            label = "NO HELMET"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_path = RESULT_DIR / f"result_{timestamp}.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


def cleanup_worker() -> None:
    """Periodically delete files older than 10 minutes from upload/result folders."""
    while True:
        cutoff = time.time() - CLEANUP_INTERVAL_SECONDS
        for folder in (UPLOAD_DIR, RESULT_DIR):
            for file_path in folder.glob("*"):
                try:
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                        file_path.unlink()
                except FileNotFoundError:
                    # File might have been deleted by another thread; ignore.
                    continue
        time.sleep(CLEANUP_INTERVAL_SECONDS)


def start_cleanup_thread() -> None:
    """Spawn a background thread to keep folders tidy."""
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()


start_cleanup_thread()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    ensure_directories()
    file = request.files.get("image")
    if not file:
        return {"error": "No file uploaded."}, 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"upload_{timestamp}.jpg"
    upload_path = UPLOAD_DIR / filename
    file.save(upload_path)

    result_path = run_inference(upload_path)

    result_url = url_for("static", filename=f"results/{result_path.name}")
    return {"result_url": result_url}


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded files (primarily useful for debugging)."""
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
