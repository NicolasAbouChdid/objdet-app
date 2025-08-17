import io, os, sys, pathlib
from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import torch
import numpy as np
import cv2

# ---- Windows checkpoint fix (Linux-saved checkpoints on Windows) ----
if sys.platform.startswith("win"):
    try:
        pathlib.PosixPath = pathlib.WindowsPath
    except Exception:
        pass
# ---------------------------------------------------------------------

APP_PORT = int(os.environ.get("PORT", 8080))
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")

# === Filtering thresholds ===
MAX_BOX_AREA_RATIO = 0.4   # Ignore boxes covering > 40% of image *and* centered
MIN_ASPECT_RATIO   = 1.5   # Ignore roughly square/wide boxes (filters faces)
CENTER_TOLERANCE   = 0.2   # Consider "centered" if within 20% of width

app = Flask(__name__)

# Load YOLOv5 from a local repo we cloned into the image (/app/yolov5)
model = torch.hub.load(
    "yolov5",
    "custom",
    path=MODEL_PATH,
    source="local",
    trust_repo=True
)
model.conf = 0.25
model.iou = 0.45
model.max_det = 100

CLASS_COLORS = {
    'Glass Bottle': (255, 255, 0),  # BGR cyan
    'Plastic Bottle': (0, 0, 255)   # BGR red
}

def apply_filter_and_draw(image_pil, results):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    img_area = w * h
    img_center_x = w / 2.0

    det = results.xyxy[0]  # (N, 6): [xmin, ymin, xmax, ymax, conf, cls]
    names = results.names

    if det is None or len(det) == 0:
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = xyxy
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        box_area = box_w * box_h
        aspect_ratio = box_h / (box_w + 1e-6)

        box_center_x = (x1 + x2) / 2.0
        is_centered = abs(box_center_x - img_center_x) < (CENTER_TOLERANCE * w)

        drop = (box_area / img_area > MAX_BOX_AREA_RATIO and is_centered) or (aspect_ratio < MIN_ASPECT_RATIO)
        if drop:
            continue

        cls = int(cls)
        label_name = names[cls] if cls in names else str(cls)
        color = CLASS_COLORS.get(label_name, (255, 255, 255))

        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{label_name} {float(conf):.2f}"
        y_text = max(0, int(y1) - 5)
        cv2.putText(img_bgr, label, (int(x1), y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

@app.get("/health")
def health():
    return "ok", 200

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400

    img = Image.open(f.stream).convert("RGB")
    results = model(img, size=640)
    annotated = apply_filter_and_draw(img, results)

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
