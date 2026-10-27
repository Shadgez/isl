import os
import pickle
import threading

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "isl_nn_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "isl_classes.pickle")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

missing = [os.path.basename(p) for p in (MODEL_PATH, CLASSES_PATH, LANDMARKER_PATH) if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "rb") as f:
    class_names = [str(x) for x in pickle.load(f)]

base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.25,
    min_hand_presence_confidence=0.25,
)
detector = vision.HandLandmarker.create_from_options(options)
detector_lock = threading.Lock()
app = Flask(__name__)


def extract_features(landmarks_list):
    # Intentionally matches the feature extraction in the supplied app.py.
    sorted_hands = sorted(landmarks_list, key=lambda hand: hand[0].x)
    all_x = [lm.x for hand in sorted_hands for lm in hand]
    all_y = [lm.y for hand in sorted_hands for lm in hand]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    scale = max(max_x - min_x, max_y - min_y, 1e-4)
    vector = []
    for i in range(2):
        if i < len(sorted_hands):
            for lm in sorted_hands[i]:
                vector.extend([(lm.x - min_x) / scale, (lm.y - min_y) / scale])
        else:
            vector.extend([0.0] * 42)
    return np.array([vector], dtype=np.float32)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "classes": len(class_names), "model_input_shape": str(model.input_shape)})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No camera frame was received."}), 400
    raw = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "Could not decode camera frame."}), 400
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    try:
        with detector_lock:
            result = detector.detect(mp_image)
        if not result.hand_landmarks:
            return jsonify({"letter": None, "confidence": 0, "message": "No hands detected"})
        features = extract_features(result.hand_landmarks)
        expected = model.input_shape[-1] if isinstance(model.input_shape, tuple) else None
        if expected is not None and features.shape[1] != expected:
            app.logger.error("Feature length %s does not match model input %s", features.shape[1], expected)
            return jsonify({"error": f"Model expects {expected} features but app generated {features.shape[1]}. Check training preprocessing."}), 500
        predictions = model.predict(features, verbose=0)[0]
        idx = int(np.argmax(predictions))
        confidence = float(predictions[idx])
        # Return top predictions as a debugging aid; this does not change the model.
        top_indices = np.argsort(predictions)[-3:][::-1]
        top = [{"label": class_names[int(i)] if int(i) < len(class_names) else str(i), "score": float(predictions[int(i)])} for i in top_indices]
        letter = class_names[idx] if idx < len(class_names) else str(idx)
        return jsonify({"letter": letter, "confidence": confidence, "message": "Hand detected", "top": top})
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the terminal logs."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
