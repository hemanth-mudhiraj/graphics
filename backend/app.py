"""Flask backend for CheXNet chest X-ray analysis with Grad-CAM and Gemini integration."""

import os
from pathlib import Path

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from model import load_model, predict_with_heatmap
from gemini_service import get_recommendations

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).parent.parent / "best_model.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
CORS(app)

# Global model reference
model = None


def init_model():
    """Load the model on startup."""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("Set MODEL_PATH env var or place best_model.pth in the project root.")
    else:
        print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
        model = load_model(MODEL_PATH, DEVICE)
        print("Model loaded successfully!")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Place best_model.pth in the project root."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.content_type or not file.content_type.startswith("image/"):
        return jsonify({"error": "Please upload an image file (JPEG, PNG, etc.)"}), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Maximum size is 10 MB."}), 400

    try:
        result = predict_with_heatmap(model, image_bytes, DEVICE)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Get recommendations from Gemini
    top_disease = result["top_prediction"]
    top_prob = result["top_probability"]
    recommendations = get_recommendations(top_disease, top_prob)

    result["recommendations"] = recommendations

    return jsonify(result)


# Initialize model on import
init_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
