from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import os


# -------------------
# Flask app setup
# -------------------
app = Flask(__name__)

# CORS configuration
CORS(app,
     resources={r"/*": {
         "origins": [
             "https://osteodetector-frontend.vercel.app",
             "http://localhost:5173",
             "http://localhost:3000",
             "http://127.0.0.1:5173"
         ],
         "allow_headers": ["Content-Type", "Authorization"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "supports_credentials": True
     }})


# -------------------
# Model config
# -------------------
MODEL_PATH = "model.keras"
IMG_SIZE = (224, 224)


# Same order as in training: image_dataset_from_directory -> ['Normal', 'Osteoarthritis']
CLASS_NAMES = ["Normal", "Osteoarthritis"]


print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")


# -------------------
# Helper: preprocess uploaded image
# -------------------
def preprocess_image(file_storage):
    image_bytes = file_storage.read()

    # Optional simple size guard (5 MB)
    if len(image_bytes) > 5 * 1024 * 1024:
        raise ValueError("File too large (max 5MB)")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    # Match training: raw 0–255 pixels, no /255.0
    arr = np.array(img).astype("float32")      # (H, W, 3)
    arr = np.expand_dims(arr, axis=0)         # (1, H, W, 3)
    return arr


# -------------------
# Prediction endpoint
# -------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Handle preflight requests
    if request.method == "OPTIONS":
        return "", 204

    if "image" not in request.files:
        return jsonify({"error": "No image file provided with key 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = preprocess_image(file)

        # Softmax output: (1, 2) -> [p_normal, p_osteo]
        probs = model.predict(img_array)[0]     # shape (2,)
        idx = int(np.argmax(probs))             # 0 or 1
        predicted_class = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        osteo_prob = float(probs[CLASS_NAMES.index("Osteoarthritis")])
        normal_prob = float(probs[CLASS_NAMES.index("Normal")])

        return jsonify({
            "predicted_class": predicted_class,   # "Normal" or "Osteoarthritis"
            "confidence": confidence,             # prob of predicted_class
            "classes": CLASS_NAMES,               # ["Normal", "Osteoarthritis"]
            "probabilities": [normal_prob, osteo_prob],
            "normal_probability": normal_prob,
            "osteo_probability": osteo_prob
        }), 200

    except ValueError as ve:
        # Explicit input issue (e.g. too large)
        print("Validation error:", ve)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Unexpected error
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction failed"}), 500


# -------------------
# Health check
# -------------------
@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({"status": "ok"}), 200


# -------------------
# Run server
# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
