import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import json
import numpy as np
import tensorflow as tf
from gradcam_utils import generate_gradcam, preprocess_image  # optional if you want Grad-CAM

app = FastAPI()

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)

# Load TFLite model
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load TFLite model: {e}")

# Load class names
try:
    class_names = sorted(os.listdir(os.path.join(BASE_DIR, "augmented_dataset")))
    print(f"✅ Loaded {len(class_names)} class names.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to read class names: {e}")

# Load remedies
try:
    with open(os.path.join(BASE_DIR, "remedies.json"), "r") as f:
        remedies = json.load(f)
    print("✅ Remedies loaded.")
except FileNotFoundError:
    remedies = {}
    print("⚠️ remedies.json not found. Proceeding without remedies.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    img_path = f"uploads/{file.filename}"
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess image
    try:
        img_array = preprocess_image(img_path)  # should return shape [1, H, W, C], dtype=float32
        # TFLite expects float32
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_index = predictions.argmax()
        class_name = class_names[class_index]
        confidence = float(predictions[class_index])
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})

    # Optional Grad-CAM
    gradcam_path = f"uploads/gradcam_{file.filename}"
    try:
        generate_gradcam(interpreter, img_path, gradcam_path)  # Adjust function for TFLite if needed
    except Exception as e:
        gradcam_path = ""
        print(f"⚠️ Grad-CAM generation failed: {e}")

    return {
        "class": class_name,
        "confidence": round(confidence, 3),
        "remedy": remedies.get(class_name, "No remedy found."),
        "gradcam": gradcam_path
    }

@app.get("/gradcam/{filename}")
async def get_gradcam(filename: str):
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(status_code=404, content={"message": "File not found."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
