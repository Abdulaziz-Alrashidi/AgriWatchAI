import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# config
MODEL_PATH = "agriwatch_16floats.tflite"
IMAGE_PATH = "yourpath"
CLASS_NAMES = [
    "Pepper_Bacterial_spot", "Pepper_healthy",
    "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot", "Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

# Model normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocess image
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((224, 224))
input_data = np.array(img, dtype=np.float32) / 255.0
input_data = (input_data - MEAN) / STD
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Postprocessing
pred_index = np.argmax(output_data)
pred_class = CLASS_NAMES[pred_index]
confidence = output_data[pred_index]

print(f"Predicted Class: {pred_class} | Confidence: {confidence:.2f}")
