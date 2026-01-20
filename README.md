
# AgriWatch AI: Edge-Optimized Plant Disease Classification

AgriWatch AI is a high-performance computer vision system designed for real-time plant disease detection on edge hardware. Utilizing a **MobileNetV3** architecture, it classifies 16 healthy and diseased states in Pepper, Potato, and Tomato crops.

The project delivers a production-ready **3.1 MB** model optimized for deployment on Raspberry Pi and other ARM-based embedded devices.

## 📊 Performance at a Glance

* **Accuracy:** 92.10%
* **Model Size:** 3.1 MB
* **Precision:** Float16 (TFLite)
* **Target Hardware:** Raspberry Pi, Jetson Nano, Mobile ARM CPUs

---

## 🛠️ Deployment & Usage

AgriWatch AI is designed for immediate integration. The inference engine is optimized for the `tflite-runtime` to ensure a minimal footprint on edge devices.

### 1. Requirements

Refer to /depolyment

Ensure you have Python installed on your edge device. Install the lightweight dependencies:

```bash
pip install tflite-runtime pillow numpy

```

### 2. Running Inference

Use the `deploy.py` script to perform classification on a leaf image:

```bash
python deploy.py --image path/to/leaf_image.jpg

```

**Example Output:**
`Predicted Class: Tomato_Early_blight | Confidence: 0.87`

---

## 🔬 Methodology Summary

The development of AgriWatch AI involved a dual-track experimental workflow to determine the optimal deployment strategy:

### Workflow 1: Quantization-Aware Training (QAT)

* Explored simulated 8-bit quantization during fine-tuning.
* **Finding:** While theoretically sound, QAT was ultimately deprecated due to the **complexity and fragility of the exportation pipeline.** The transition through the ONNX-to-TensorFlow bridge introduced parity risks and non-deterministic behavior, making future refinement appears unreliable for a production environment.

### Workflow 2: Static Float16 Quantization (Selected)

* Model was exported via `PyTorch` ⮕ `ONNX` ⮕ `TensorFlow` ⮕ `TFLite`.
* **Result:** This path was prioritized for its maintainable and stable deployment lifecycle, maintaining a high accuracy of **92.10%**.

---

## 📂 Project Structure

* `agriwatch_model.tflite`: The 3.1 MB optimized model binary.
* `deploy.py`: Python script for edge inference.
* `TECHNICAL_DOC.md`: Detailed engineering report and experimental data.

---

## 🌿 Supported Classes

* **Pepper:** Bacterial_spot, healthy.
* **Potato:** Early_blight, Late_blight, healthy.
* **Tomato:** Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot, Spider_mites, Target_Spot, YellowLeaf_Curl_Virus, Mosaic_virus, healthy.

Refer to Techinical Documentation.ipynb for the full techinical documentation

---

**Would you like me to help you refine the `deploy.py` script to ensure it handles the Float16 tensors correctly for your Raspberry Pi setup?**
