# 🧛‍♂️ Deepfake Detection Platform (MesoNet-4)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)

A Deep Learning application engineered to detect facial forgery in images and video streams. The system utilizes the **MesoNet-4** Convolutional Neural Network architecture coupled with OpenCV facial extraction pipelines to detect mesoscopic image artifacts and unnatural facial noise.

---

## 🌟 Key Features

- 🕵️ **MesoNet-4 Deep Learning Model**: Custom CNN architecture optimized for detecting micro-forgery artifacts in facial frames.
- 🎯 **Automated Facial Localization**: Integrates OpenCV Haar Cascade classifiers to crop facial regions with adaptive spatial padding prior to inference.
- 🎥 **Multi-Format Media Support**: Seamlessly processes `.mp4`, `.avi`, `.mov`, `.jpg`, `.png` inputs.
- 🔗 **Direct URL Media Fetching**: Enter direct video URLs or online links; integrated `yt-dlp` automatically downloads and processes media on the fly.
- 📊 **Interactive Web Dashboard**: Built with Streamlit, providing real-time video/image previews, verdict metrics (`REAL` vs `FAKE`), confidence percentages, and historical logs.
- ⚡ **Standalone CLI Inference**: Dedicated command-line script (`predict.py`) for quick terminal testing and batch evaluation.

---

## 🏗️ Project Architecture

```
project deepfake/
├── app.py                     # Streamlit Interactive Web Dashboard
├── model.py                   # MesoNet-4 Architecture Definition (Keras/TensorFlow)
├── predict.py                 # Standalone CLI Prediction Script
├── extract_faces.py           # OpenCV Face Extraction Helper Pipeline
├── data_loader.py             # Dataset Loader & Frame Preprocessing
├── train.py                   # Model Training Script
├── evaluate.py                # Model Evaluation & Metrics Script
├── requirements.txt           # Python Dependencies
├── .gitignore                 # Environment & Cache Exclusions
├── .streamlit/                # Streamlit UI Configuration
│   └── config.toml
└── weights/                   # Pre-trained Model Weights
    └── meso4_best.weights.h5
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/MeDhruv007/Deepfake-Video-Detection.git
cd Deepfake-Video-Detection
```

### 2. Create & Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / MacOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💻 CLI Usage

You can also run inference directly from the terminal using `predict.py`:

```bash
python predict.py path/to/video.mp4
# OR
python predict.py path/to/image.jpg
```

**Example Output:**
```text
Scanning video frames to detect faces: 0000.mp4...
Found 13 faces. Running Meso-4 inference...

==============================
 File: 0000.mp4
 Prediction: FAKE
 Confidence: 50.99%
==============================
```

---

## 🔬 Model Details

- **Model Type**: MesoNet-4 (Mesoscopic Video Forgery Detection)
- **Input Resolution**: 256x256x3 (RGB)
- **Layer Breakdown**: 4 Sequential Convolutional & Pooling Layers + Dense Classification Layer with Dropout Regularization.
- **Target Metrics**: Focused on eye/mouth boundary artifacts and frame-by-frame temporal consistency.

---

## 📄 References & Acknowledgments

- *Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network" (IEEE WIFS 2018)*
- OpenCV Haar Cascades for facial bounding box extraction.

---

© 2025 Dhruv Patel — Built with Python, TensorFlow & Streamlit.
