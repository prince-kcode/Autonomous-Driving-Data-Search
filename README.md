# 🚗 Autonomous Driving Data Search & Multi-Model AI Engine

An end-to-end Autonomous Driving Data Search and Scenario Intelligence platform powered by **Vector Embeddings (CLIP + Qdrant)**, deep learning models, and real-time computer vision.

---

## 🌟 Key Features

- **Vector Scenario Search**: Retrieve exact driving conditions (e.g., *"a red car on a rainy road at night"*, *"pedestrians crossing the street"*) using CLIP text-image vector embeddings stored in Qdrant.
- **3 Integrated AI Models**:
  1. **Model 1: Autonomous Object Recognizer (Trained Custom Deep CNN)**  
     Fine-tuned MobileNetV3-Small deep convolutional neural network trained specifically on 11 autonomous driving entity classes (Cars, Pedestrians, Cyclists, Red/Yellow/Green Traffic Lights, Traffic Signs, Trucks/Buses, Motorcycles, Road Hazards, and Forest Sunlight Glare negative samples).
  2. **Model 2: Scenario & Risk Analyzer (Multimodal CLIP Reasoning)**  
     Real-time perception of environmental conditions (Weather, Time/Lighting, Road Topology), calculation of AV Safety Hazard Index (0-100%), and autonomous driving control recommendations (e.g., headway buffer, traction control, AEB standby).
  3. **Model 3: YOLO Object Detection (YOLO11 + Signal Verifier)**  
     High-speed spatial bounding-box detection of vehicles, pedestrians, cyclists, and traffic signals with HSV signal-state verification (Red/Yellow/Green) and sunlight glare false-positive suppression.
- **Interactive Web Dashboard**: Streamlit dashboard with 3 specialized studios:
  - 🔍 **Scenario Search & Retrieval**
  - 🔬 **Real-Time Vision Telemetry (Live 3-Model Diagnostics)**
  - 🧠 **Model Training & Benchmark Studio (Loss & Accuracy telemetry)**
- **Hardware Acceleration**: Automatic GPU acceleration (CUDA on NVIDIA RTX GPUs / MPS on Apple Silicon / CPU fallback).
- **Embedded or Server Qdrant**: Operates either with a remote Qdrant Docker daemon or fully self-contained using embedded disk storage (`./qdrant_storage`).

---

## 📂 Project Architecture

```
Autonomous-Driving-Data-Search-main/
├── core/
│   ├── device.py               # Hardware detection (CUDA, MPS, CPU)
│   ├── qdrant_manager.py       # Resilient Qdrant connection manager
│   ├── object_recognizer.py    # Model 1: Autonomous Object Recognizer
│   ├── scene_analyzer.py       # Model 2: Scenario & Risk Analyzer
│   └── yolo_detector.py        # Model 3: YOLO Object Detector
├── data/                       # Verified autonomous driving scenario dataset
├── models/
│   ├── object_recognizer.pth   # Trained Model 1 weights
│   ├── yolo11_autodrive.pt     # Custom YOLO11 fine-tuned checkpoint
│   ├── training_history.json   # Training accuracy & loss telemetry
│   └── all_models_summary.json # Multi-model calibration summary
├── app.py                      # Interactive Streamlit Web UI
├── main.py                     # FastAPI REST API Backend
├── populate_db.py              # CLIP Vector Indexing into Qdrant
├── train.py                    # PyTorch Training Script for Model 1
├── train_yolo.py               # YOLO11 Dataset & Fine-Tuning Script
├── train_all.py                # 3-Model Orchestration & Calibration Script
├── ingest_dataset.py           # Automated external dataset ingestion & retraining pipeline
├── test_system.py              # End-to-end system verification
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify System
Run the end-to-end verification script:
```bash
python test_system.py
```

### 3. Train Model 1 (Optional - Pre-trained weights included)
```bash
python train.py
```

### 4. Index Dataset into Qdrant
```bash
python populate_db.py
```

### 5. Launch the Live Demo
Start the Streamlit Web Application:
```bash
streamlit run app.py
```
*(Optional)* Start the FastAPI REST API backend in a separate terminal:
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

---

## 🤖 The 3 Models Explained

| Model | Purpose | Architecture | Output |
|---|---|---|---|
| **Model 1: Recognize Object** | Classifies driving objects into 11 AV categories (incl. R/Y/G signals + glare negative) | MobileNetV3-Small Deep CNN | Class name, Top-K probability scores |
| **Model 2: Scenario Analyzer** | Environmental perception & safety risk | Multimodal CLIP ViT-B/32 | Weather, Lighting, Road type, Hazard % (Low/Moderate/High/Critical), AV actions |
| **Model 3: YOLO Detector** | Spatial bounding-box detection & signal color verification | Ultralytics YOLO11 + HSV Verifier | Bounding boxes, class/signal tags, count breakdown |

---

## 📡 API Endpoints

- `GET /` — System health and active GPU compute device.
- `GET /search?text_query=...&limit=3` — Natural language vector scenario search.
- `POST /analyze` — Run all 3 models on an uploaded image or filename.
- `POST /train?epochs=8&learning_rate=0.0008` — Retrain Model 1 with live weight reload.
- `POST /index` — Re-index images into Qdrant vector database.
- `GET /images/{filename}` — Static scenario image server.
