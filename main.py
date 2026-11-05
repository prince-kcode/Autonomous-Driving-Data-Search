import os
import io
import torch
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from core.device import get_device, get_device_info
from core.qdrant_manager import get_qdrant_client, COLLECTION_NAME
from core.object_recognizer import ObjectRecognizer
from core.scene_analyzer import SceneAnalyzer
from core.yolo_detector import YOLODetector
from populate_db import index_images
from train import train_model

# 1. Hardware device configuration (CUDA > MPS > CPU)
device = get_device()
print(f"[Main] Using device: {device}")

# 2. Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
try:
    clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
except Exception:
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
clip_model.eval()

# 3. Connect to Qdrant (auto-selects server or embedded)
qdrant_client = get_qdrant_client()

# 4. Initialize the 3 AI Models
print("[Main] Loading 3 Autonomous Driving AI Models...")
# Model 1: Recognize Object
model1_recognizer = ObjectRecognizer()

# Model 2: Analyze Driving Scenario & Risk
model2_analyzer = SceneAnalyzer()

# Model 3: YOLO Object Detection
model3_yolo = YOLODetector()
print("[Main] All 3 AI Models successfully initialized!")

# 5. Initialize FastAPI
app = FastAPI(
    title="Autonomous Driving Data Search & Multi-Model AI Engine",
    description="Vector search with CLIP + Model 1 (Object Recognition) + Model 2 (Scene Analysis) + Model 3 (YOLO Detection)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """
    Health check and platform status.
    """
    count = 0
    try:
        if qdrant_client.collection_exists(COLLECTION_NAME):
            info = qdrant_client.get_collection(COLLECTION_NAME)
            count = getattr(info, "points_count", 0)
    except Exception:
        count = 0

    return {
        "status": "online",
        "service": "Autonomous Driving Data Search & AI Analysis Engine",
        "compute": get_device_info(),
        "models": {
            "model_1": "Autonomous Object Recognizer (MobileNetV3 Deep CNN)",
            "model_2": "Scenario & Risk Analyzer (Multimodal CLIP)",
            "model_3": "YOLO Object Detector (YOLOv8)"
        },
        "indexed_images": count,
        "collection": COLLECTION_NAME
    }


@app.get("/search")
def search_images(
    text_query: str = Query(..., min_length=1, description="Natural language scenario description"),
    limit: int = Query(default=3, ge=1, le=50, description="Max results to return"),
    include_analysis: bool = Query(default=True, description="Whether to run 3 models on matching images")
):
    """
    Vector search for matching autonomous driving images using natural language text query.
    """
    try:
        # Process text query through CLIP
        inputs = clip_processor(
            text=[text_query],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Generate text embeddings
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)

        if not isinstance(text_features, torch.Tensor):
            text_features = getattr(text_features, "text_embeds", getattr(text_features, "pooler_output", text_features))

        # Normalize embeddings for cosine similarity
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        search_vector = text_features.detach().cpu().numpy()[0].astype(float).tolist()

        # Query Qdrant
        search_result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=search_vector,
            limit=limit,
            with_payload=True
        ).points

        results = []
        for hit in search_result:
            filename = hit.payload.get("filename", "Unknown")
            img_path = os.path.join("data", filename)

            item = {
                "filename": filename,
                "title": hit.payload.get("title", filename),
                "score": round(float(hit.score), 4),
                "image_url": f"/images/{filename}"
            }

            # Run 3 models on top results if requested
            if include_analysis and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    # Model 1
                    item["model_1_recognition"] = model1_recognizer.predict(pil_img)
                    # Model 2
                    item["model_2_analysis"] = model2_analyzer.analyze(pil_img)
                    # Model 3
                    yolo_res = model3_yolo.detect(pil_img)
                    item["model_3_yolo"] = {
                        "total_objects": yolo_res["total_objects"],
                        "counts": yolo_res["counts"],
                        "detections": yolo_res["detections"]
                    }
                except Exception as model_err:
                    item["analysis_error"] = str(model_err)

            results.append(item)

        return results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image search failed: {str(e)}"
        )


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(None),
    filename: str = Query(None, description="Existing filename in data/ directory")
):
    """
    Executes all 3 AI models on any provided image (uploaded file or data/ image).
    """
    pil_img = None

    if file is not None:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    elif filename:
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Image '{filename}' not found in data/ folder")
        pil_img = Image.open(path).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or a filename parameter")

    # Run Model 1: Object Recognition
    m1_out = model1_recognizer.predict(pil_img)

    # Run Model 2: Scenario Analysis
    m2_out = model2_analyzer.analyze(pil_img)

    # Run Model 3: YOLO Detection
    m3_out = model3_yolo.detect(pil_img)

    return {
        "model_1_recognition": m1_out,
        "model_2_analysis": m2_out,
        "model_3_yolo": {
            "total_objects": m3_out["total_objects"],
            "counts": m3_out["counts"],
            "detections": m3_out["detections"]
        }
    }


@app.post("/index")
def trigger_index(recreate: bool = False):
    """
    Triggers database re-indexing for images in data/ directory.
    """
    count = index_images(data_dir="data", recreate_collection=recreate)
    return {"message": "Indexing completed", "indexed_images": count}


@app.post("/train")
def trigger_training(epochs: int = 6, learning_rate: float = 0.001):
    """
    Retrains Model 1 (Autonomous Object Recognizer) and refreshes loaded weights.
    """
    history = train_model(epochs=epochs, learning_rate=learning_rate)
    # Reload weights in memory
    model1_recognizer._load_model()
    return {
        "message": "Model 1 training completed successfully!",
        "best_accuracy": history["val_acc"][-1] if history["val_acc"] else 0,
        "epochs_trained": len(history["epochs"])
    }


@app.get("/images/{filename}")
def get_image(filename: str):
    """
    Serves static scenario images from data/ directory.
    """
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
