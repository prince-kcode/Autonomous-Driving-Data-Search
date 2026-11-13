import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client.models import PointStruct
from core.device import get_device
from core.qdrant_manager import get_qdrant_client, init_collection, COLLECTION_NAME

def index_images(
    data_dir: str = "data",
    recreate_collection: bool = False,
    clip_model=None,
    clip_processor=None,
    client=None,
):
    """
    Indexes all autonomous driving scenario images from data_dir into Qdrant.
    """
    device = get_device()
    print(f"[PopulateDB] Using compute device: {device}")

    # Load or reuse CLIP model & processor
    if clip_model is not None and clip_processor is not None:
        model = clip_model
        processor = clip_processor
    else:
        model_name = "openai/clip-vit-base-patch32"
        try:
            model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)

    model.eval()

    # Initialize or reuse Qdrant
    client = client if client is not None else get_qdrant_client()
    init_collection(client, collection_name=COLLECTION_NAME, recreate=recreate_collection)

    os.makedirs(data_dir, exist_ok=True)
    image_files = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])

    if not image_files:
        print(f"[PopulateDB] No images found in '{data_dir}'. Add some images first.")
        return 0

    points = []
    print(f"[PopulateDB] Generating CLIP embeddings for {len(image_files)} images...")

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(data_dir, image_name)
        try:
            with Image.open(image_path) as raw_img:
                img_format = raw_img.format or "JPEG"
                image = raw_img.convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                features = model.get_image_features(**inputs)

            if not isinstance(features, torch.Tensor):
                features = getattr(features, "image_embeds", getattr(features, "pooler_output", features))

            # Normalize embedding for Cosine similarity
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            embedding = features.detach().cpu().numpy()[0].astype(float).tolist()

            # Human-readable scenario tag based on filename
            tag = image_name.replace("_", " ").rsplit(".", 1)[0].title()

            points.append(PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "filename": image_name,
                    "title": tag,
                    "width": image.width,
                    "height": image.height,
                    "format": img_format
                }
            ))
            print(f"  [{idx+1}/{len(image_files)}] Indexed: {image_name} ({image.width}x{image.height})")
        except Exception as e:
            print(f"  [Error] Failed to process {image_name}: {e}")

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\n[PopulateDB] Successfully indexed {len(points)} images into Qdrant collection '{COLLECTION_NAME}'!")
        return len(points)
    return 0


populate_qdrant = index_images

if __name__ == "__main__":
    index_images(recreate_collection=True)
