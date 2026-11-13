import os
import sys
import glob
import shutil
import zipfile
import argparse
from PIL import Image

def ingest_dataset(source_path: str, target_data_dir: str = "data", reindex: bool = True, train: bool = False):
    """
    Ingests autonomous driving dataset images from a folder or zip archive into the platform:
    1. Unpacks zip archive if provided.
    2. Identifies valid RGB driving scenario images (.jpg, .jpeg, .png, .webp).
    3. Normalizes filenames and copies images into `data/`.
    4. Detects any YOLO format annotations or data.yaml and updates `yolo_data/`.
    5. Optionally reindexes vector database via `populate_db.py`.
    6. Optionally retrains all 3 models via `train_all.py`.
    """
    os.makedirs(target_data_dir, exist_ok=True)
    temp_dir = None

    print("=" * 65)
    print("      AUTONOMOUS DRIVING DATASET INGESTION PIPELINE")
    print("=" * 65)
    print(f"[Ingest] Source input: '{source_path}'")

    if not os.path.exists(source_path):
        print(f"[Error] Source path '{source_path}' does not exist!")
        return {"status": "error", "message": f"Source path '{source_path}' not found."}

    # 1. Handle zip archive
    if os.path.isfile(source_path) and source_path.lower().endswith(".zip"):
        temp_dir = os.path.abspath("temp_dataset_unpacked")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[Ingest] Extracting zip archive to temporary folder '{temp_dir}'...")
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        search_root = temp_dir
    else:
        search_root = source_path

    # 2. Discover all images
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    found_images = []
    for root, _, files in os.walk(search_root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                found_images.append(os.path.join(root, f))

    print(f"[Ingest] Discovered {len(found_images)} candidate images in source.")
    if not found_images:
        print("[Warning] No valid images found to ingest.")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return {"status": "no_images", "count": 0}

    # 3. Copy & validate images into target_data_dir
    copied_count = 0
    skipped_count = 0
    for img_path in found_images:
        try:
            # Validate image opens cleanly
            with Image.open(img_path) as im:
                im.verify()
            
            base_name = os.path.basename(img_path)
            # Avoid collision
            dest_path = os.path.join(target_data_dir, base_name)
            counter = 1
            while os.path.exists(dest_path):
                # If identical file size, skip duplicate
                if os.path.getsize(dest_path) == os.path.getsize(img_path):
                    break
                name_part, ext_part = os.path.splitext(base_name)
                dest_path = os.path.join(target_data_dir, f"{name_part}_{counter}{ext_part}")
                counter += 1

            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[Warning] Skipping corrupted or unreadable image {img_path}: {e}")

    print(f"[Ingest] Successfully ingested {copied_count} new driving scenario images into '{target_data_dir}' (Skipped {skipped_count} existing).")

    # 4. Check for YOLO data / labels
    yolo_labels = []
    for root, _, files in os.walk(search_root):
        for f in files:
            if f.endswith(".txt") and f not in ["classes.txt", "requirements.txt", "README.txt"]:
                yolo_labels.append(os.path.join(root, f))
    if yolo_labels:
        print(f"[Ingest] Discovered {len(yolo_labels)} YOLO bounding box annotation files.")
        dest_yolo_labels = os.path.join("yolo_data", "labels", "train")
        os.makedirs(dest_yolo_labels, exist_ok=True)
        for yl in yolo_labels:
            shutil.copy2(yl, os.path.join(dest_yolo_labels, os.path.basename(yl)))

    # Clean up temp extraction folder if used
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 5. Reindex Qdrant Vector Database
    if reindex:
        print("\n[Ingest] Triggering CLIP Vector Indexing across updated dataset...")
        try:
            from populate_db import populate_qdrant
            populate_qdrant()
        except Exception as e:
            print(f"[Warning] Could not reindex vectors: {e}")

    # 6. Retrain all models if requested
    if train:
        print("\n[Ingest] Triggering retraining pipeline for all 3 models...")
        try:
            from train_all import train_all_models
            train_all_models(m1_epochs=8, yolo_epochs=3, retrain_m1=True, retrain_yolo=True)
        except Exception as e:
            print(f"[Warning] Retraining failed: {e}")

    print("\n" + "=" * 65)
    print("           DATASET INGESTION PIPELINE COMPLETE")
    print("=" * 65)
    return {
        "status": "success",
        "new_images_ingested": copied_count,
        "total_images_in_data": len(os.listdir(target_data_dir))
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest external driving datasets into AV-Vision platform")
    parser.add_argument("--source", type=str, default="data", help="Path to downloaded dataset zip file or folder")
    parser.add_argument("--target", type=str, default="data", help="Target dataset directory (default: 'data')")
    parser.add_argument("--no-reindex", action="store_true", help="Skip Qdrant vector indexing")
    parser.add_argument("--train", action="store_true", help="Automatically train all models after ingestion")
    args = parser.parse_args()

    ingest_dataset(
        source_path=args.source,
        target_data_dir=args.target,
        reindex=not args.no_reindex,
        train=args.train
    )

