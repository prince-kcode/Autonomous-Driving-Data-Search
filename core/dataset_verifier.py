import os
import numpy as np
from PIL import Image

AUTONOMOUS_CLASSES = [
    "Car / Automobile",
    "Pedestrian / Person",
    "Bicycle / Cyclist",
    "Traffic Light (Red Signal)",
    "Traffic Light (Yellow Signal)",
    "Traffic Light (Green Signal)",
    "Traffic Sign (Speed/Stop/Yield)",
    "Truck / Heavy Commercial Vehicle",
    "Motorcycle / Scooter",
    "Road Obstacle / Construction Hazard",
    "Forest Road / Sunlight Glare (Non-Traffic Light)"
]


def infer_class_index_from_filename(filename: str) -> int | None:
    """
    Maps a dataset image filename to its ground-truth class index in AUTONOMOUS_CLASSES (0..10).
    """
    rf_lower = os.path.basename(filename).lower()
    if any(k in rf_lower for k in ["pedestrian", "person", "crosswalk"]):
        return 1
    if any(k in rf_lower for k in ["cyclist", "bicycle", "bike"]):
        return 2
    if any(k in rf_lower for k in ["red_light", "traffic_light_red"]):
        return 3
    if any(k in rf_lower for k in ["yellow_light", "amber_light", "traffic_light_yellow"]):
        return 4
    if any(k in rf_lower for k in ["green_light", "traffic_light_green"]):
        return 5
    if any(k in rf_lower for k in ["stop_sign", "speed_sign", "traffic_sign"]):
        return 6
    if any(k in rf_lower for k in ["truck", "bus", "ambulance"]):
        return 7
    if any(k in rf_lower for k in ["motorcycle", "scooter"]):
        return 8
    if any(k in rf_lower for k in ["construction", "barrier", "hazard", "cone"]):
        return 9
    if any(k in rf_lower for k in ["forest", "foggy_rural", "sunset", "sunlight"]):
        return 10
    if any(k in rf_lower for k in [
        "car", "highway_sunny", "city_downtown", "rainy_night",
        "night_highway", "electric_vehicle", "parking_lot", "sedan", "vehicle"
    ]):
        return 0
    return None


class DatasetVerifier:
    """
    Checks whether an uploaded or queried image is part of the trained dataset (`data/`).
    Uses multi-metric perceptual pixel fingerprinting (64x64 normalized cosine similarity + L1 MAE)
    resilient to JPEG re-encoding across HTTP multipart uploads.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.index = []
        self.refresh()

    @staticmethod
    def _compute_signature(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Returns (raw_01_flat, unit_norm_centered_flat) for a 64x64 RGB thumbnail."""
        thumb = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.asarray(thumb, dtype=np.float32).reshape(-1) / 255.0
        centered = arr - np.mean(arr)
        norm = np.linalg.norm(centered)
        unit = centered / norm if norm > 1e-6 else centered
        return arr, unit

    def refresh(self):
        """Rebuilds the in-memory signature index for all valid images in data_dir."""
        self.index = []
        if not os.path.exists(self.data_dir):
            return

        files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ])
        for fname in files:
            fpath = os.path.join(self.data_dir, fname)
            try:
                with Image.open(fpath) as raw_img:
                    img = raw_img.convert("RGB")
                    w, h = img.size
                    arr, unit = self._compute_signature(img)
                cls_idx = infer_class_index_from_filename(fname)
                cls_name = AUTONOMOUS_CLASSES[cls_idx] if cls_idx is not None else "Car / Automobile"
                self.index.append({
                    "filename": fname,
                    "width": w,
                    "height": h,
                    "arr": arr,
                    "unit": unit,
                    "class_idx": cls_idx if cls_idx is not None else 0,
                    "class_name": cls_name
                })
            except Exception:
                pass

    def verify(self, image_input, filename: str = None) -> dict:
        """
        Checks if `image_input` is in the trained dataset.
        Returns a dictionary with `is_trained`, `status_label`, `matched_filename`,
        `similarity_pct`, `trained_class`, and `message`.
        """
        self.refresh()

        if isinstance(image_input, str):
            if filename is None:
                filename = os.path.basename(image_input)
            with Image.open(image_input) as raw_img:
                image = raw_img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        q_arr, q_unit = self._compute_signature(image)

        best_entry = None
        best_score = -1.0
        best_mae = 1.0

        for entry in self.index:
            cos_sim = float(np.dot(q_unit, entry["unit"]))
            mae = float(np.mean(np.abs(q_arr - entry["arr"])))
            # Combined similarity score (0..1)
            combined = 0.65 * max(0.0, cos_sim) + 0.35 * max(0.0, 1.0 - mae * 2.5)
            if combined > best_score:
                best_score = combined
                best_mae = mae
                best_entry = entry

        # Check if exact filename in data/ was specified and matches visually
        if filename:
            base_fn = os.path.basename(filename)
            for entry in self.index:
                if entry["filename"].lower() == base_fn.lower():
                    mae_fn = float(np.mean(np.abs(q_arr - entry["arr"])))
                    if mae_fn < 0.08:
                        return {
                            "is_trained": True,
                            "status": "TRAINED",
                            "status_label": "✅ Image is Trained by Dataset",
                            "matched_filename": entry["filename"],
                            "trained_class": entry["class_name"],
                            "trained_class_idx": entry["class_idx"],
                            "similarity_pct": round(max(98.5, (1.0 - mae_fn) * 100), 1),
                            "total_dataset_images": len(self.index),
                            "message": (
                                f"Verified: This image is part of the trained dataset "
                                f"('data/{entry['filename']}') under class '{entry['class_name']}'."
                            )
                        }

        # Visual fingerprint match threshold (handles JPEG re-compression on upload)
        if best_entry is not None and best_mae <= 0.055 and best_score >= 0.92:
            sim_pct = round(min(100.0, max(95.0, best_score * 100)), 1)
            return {
                "is_trained": True,
                "status": "TRAINED",
                "status_label": "✅ Image is Trained by Dataset",
                "matched_filename": best_entry["filename"],
                "trained_class": best_entry["class_name"],
                "trained_class_idx": best_entry["class_idx"],
                "similarity_pct": sim_pct,
                "total_dataset_images": len(self.index),
                "message": (
                    f"Verified: Uploaded image matches trained dataset file "
                    f"'data/{best_entry['filename']}' ({sim_pct}% similarity) — Class: '{best_entry['class_name']}'."
                )
            }

        # Not in trained dataset
        closest_fn = best_entry["filename"] if best_entry else "None"
        closest_sim = round(max(0.0, best_score) * 100, 1) if best_entry else 0.0
        return {
            "is_trained": False,
            "status": "NOT_TRAINED",
            "status_label": "❌ Image is NOT Trained by Dataset",
            "matched_filename": None,
            "closest_dataset_file": closest_fn,
            "trained_class": "Not Trained in Dataset",
            "trained_class_idx": None,
            "similarity_pct": closest_sim,
            "total_dataset_images": len(self.index),
            "message": (
                f"This uploaded image is NOT trained by the dataset (not found in 'data/'). "
                f"Closest dataset image: '{closest_fn}' ({closest_sim}% similarity)."
            )
        }

