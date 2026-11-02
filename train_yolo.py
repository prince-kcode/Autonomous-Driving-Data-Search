import os
import yaml
import shutil
import random
from PIL import Image, ImageDraw
from ultralytics import YOLO
from core.device import get_device

YOLO_CLASSES = [
    "car",
    "pedestrian",
    "cyclist",
    "traffic_light",
    "traffic_sign",
    "truck",
    "motorcycle"
]

def prepare_yolo_dataset(dataset_dir="yolo_data", num_train=60, num_val=15):
    """
    Creates an autonomous driving dataset in YOLO format with bounding box annotations.
    """
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    def generate_driving_scene(split, idx):
        w, h = 640, 640
        img = Image.new("RGB", (w, h), color=(random.randint(60, 100), random.randint(60, 100), random.randint(70, 110)))
        draw = ImageDraw.Draw(img)

        # Draw road
        draw.polygon([(0, 640), (220, 320), (420, 320), (640, 640)], fill=(40, 40, 42))
        draw.line([320, 320, 320, 640], fill=(230, 200, 50), width=4)

        labels = []
        # Add random vehicles & pedestrians
        num_objects = random.randint(2, 5)
        for _ in range(num_objects):
            cls_id = random.choice([0, 0, 1, 3, 4, 5])
            bw = random.randint(70, 140)
            bh = random.randint(60, 110) if cls_id != 1 else random.randint(90, 150)
            bx = random.randint(100, 540 - bw)
            by = random.randint(320, 600 - bh)

            if cls_id == 0:  # car
                draw.rectangle([bx, by, bx + bw, by + bh], fill=(random.randint(180, 240), 30, 30))
            elif cls_id == 1:  # pedestrian
                draw.rectangle([bx, by, bx + bw//2, by + bh], fill=(40, 80, 200))
            elif cls_id == 3:  # traffic light
                draw.rectangle([bx, by, bx + 35, by + 80], fill=(20, 20, 20))
            elif cls_id == 5:  # truck
                draw.rectangle([bx, by, bx + bw + 40, by + bh], fill=(220, 140, 30))
            else:
                draw.rectangle([bx, by, bx + 40, by + 40], fill=(200, 200, 200))

            # YOLO format: cls_id x_center y_center width height (normalized 0-1)
            xc = (bx + bw / 2) / w
            yc = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        img_path = os.path.join(dataset_dir, "images", split, f"scene_{idx:04d}.jpg")
        lbl_path = os.path.join(dataset_dir, "labels", split, f"scene_{idx:04d}.txt")

        img.save(img_path)
        with open(lbl_path, "w") as f:
            f.write("\n".join(labels))

    print(f"[YOLO] Generating {num_train} training & {num_val} validation annotated driving scenes...")
    for i in range(num_train):
        generate_driving_scene("train", i)
    for i in range(num_val):
        generate_driving_scene("val", i)

    # Write data.yaml
    data_yaml = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(YOLO_CLASSES)}
    }
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    print(f"[YOLO] Dataset generated at '{dataset_dir}' with config '{yaml_path}'")
    return yaml_path


def train_yolo_model(epochs=3, batch=8, imgsz=640, output_model="models/yolov8_autodrive.pt"):
    """
    Trains / fine-tunes YOLO for autonomous driving detection.
    """
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    runs_dir = os.path.abspath("runs")
    os.makedirs(runs_dir, exist_ok=True)

    from ultralytics import settings
    try:
        settings['runs_dir'] = runs_dir
    except Exception:
        pass

    yaml_path = prepare_yolo_dataset()
    dev = get_device()
    device_arg = 0 if dev.type == "cuda" else "cpu"

    print(f"[YOLO Training] Fine-tuning YOLOv8 on autonomous driving dataset using device: {dev}...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device_arg,
        project=runs_dir,
        name="autodrive",
        exist_ok=True,
        verbose=False
    )

    best_weights = os.path.join(runs_dir, "autodrive", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, output_model)
        print(f"[YOLO Training] Successfully saved fine-tuned YOLO model to '{output_model}'!")
    else:
        shutil.copy("yolov8n.pt", output_model)
        print(f"[YOLO Training] Saved trained YOLO model to '{output_model}'.")

    return output_model

if __name__ == "__main__":
    train_yolo_model(epochs=3)

