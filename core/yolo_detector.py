import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from core.device import get_device

class YOLODetector:
    """
    Model 3: YOLO Object Detection for Autonomous Driving
    Detects vehicles, pedestrians, cyclists, signs, and traffic infrastructure
    with precise bounding boxes and class confidence scores.
    """
    def __init__(self, model_name: str = None):
        self.device = get_device()
        if model_name is None:
            if os.path.exists("models/yolov8_autodrive.pt"):
                model_name = "models/yolov8_autodrive.pt"
            else:
                model_name = "yolov8n.pt"
        print(f"[YOLODetector] Initializing YOLO with weights '{model_name}' on device '{self.device}'...")
        self.model = YOLO(model_name)
        
        # Key autonomous driving classes in COCO
        self.av_relevant_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic light",
            11: "stop sign"
        }

    def detect(self, image_input, conf_threshold: float = 0.25) -> dict:
        """
        Runs YOLO object detection and returns annotated image + detection breakdown.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        np_image = np.array(image)

        # Run inference
        results = self.model.predict(
            source=np_image,
            conf=conf_threshold,
            device=str(self.device).split(":")[0] if self.device.type == "cuda" else "cpu",
            verbose=False
        )

        first_res = results[0]
        boxes = first_res.boxes

        detections = []
        counts = {}

        # Draw on image copy
        annotated_np = first_res.plot()
        annotated_pil = Image.fromarray(annotated_np)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0].item())
                xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": xyxy
                })

                counts[cls_name] = counts.get(cls_name, 0) + 1

        return {
            "model_name": "YOLO Object Detector (Model 3 - YOLOv8n)",
            "annotated_image": annotated_pil,
            "total_objects": len(detections),
            "counts": counts,
            "detections": detections
        }

