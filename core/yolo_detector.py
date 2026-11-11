import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from core.device import get_device

class YOLODetector:
    """
    Model 3: YOLO Object Detection for Autonomous Driving.
    Detects vehicles, pedestrians, cyclists, signs, and traffic lights.
    Includes specialized verification for traffic lights:
    - Filters out natural sunlight/sky glare false positives.
    - Classifies active signal colors: Red, Yellow/Amber, and Green.
    """
    def __init__(self, model_name: str = None):
        self.device = get_device()
        # Default to official standard YOLOv8n for reliable generalization
        if model_name is None:
            model_name = "yolov8n.pt"
        print(f"[YOLODetector] Initializing YOLO with weights '{model_name}' on device '{self.device}'...")
        self.model = YOLO(model_name)
        
        # Autonomous driving relevant classes in COCO
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

    @staticmethod
    def verify_traffic_light(crop_rgb: np.ndarray) -> tuple[bool, str, float]:
        """
        Verifies if a candidate bounding box is an authentic traffic light:
        - Rejects diffuse sunlight glare and sky brightness.
        - Analyzes chromatic saturation in Red, Yellow, and Green spectrums.
        Returns:
            (is_valid, state_label, signal_strength)
        """
        if crop_rgb is None or crop_rgb.size == 0 or crop_rgb.shape[0] < 8 or crop_rgb.shape[1] < 8:
            return False, "invalid_crop", 0.0

        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        total_px = crop_rgb.shape[0] * crop_rgb.shape[1]

        # 1. Reject diffuse sunlight / sky flare (high brightness with low saturation)
        glare_pixels = (v > 150) & (s < 55)
        glare_ratio = np.sum(glare_pixels) / total_px

        # 2. Check for Red Signal (Hue 0-10 or 165-180, high saturation & brightness)
        red_mask = ((h <= 10) | (h >= 165)) & (s >= 70) & (v >= 65)
        r_pct = np.sum(red_mask) / total_px

        # 3. Check for Yellow / Amber Signal (Hue 14-36)
        yellow_mask = (h >= 14) & (h <= 36) & (s >= 75) & (v >= 75)
        y_pct = np.sum(yellow_mask) / total_px

        # 4. Check for Green Signal (Hue 38-88)
        green_mask = (h >= 38) & (h <= 88) & (s >= 70) & (v >= 65)
        g_pct = np.sum(green_mask) / total_px

        # If glare dominates and there is no strong concentrated signal, reject as sunlight
        if glare_ratio > 0.25 and max(r_pct, y_pct, g_pct) < 0.02:
            return False, "sunlight_glare", 0.0

        # Determine dominant active signal state
        scores = {"red": r_pct, "yellow": y_pct, "green": g_pct}
        best_color = max(scores, key=scores.get)
        best_pct = scores[best_color]

        if best_pct >= 0.012:
            return True, f"traffic light ({best_color})", round(float(best_pct * 10), 2)

        # 5. Check if housing structure exists (dark enclosure with vertical aspect ratio)
        aspect = crop_rgb.shape[0] / max(1, crop_rgb.shape[1])
        dark_ratio = np.sum(v < 65) / total_px
        if aspect >= 1.3 and dark_ratio > 0.25 and glare_ratio < 0.2:
            return True, "traffic light (casing)", 0.5

        return False, "false_positive_glare", 0.0

    @staticmethod
    def draw_detections(image_input, detections: list) -> Image.Image:
        """
        Draws bounding boxes and color-coded labels onto an image from detection dicts.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            annotated_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            annotated_np = np.array(image_input.convert("RGB"))
        else:
            annotated_np = image_input.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            display_name = det["class_name"]
            conf = det["confidence"]

            color = (0, 230, 80)
            if "red" in display_name:
                color = (255, 30, 30)
            elif "yellow" in display_name:
                color = (255, 205, 0)
            elif "green" in display_name:
                color = (30, 240, 50)
            elif "car" in display_name or "truck" in display_name or "bus" in display_name:
                color = (0, 140, 255)
            elif "person" in display_name:
                color = (255, 100, 20)

            cv2.rectangle(annotated_np, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{display_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_np, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_np, label_txt, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return Image.fromarray(annotated_np)

    def detect(self, image_input, conf_threshold: float = 0.30) -> dict:
        """
        Runs YOLO object detection with traffic light color verification
        and glare suppression.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        np_image = np.array(image)

        # Pass PIL RGB image directly (Ultralytics treats numpy arrays as BGR)
        if self.device.type == "cuda":
            device_arg = 0
        elif self.device.type == "mps":
            device_arg = "mps"
        else:
            device_arg = "cpu"

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            device=device_arg,
            verbose=False
        )

        first_res = results[0]
        boxes = first_res.boxes

        detections = []
        counts = {}

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                raw_name = self.model.names.get(cls_id, f"class_{cls_id}").lower()
                conf = float(box.conf[0].item())
                xyxy = [int(c) for c in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(np_image.shape[1], xyxy[2]), min(np_image.shape[0], xyxy[3])

                # Filter only autonomous driving entities
                if cls_id not in self.av_relevant_classes and raw_name not in ["car", "truck", "bus", "person", "bicycle", "motorcycle", "traffic light", "stop sign"]:
                    continue

                display_name = raw_name

                # Specialized verification for Traffic Lights
                if "traffic" in raw_name and "light" in raw_name:
                    crop = np_image[y1:y2, x1:x2]
                    is_valid, state, _ = self.verify_traffic_light(crop)
                    if not is_valid:
                        # Suppress false positive (sunlight glare, white sky, tree light)
                        continue
                    display_name = state

                # Suppress low-confidence cyclists on tiny blurry patches
                if "bicycle" in raw_name or "cyclist" in raw_name:
                    if conf < 0.35 and (x2 - x1 < 25 or y2 - y1 < 25):
                        continue

                # Add valid detection
                detections.append({
                    "class_name": display_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })
                counts[display_name] = counts.get(display_name, 0) + 1

        annotated_pil = self.draw_detections(np_image, detections)

        return {
            "model_name": "YOLO Object Detector (Model 3 - Real-time Perception with Signal Verification)",
            "annotated_image": annotated_pil,
            "total_objects": len(detections),
            "counts": counts,
            "detections": detections
        }
