import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from core.device import get_device

class YOLODetector:
    """
    Model 3: YOLO11 Object Detection for Autonomous Driving.
    Detects vehicles, pedestrians, cyclists, signs, and traffic lights.
    Includes specialized verification for traffic lights:
    - Filters out natural sunlight/sky glare false positives.
    - Classifies active signal colors: Red, Yellow/Amber, and Green.
    """
    def __init__(self, model_name: str = None):
        self.device = get_device()
        # Default to standard official YOLO11n for robust real-world generalization
        if model_name is None:
            if os.path.exists("yolo11n.pt"):
                model_name = "yolo11n.pt"
            elif os.path.exists("models/yolo11_autodrive.pt"):
                model_name = "models/yolo11_autodrive.pt"
            else:
                model_name = "yolo11n.pt"
        print(f"[YOLODetector] Initializing YOLO11 with weights '{model_name}' on device '{self.device}'...")
        self.model = YOLO(model_name)
        
        # Autonomous driving relevant classes in COCO (vehicles, VRU, signals, signs, animals & road obstacles)
        self.av_relevant_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            28: "suitcase",
            32: "sports ball",
            36: "skateboard"
        }
        self.animal_classes = {
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe"
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
        dark_ratio = np.sum(v < 70) / total_px

        # Authentic 3-aspect traffic signal has dark housing enclosure (dark_ratio >= 0.16)
        # and a concentrated circular lamp (0.012 <= best_pct <= 0.45, rejecting solid yellow road signs)
        if 0.012 <= best_pct <= 0.45 and dark_ratio >= 0.16:
            return True, f"traffic light ({best_color})", round(float(best_pct * 10), 2)

        # Strict Traffic Light Validation:
        # Authentic traffic lights must show an active illuminated signal (Red, Yellow/Amber, or Green).
        # Reject unlit dark rectangles (such as parked cars, tree trunks, or building shadows) to prevent false positives.
        return False, "no_active_signal", 0.0

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
            elif any(a in display_name for a in ("cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")):
                color = (255, 70, 210)

            cv2.rectangle(annotated_np, (x1, y1), (x2, y2), color, 2)
            label_txt = f"{display_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_np, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated_np, label_txt, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return Image.fromarray(annotated_np)

    @staticmethod
    def detect_construction_hazards(np_image: np.ndarray) -> list:
        """
        Detects orange-and-white reflective road construction cones and barriers
        (which are not part of standard 80-class COCO vocabulary).
        """
        h_img, w_img = np_image.shape[:2]
        hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # High-saturation safety orange mask
        orange_mask = ((h >= 4) & (h <= 20) & (s >= 180) & (v >= 180)).astype(np.uint8) * 255
        white_mask = ((s <= 30) & (v >= 220)).astype(np.uint8)

        # Close gaps across reflective white collar stripes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 25))
        closed = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cone_detections = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 35 or bh < 45 or bw > int(w_img * 0.65) or bh > int(h_img * 0.65):
                continue
            box_area = bw * bh
            o_ratio = float(np.sum(orange_mask[y:y + bh, x:x + bw] > 0)) / box_area
            w_ratio = float(np.sum(white_mask[y:y + bh, x:x + bw] > 0)) / box_area
            # Must contain both safety orange body and bright white reflective bands
            if o_ratio >= 0.25 and 0.08 <= w_ratio <= 0.50:
                label = "construction cone" if bw < int(w_img * 0.35) else "construction barrier"
                cone_detections.append({
                    "class_name": label,
                    "confidence": round(min(0.96, 0.70 + o_ratio * 0.4), 3),
                    "bbox": [int(x), int(y), int(x + bw), int(y + bh)]
                })
        return cone_detections

    def detect(self, image_input, conf_threshold: float = 0.30) -> dict:
        """
        Runs YOLO11 object detection with traffic light color verification,
        aerial parking-lot vehicle recovery, construction cone detection, and glare suppression.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        np_image = np.array(image)
        h_img, w_img = np_image.shape[:2]

        # Pass PIL RGB image directly (Ultralytics treats numpy arrays as BGR)
        if self.device.type == "cuda":
            device_arg = 0
        elif self.device.type == "mps":
            device_arg = "mps"
        else:
            device_arg = "cpu"

        base_conf = min(conf_threshold, 0.22)
        results = self.model.predict(
            source=image,
            conf=base_conf,
            device=device_arg,
            verbose=False
        )

        first_res = results[0]
        boxes = first_res.boxes

        # Check if aerial top-down parking lot, animal silhouette, or high-res distant object pass is needed
        raw_names_640 = [
            self.model.names.get(int(b.cls[0].item()), "").lower()
            for b in (boxes if boxes is not None else [])
            if float(b.conf[0].item()) >= conf_threshold or "traffic light" in self.model.names.get(int(b.cls[0].item()), "").lower()
        ]
        av_keywords = [
            "car", "truck", "bus", "person", "pedestrian", "bicycle",
            "cyclist", "motorcycle", "traffic light", "stop sign", "traffic sign",
            "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "fire hydrant", "parking meter", "bench",
            "backpack", "suitcase", "sports ball", "skateboard"
        ]
        has_standard_av = any(any(k in rn.replace("_", " ") for k in av_keywords) for rn in raw_names_640)
        has_animal_640 = any(rn.replace("_", " ") in self.animal_classes for rn in raw_names_640)
        max_conf_640 = max(
            [
                float(b.conf[0].item())
                for b in (boxes if boxes is not None else [])
                if any(k in self.model.names.get(int(b.cls[0].item()), "").lower().replace("_", " ") for k in av_keywords)
            ] or [0.0]
        )
        aerial_phone_count = sum(
            1 for b in (boxes if boxes is not None else [])
            if self.model.names.get(int(b.cls[0].item()), "").lower() == "cell phone" and float(b.conf[0].item()) >= conf_threshold
        )

        is_aerial_parking = False
        if aerial_phone_count >= 3 or not has_standard_av or has_animal_640 or max_conf_640 < 0.65:
            results_hi = self.model.predict(
                source=image,
                imgsz=1280,
                conf=base_conf,
                device=device_arg,
                verbose=False
            )
            boxes_hi = results_hi[0].boxes
            if boxes_hi is not None and len(boxes_hi) > 0:
                hi_names = [
                    self.model.names.get(int(b.cls[0].item()), "").lower()
                    for b in boxes_hi
                    if float(b.conf[0].item()) >= conf_threshold or "traffic light" in self.model.names.get(int(b.cls[0].item()), "").lower()
                ]
                max_conf_hi = max(
                    [
                        float(b.conf[0].item())
                        for b in boxes_hi
                        if any(k in self.model.names.get(int(b.cls[0].item()), "").lower().replace("_", " ") for k in av_keywords)
                    ] or [0.0]
                )
                if sum(1 for rn in hi_names if rn == "cell phone") >= 3:
                    is_aerial_parking = True
                    boxes = boxes_hi
                elif any(any(k in rn.replace("_", " ") for k in av_keywords) for rn in hi_names):
                    if not has_standard_av or has_animal_640 or max_conf_hi > max_conf_640:
                        boxes = boxes_hi

        def _extract_valid_detections(candidate_boxes, allow_any_object: bool = False):
            dets = []
            cnts = {}
            if candidate_boxes is None or len(candidate_boxes) == 0:
                return dets, cnts
            for box in candidate_boxes:
                cls_id = int(box.cls[0].item())
                raw_name = self.model.names.get(cls_id, f"class_{cls_id}").lower()
                conf = float(box.conf[0].item())
                xyxy = [int(c) for c in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = max(0, xyxy[0]), max(0, xyxy[1]), min(w_img, xyxy[2]), min(h_img, xyxy[3])
                bw, bh = x2 - x1, y2 - y1

                raw_norm = raw_name.replace("_", " ")
                min_req_conf = 0.24 if raw_norm in self.animal_classes else (0.35 if allow_any_object else conf_threshold)
                if "traffic light" not in raw_norm and conf < min_req_conf:
                    continue

                # Remap top-down aerial parking stall vehicles misclassified by COCO as 'cell phone'
                if is_aerial_parking and raw_norm == "cell phone" and bw < w_img * 0.15 and bh < h_img * 0.25:
                    raw_norm = "car"
                # Remap compact road sports car with rear wing misclassified as 'train'
                elif raw_norm == "train" and bw < w_img * 0.60 and bh < h_img * 0.55:
                    raw_norm = "car"

                # Filter only autonomous driving, animal wildlife, or road obstacle entities unless fallback active
                if not allow_any_object and cls_id not in self.av_relevant_classes and not any(k in raw_norm for k in av_keywords):
                    continue

                display_name = raw_norm

                # Specialized verification for Traffic Lights
                if "traffic light" in raw_norm:
                    crop = np_image[y1:y2, x1:x2]
                    is_valid, state, _ = self.verify_traffic_light(crop)
                    if not is_valid:
                        # Suppress false positive (sunlight glare, white sky, tree light)
                        continue
                    display_name = state

                # Suppress low-confidence cyclists on tiny blurry patches
                if "bicycle" in raw_norm or "cyclist" in raw_norm:
                    if conf < 0.35 and (bw < 25 or bh < 25):
                        continue

                dets.append({
                    "class_name": display_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })
                cnts[display_name] = cnts.get(display_name, 0) + 1
            return dets, cnts

        detections, counts = _extract_valid_detections(boxes)

        # Low-light / backlit silhouette recovery pass (CLAHE contrast enhancement) if nothing detected yet
        if not detections and float(np.mean(np_image)) < 130.0:
            lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l_eq = clahe.apply(l_ch)
            enhanced_rgb = cv2.cvtColor(cv2.merge((l_eq, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
            res_enh = self.model.predict(
                source=Image.fromarray(enhanced_rgb),
                imgsz=1280,
                conf=0.22,
                device=device_arg,
                verbose=False
            )
            detections, counts = _extract_valid_detections(res_enh[0].boxes)

        # Check for road construction cones / barriers if no standard COCO objects matched
        if not detections:
            cone_dets = self.detect_construction_hazards(np_image)
            for cd in cone_dets:
                detections.append(cd)
                counts[cd["class_name"]] = counts.get(cd["class_name"], 0) + 1

        # General COCO foreground object fallback if no standard vehicle/animal/cone was found
        if not detections:
            detections, counts = _extract_valid_detections(boxes, allow_any_object=True)

        annotated_pil = self.draw_detections(np_image, detections)

        return {
            "model_name": "YOLO11 Object Detector (Model 3 - Real-time Perception with Signal Verification)",
            "annotated_image": annotated_pil,
            "total_objects": len(detections),
            "counts": counts,
            "detections": detections
        }
