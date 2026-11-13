import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from core.device import get_device
from core.dataset_verifier import DatasetVerifier, AUTONOMOUS_CLASSES

class ObjectRecognizer:
    """
    Model 1: Autonomous Driving Object Recognition & Classification Engine.
    Trained specifically to recognize Red, Yellow, Green traffic lights,
    vehicles, pedestrians, signs, and distinguish them from natural sunlight glare.
    Includes Dataset Training Verification to check whether an uploaded image
    belongs to the trained dataset.
    """
    def __init__(self, weights_path: str = "models/object_recognizer.pth"):
        self.device = get_device()
        self.classes = AUTONOMOUS_CLASSES
        self.weights_path = weights_path
        self.model = None
        self.dataset_verifier = DatasetVerifier()
        self._load_model()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self):
        from train import AutonomousDrivingNet
        self.model = AutonomousDrivingNet(num_classes=len(self.classes), pretrained=False).to(self.device)
        
        if os.path.exists(self.weights_path):
            try:
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[ObjectRecognizer] Loaded trained weights from '{self.weights_path}'.")
            except Exception as e:
                print(f"[ObjectRecognizer] Warning: Could not load weights ({e}). Initializing backbone.")
        else:
            print(f"[ObjectRecognizer] Weights file '{self.weights_path}' not found yet. Run train.py to train.")

        self.model.eval()

    def predict(self, image_input, top_k: int = 4, filename: str = None) -> dict:
        """
        Accepts PIL Image, file path, or NumPy array and returns object predictions
        along with dataset training verification.
        """
        if isinstance(image_input, str):
            if filename is None:
                filename = os.path.basename(image_input)
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        ds_check = self.dataset_verifier.verify(image, filename=filename)

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, min(top_k, len(self.classes)))

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                "class_name": self.classes[idx.item()],
                "confidence": round(prob.item(), 4),
                "confidence_pct": round(prob.item() * 100, 1)
            })

        all_scores = {
            self.classes[i]: round(probs[i].item(), 4)
            for i in range(len(self.classes))
        }

        is_trained = ds_check["is_trained"]
        primary_obj = predictions[0]["class_name"]
        display_obj = primary_obj if is_trained else "Not Trained in Dataset"

        return {
            "model_name": "Autonomous Driving Object Recognizer (Model 1 - MobileNetV3 Deep CNN)",
            "is_trained_in_dataset": is_trained,
            "dataset_status_label": ds_check["status_label"],
            "dataset_check": ds_check,
            "primary_object": primary_obj,
            "display_object": display_obj,
            "primary_confidence": predictions[0]["confidence"],
            "primary_confidence_pct": predictions[0]["confidence_pct"],
            "top_predictions": predictions,
            "all_scores": all_scores
        }
