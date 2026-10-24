import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw, ImageFilter
from core.device import get_device

AUTONOMOUS_CLASSES = [
    "Car / Automobile",
    "Pedestrian / Person",
    "Bicycle / Cyclist",
    "Traffic Light (Red/Green/Yellow)",
    "Traffic Sign (Speed/Stop/Yield)",
    "Truck / Heavy Commercial Vehicle",
    "Motorcycle / Scooter",
    "Road Obstacle / Construction Hazard"
]

class AutonomousDrivingNet(nn.Module):
    """
    Deep neural network for Autonomous Driving Object Recognition.
    Uses MobileNetV3-Small backbone with a custom multi-layer classification head.
    """
    def __init__(self, num_classes=len(AUTONOMOUS_CLASSES), pretrained=True):
        super(AutonomousDrivingNet, self).__init__()
        weights = "DEFAULT" if pretrained else None
        try:
            self.backbone = models.mobilenet_v3_small(weights=weights)
        except Exception:
            self.backbone = models.mobilenet_v3_small(weights=None)
        
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.15),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class AutonomousDrivingDataset(Dataset):
    """
    Constructs a large, diverse autonomous driving training dataset
    by combining crops from real driving scenarios with procedural patterns
    and heavy domain augmentations (lighting, weather, camera angles).
    """
    def __init__(self, data_dir="data", samples_per_class=180, transform=None):
        self.samples = []
        self.transform = transform
        self.data_dir = data_dir
        self.samples_per_class = samples_per_class
        self._build_dataset()

    def _generate_synthetic_sample(self, class_idx: int) -> Image.Image:
        bg_color = (
            random.randint(20, 110),
            random.randint(20, 110),
            random.randint(20, 110)
        )
        img = Image.new("RGB", (224, 224), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Ground context with perspective
        horizon = random.randint(110, 140)
        draw.rectangle([0, horizon, 224, 224], fill=(45, 45, 48))
        draw.line([112, horizon, random.randint(90, 134), 224], fill=(240, 210, 60), width=random.randint(2, 4))
        
        if class_idx == 0:  # Car / Automobile
            car_color = random.choice([
                (210, 30, 30), (30, 90, 210), (220, 220, 220), (25, 25, 25), (40, 160, 80), (220, 140, 20)
            ])
            body_top = random.randint(90, 120)
            draw.rectangle([50, body_top, 174, body_top + 45], fill=car_color)
            draw.polygon([(65, body_top), (85, body_top - 25), (140, body_top - 25), (160, body_top)], fill=(80, 110, 140))
            draw.ellipse([60, body_top + 35, 90, body_top + 65], fill=(20, 20, 20))
            draw.ellipse([135, body_top + 35, 165, body_top + 65], fill=(20, 20, 20))
            
        elif class_idx == 1:  # Pedestrian
            skin = (235, 195, 160)
            clothes = random.choice([(40, 40, 180), (180, 40, 40), (40, 160, 60), (200, 200, 40)])
            head_y = random.randint(35, 55)
            draw.ellipse([100, head_y, 124, head_y + 24], fill=skin)
            draw.rectangle([94, head_y + 24, 130, head_y + 80], fill=clothes)
            draw.rectangle([96, head_y + 80, 110, head_y + 130], fill=(30, 30, 30))
            draw.rectangle([114, head_y + 80, 128, head_y + 130], fill=(30, 30, 30))
            
        elif class_idx == 2:  # Bicycle / Cyclist
            wheel_y = random.randint(125, 145)
            draw.ellipse([60, wheel_y, 95, wheel_y + 35], outline=(190, 190, 190), width=3)
            draw.ellipse([135, wheel_y, 170, wheel_y + 35], outline=(190, 190, 190), width=3)
            draw.line([77, wheel_y + 17, 115, wheel_y + 17], fill=(210, 40, 40), width=4)
            draw.line([115, wheel_y + 17, 152, wheel_y + 17], fill=(210, 40, 40), width=4)
            draw.ellipse([105, wheel_y - 65, 127, wheel_y - 43], fill=(225, 185, 150))
            draw.line([116, wheel_y - 43, 118, wheel_y + 5], fill=(40, 100, 220), width=5)
            
        elif class_idx == 3:  # Traffic Light
            draw.rectangle([90, 30, 134, 135], fill=(25, 25, 25), outline=(90, 90, 90), width=2)
            draw.rectangle([108, 135, 116, 215], fill=(70, 70, 70))
            active_color = random.choice(["red", "yellow", "green"])
            draw.ellipse([98, 40, 126, 68], fill=(255, 30, 30) if active_color == "red" else (60, 15, 15))
            draw.ellipse([98, 70, 126, 98], fill=(255, 205, 0) if active_color == "yellow" else (60, 50, 10))
            draw.ellipse([98, 100, 126, 128], fill=(30, 245, 45) if active_color == "green" else (10, 60, 15))
            
        elif class_idx == 4:  # Traffic Sign
            draw.rectangle([108, 105, 116, 215], fill=(130, 130, 130))
            sign_type = random.choice(["stop", "yield", "speed"])
            if sign_type == "stop":
                draw.polygon([(82, 45), (102, 30), (122, 30), (142, 45), (142, 68), (122, 83), (102, 83), (82, 68)], fill=(215, 20, 20))
                draw.text((94, 49), "STOP", fill=(255, 255, 255))
            elif sign_type == "yield":
                draw.polygon([(78, 35), (146, 35), (112, 92)], fill=(245, 245, 245), outline=(225, 20, 20), width=3)
            else:
                draw.ellipse([80, 35, 144, 99], fill=(255, 255, 255), outline=(225, 20, 20), width=4)
                draw.text((98, 58), "60", fill=(20, 20, 20))
                
        elif class_idx == 5:  # Truck / Heavy Commercial Vehicle
            truck_color = random.choice([(235, 130, 25), (35, 110, 215), (230, 230, 230), (45, 140, 70)])
            draw.rectangle([35, 60, 155, 160], fill=truck_color)
            draw.rectangle([155, 88, 192, 160], fill=(truck_color[0]//2, truck_color[1]//2, truck_color[2]//2))
            draw.rectangle([160, 95, 186, 118], fill=(130, 175, 215))
            draw.ellipse([50, 148, 78, 176], fill=(20, 20, 20))
            draw.ellipse([115, 148, 143, 176], fill=(20, 20, 20))
            draw.ellipse([160, 148, 188, 176], fill=(20, 20, 20))
            
        elif class_idx == 6:  # Motorcycle / Scooter
            draw.ellipse([65, 125, 96, 156], outline=(35, 35, 35), width=4)
            draw.ellipse([128, 125, 159, 156], outline=(35, 35, 35), width=4)
            draw.polygon([(85, 135), (112, 108), (136, 130)], fill=(225, 45, 45))
            draw.ellipse([108, 75, 128, 95], fill=(25, 25, 25))
            draw.line([114, 95, 118, 125], fill=(30, 75, 160), width=5)
            
        elif class_idx == 7:  # Road Obstacle / Construction Hazard
            draw.polygon([(92, 70), (132, 70), (148, 168), (76, 168)], fill=(255, 110, 15))
            draw.rectangle([88, 98, 136, 114], fill=(245, 245, 245))
            draw.rectangle([83, 132, 141, 148], fill=(245, 245, 245))
            draw.rectangle([68, 168, 156, 178], fill=(25, 25, 25))

        if random.random() > 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
            
        return img

    def _build_dataset(self):
        # 1. Procedural domain samples
        for class_idx in range(len(AUTONOMOUS_CLASSES)):
            for _ in range(self.samples_per_class):
                img = self._generate_synthetic_sample(class_idx)
                self.samples.append((img, class_idx))

        # 2. Extract crops and augmentations from real images in data_dir
        if os.path.exists(self.data_dir):
            real_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            for rf in real_files:
                try:
                    img_path = os.path.join(self.data_dir, rf)
                    real_img = Image.open(img_path).convert("RGB")
                    w, h = real_img.size
                    
                    # Determine probable class from filename keywords
                    c_idx = 0  # default car
                    rf_lower = rf.lower()
                    if any(k in rf_lower for k in ["pedestrian", "person", "crosswalk"]):
                        c_idx = 1
                    elif any(k in rf_lower for k in ["cyclist", "bicycle", "bike"]):
                        c_idx = 2
                    elif any(k in rf_lower for k in ["traffic_light", "signal"]):
                        c_idx = 3
                    elif any(k in rf_lower for k in ["stop_sign", "sign"]):
                        c_idx = 4
                    elif any(k in rf_lower for k in ["truck", "bus"]):
                        c_idx = 5
                    elif any(k in rf_lower for k in ["motorcycle", "scooter"]):
                        c_idx = 6
                    elif any(k in rf_lower for k in ["construction", "barrier", "hazard"]):
                        c_idx = 7

                    # Extract 4 multi-scale crops per real image
                    for _ in range(4):
                        crop_w = int(w * random.uniform(0.5, 0.9))
                        crop_h = int(h * random.uniform(0.5, 0.9))
                        x1 = random.randint(0, max(0, w - crop_w))
                        y1 = random.randint(0, max(0, h - crop_h))
                        crop = real_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((224, 224))
                        self.samples.append((crop, c_idx))
                except Exception:
                    pass

        random.shuffle(self.samples)
        print(f"[Dataset] Compiled {len(self.samples)} augmented autonomous driving training samples across 8 classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def train_model(
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 0.0008,
    save_path: str = "models/object_recognizer.pth",
    history_path: str = "models/training_history.json",
    callback=None
):
    """
    Trains the Autonomous Driving Object Recognition model and saves weights & telemetry.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = get_device()
    print(f"[Training] Using compute device: {device}")

    # Rich domain augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = AutonomousDrivingDataset(samples_per_class=180, transform=train_transform)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = AutonomousDrivingNet(num_classes=len(AUTONOMOUS_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "classes": AUTONOMOUS_CLASSES,
        "total_samples": len(full_dataset)
    }

    print(f"[Training] Beginning training for {epochs} epochs on {len(train_set)} training samples...")
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_epoch_loss = running_loss / total
        train_epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_epoch_loss, 4))
        history["train_acc"].append(round(train_epoch_acc * 100, 2))
        history["val_loss"].append(round(val_epoch_loss, 4))
        history["val_acc"].append(round(val_epoch_acc * 100, 2))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc*100:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc*100:.2f}%")

        if callback:
            callback(epoch, epochs, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc)

        if val_epoch_acc >= best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_epoch_acc,
                "classes": AUTONOMOUS_CLASSES
            }, save_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[Training] Model 1 Complete! Best Val Accuracy: {best_val_acc*100:.2f}%. Model saved to '{save_path}'.")
    return history


if __name__ == "__main__":
    train_model(epochs=8, batch_size=16, learning_rate=0.0008)
