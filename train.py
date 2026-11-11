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
    "Traffic Light (Red Signal)",
    "Traffic Light (Yellow Signal)",
    "Traffic Light (Green Signal)",
    "Traffic Sign (Speed/Stop/Yield)",
    "Truck / Heavy Commercial Vehicle",
    "Motorcycle / Scooter",
    "Road Obstacle / Construction Hazard",
    "Forest Road / Sunlight Glare (Non-Traffic Light)"
]

class AutonomousDrivingNet(nn.Module):
    """
    Deep neural network for Autonomous Driving Object Recognition.
    Uses MobileNetV3-Small backbone with a custom multi-layer classification head.
    Trained to distinguish vehicles, pedestrians, signs, Red/Yellow/Green traffic lights,
    and suppress false positives on sunlight glare & forest roads.
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


class _TransformSubset(Dataset):
    """Applies a specific transform pipeline to a split list of (PIL.Image, label) pairs."""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


class AutonomousDrivingDataset(Dataset):
    """
    Compiles an autonomous driving training dataset with:
    - Dedicated primary signal states: Red, Yellow/Amber, and Green traffic lights.
    - Explicit negative samples for Sunlight Glare, Tree Foliage & Forest Roads.
    - Multi-scale crops from verified autonomous driving scenarios (split safely to avoid leakage).
    """
    def __init__(self, data_dir="data", samples_per_class=160, transform=None):
        self.samples = []
        self.real_crop_groups = {}
        self.transform = transform
        self.data_dir = data_dir
        self.samples_per_class = samples_per_class
        self._build_dataset()

    def _generate_synthetic_sample(self, class_idx: int) -> Image.Image:
        bg_color = (
            random.randint(25, 110),
            random.randint(25, 110),
            random.randint(25, 110)
        )
        img = Image.new("RGB", (224, 224), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Ground context
        horizon = random.randint(110, 140)
        draw.rectangle([0, horizon, 224, 224], fill=(45, 45, 48))
        draw.line([112, horizon, random.randint(90, 134), 224], fill=(240, 210, 60), width=3)
        
        if class_idx == 0:  # Car / Automobile
            car_color = random.choice([(210, 30, 30), (30, 90, 210), (220, 220, 220), (25, 25, 25), (40, 160, 80)])
            body_top = random.randint(90, 120)
            draw.rectangle([50, body_top, 174, body_top + 45], fill=car_color)
            draw.polygon([(65, body_top), (85, body_top - 25), (140, body_top - 25), (160, body_top)], fill=(80, 110, 140))
            draw.ellipse([60, body_top + 35, 90, body_top + 65], fill=(20, 20, 20))
            draw.ellipse([135, body_top + 35, 165, body_top + 65], fill=(20, 20, 20))
            
        elif class_idx == 1:  # Pedestrian
            skin = (235, 195, 160)
            clothes = random.choice([(40, 40, 180), (180, 40, 40), (40, 160, 60)])
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
            
        elif class_idx == 3:  # Traffic Light (RED Signal)
            draw.rectangle([90, 30, 134, 135], fill=(20, 20, 22), outline=(80, 80, 80), width=2)
            draw.rectangle([108, 135, 116, 215], fill=(60, 60, 60))
            # Active RED lamp at top, other two lamps off
            draw.ellipse([98, 40, 126, 68], fill=(255, 25, 25))
            draw.ellipse([98, 70, 126, 98], fill=(50, 40, 10))   # yellow off
            draw.ellipse([98, 100, 126, 128], fill=(10, 45, 15))  # green off
            
        elif class_idx == 4:  # Traffic Light (YELLOW / Amber Signal)
            draw.rectangle([90, 30, 134, 135], fill=(20, 20, 22), outline=(80, 80, 80), width=2)
            draw.rectangle([108, 135, 116, 215], fill=(60, 60, 60))
            # Active YELLOW lamp in center, other two lamps off
            draw.ellipse([98, 40, 126, 68], fill=(55, 15, 15))   # red off
            draw.ellipse([98, 70, 126, 98], fill=(255, 210, 0))  # yellow ON
            draw.ellipse([98, 100, 126, 128], fill=(10, 45, 15))  # green off
            
        elif class_idx == 5:  # Traffic Light (GREEN Signal)
            draw.rectangle([90, 30, 134, 135], fill=(20, 20, 22), outline=(80, 80, 80), width=2)
            draw.rectangle([108, 135, 116, 215], fill=(60, 60, 60))
            # Active GREEN lamp at bottom, other two lamps off
            draw.ellipse([98, 40, 126, 68], fill=(55, 15, 15))   # red off
            draw.ellipse([98, 70, 126, 98], fill=(50, 40, 10))   # yellow off
            draw.ellipse([98, 100, 126, 128], fill=(30, 250, 45)) # green ON
            
        elif class_idx == 6:  # Traffic Sign
            draw.rectangle([108, 105, 116, 215], fill=(130, 130, 130))
            draw.polygon([(82, 45), (102, 30), (122, 30), (142, 45), (142, 68), (122, 83), (102, 83), (82, 68)], fill=(215, 20, 20))
            draw.text((94, 49), "STOP", fill=(255, 255, 255))
                
        elif class_idx == 7:  # Truck / Heavy Commercial Vehicle
            truck_color = random.choice([(235, 130, 25), (35, 110, 215), (230, 230, 230)])
            draw.rectangle([35, 60, 155, 160], fill=truck_color)
            draw.rectangle([155, 88, 192, 160], fill=(truck_color[0]//2, truck_color[1]//2, truck_color[2]//2))
            draw.ellipse([50, 148, 78, 176], fill=(20, 20, 20))
            draw.ellipse([115, 148, 143, 176], fill=(20, 20, 20))
            draw.ellipse([160, 148, 188, 176], fill=(20, 20, 20))
            
        elif class_idx == 8:  # Motorcycle / Scooter
            draw.ellipse([65, 125, 96, 156], outline=(35, 35, 35), width=4)
            draw.ellipse([128, 125, 159, 156], outline=(35, 35, 35), width=4)
            draw.polygon([(85, 135), (112, 108), (136, 130)], fill=(225, 45, 45))
            draw.ellipse([108, 75, 128, 95], fill=(25, 25, 25))
            draw.line([114, 95, 118, 125], fill=(30, 75, 160), width=5)
            
        elif class_idx == 9:  # Road Obstacle / Construction Hazard
            draw.polygon([(92, 70), (132, 70), (148, 168), (76, 168)], fill=(255, 110, 15))
            draw.rectangle([88, 98, 136, 114], fill=(245, 245, 245))
            draw.rectangle([83, 132, 141, 148], fill=(245, 245, 245))

        elif class_idx == 10:  # Forest Road / Sunlight Glare (Non-Traffic Light)
            # Render forest trees, tree branches, and natural sun glare (NOT a traffic light)
            img = Image.new("RGB", (224, 224), color=(random.randint(30, 60), random.randint(55, 95), random.randint(30, 50)))
            draw = ImageDraw.Draw(img)
            # Tree trunks
            draw.rectangle([30, 0, 55, 224], fill=(60, 45, 35))
            draw.rectangle([165, 0, 195, 224], fill=(55, 40, 30))
            # Foliage clusters
            for _ in range(8):
                fx = random.randint(0, 200)
                fy = random.randint(0, 140)
                draw.ellipse([fx, fy, fx + 60, fy + 50], fill=(random.randint(30, 70), random.randint(80, 130), random.randint(30, 60)))
            # Diffuse sunlight flare / sunbeams (white/pale yellow washed out, no housing)
            sun_x = random.randint(80, 160)
            sun_y = random.randint(40, 100)
            draw.ellipse([sun_x - 35, sun_y - 35, sun_x + 35, sun_y + 35], fill=(255, 252, 230))
            draw.ellipse([sun_x - 55, sun_y - 55, sun_x + 55, sun_y + 55], outline=(255, 248, 200), width=4)

        if random.random() > 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
            
        return img

    def _build_dataset(self):
        # 1. Procedural domain samples across all 11 classes
        for class_idx in range(len(AUTONOMOUS_CLASSES)):
            for _ in range(self.samples_per_class):
                img = self._generate_synthetic_sample(class_idx)
                self.samples.append((img, class_idx))

        # 2. Extract crops from real dataset images (strictly mapped, no random traffic-light or blind fallback)
        if os.path.exists(self.data_dir):
            real_files = sorted([
                f for f in os.listdir(self.data_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            ])
            for rf in real_files:
                try:
                    rf_lower = rf.lower()
                    c_idx = None
                    if any(k in rf_lower for k in ["pedestrian", "person", "crosswalk"]):
                        c_idx = 1
                    elif any(k in rf_lower for k in ["cyclist", "bicycle", "bike"]):
                        c_idx = 2
                    elif any(k in rf_lower for k in ["red_light", "traffic_light_red"]):
                        c_idx = 3
                    elif any(k in rf_lower for k in ["yellow_light", "amber_light", "traffic_light_yellow"]):
                        c_idx = 4
                    elif any(k in rf_lower for k in ["green_light", "traffic_light_green"]):
                        c_idx = 5
                    elif any(k in rf_lower for k in ["stop_sign", "speed_sign"]):
                        c_idx = 6
                    elif any(k in rf_lower for k in ["truck", "bus", "ambulance"]):
                        c_idx = 7
                    elif any(k in rf_lower for k in ["motorcycle", "scooter"]):
                        c_idx = 8
                    elif any(k in rf_lower for k in ["construction", "barrier", "hazard"]):
                        c_idx = 9
                    elif any(k in rf_lower for k in ["forest", "foggy_rural", "sunset"]):
                        # Nature / sunlight glare negative class
                        c_idx = 10
                    elif any(k in rf_lower for k in ["car", "highway_sunny", "city_downtown", "rainy_night", "night_highway", "electric_vehicle"]):
                        c_idx = 0

                    if c_idx is None:
                        continue

                    img_path = os.path.join(self.data_dir, rf)
                    with Image.open(img_path) as raw_img:
                        real_img = raw_img.convert("RGB")
                    w, h = real_img.size

                    crops = []
                    for _ in range(4):
                        crop_w = int(w * random.uniform(0.45, 0.85))
                        crop_h = int(h * random.uniform(0.45, 0.85))
                        x1 = random.randint(0, max(0, w - crop_w))
                        y1 = random.randint(0, max(0, h - crop_h))
                        crop = real_img.crop((x1, y1, x1 + crop_w, y1 + crop_h)).resize((224, 224))
                        crops.append((crop, c_idx))
                    self.real_crop_groups[rf] = crops
                    self.samples.extend(crops)
                except Exception:
                    pass

        random.shuffle(self.samples)
        print(f"[Dataset] Compiled {len(self.samples)} augmented autonomous driving training samples across {len(AUTONOMOUS_CLASSES)} classes.")

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
    Trains Model 1 on autonomous driving classes including Red, Yellow, Green traffic lights
    and sunlight/forest negative samples.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = get_device()
    print(f"[Training] Using compute device: {device}")

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

    full_dataset = AutonomousDrivingDataset(samples_per_class=160, transform=None)

    # Hold out whole real-image crop groups cleanly so sibling crops never leak into val
    real_files = list(full_dataset.real_crop_groups.keys())
    random.shuffle(real_files)
    val_file_count = max(1, int(0.15 * len(real_files))) if real_files else 0
    val_real_files = set(real_files[:val_file_count])

    train_samples, val_samples = [], []
    real_sample_ids = set()
    for rf, crops in full_dataset.real_crop_groups.items():
        for pair in crops:
            real_sample_ids.add(id(pair))
        if rf in val_real_files:
            val_samples.extend(crops)
        else:
            train_samples.extend(crops)

    synth_only = [s for s in full_dataset.samples if id(s) not in real_sample_ids]
    random.shuffle(synth_only)
    synth_split = int(0.85 * len(synth_only))
    train_samples.extend(synth_only[:synth_split])
    val_samples.extend(synth_only[synth_split:])
    random.shuffle(train_samples)

    train_set = _TransformSubset(train_samples, transform=train_transform)
    val_set = _TransformSubset(val_samples, transform=val_transform)

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
