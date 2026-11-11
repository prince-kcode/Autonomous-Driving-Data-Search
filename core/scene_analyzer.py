import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from core.device import get_device

class SceneAnalyzer:
    """
    Model 2: Autonomous Driving Scenario & Risk Analyzer
    Evaluates weather, time/lighting, road typology, safety hazard levels,
    and vehicle control recommendations using multimodal vision-language reasoning.
    """
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", clip_model=None, clip_processor=None):
        self.device = get_device()
        if clip_model is not None and clip_processor is not None:
            self.clip_model = clip_model
            self.clip_processor = clip_processor
        else:
            try:
                self.clip_model = CLIPModel.from_pretrained(clip_model_name, local_files_only=True).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, local_files_only=True)
            except Exception:
                self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

        # Attribute definitions
        self.weather_prompts = [
            ("Clear / Sunny", "a photo of an autonomous driving scene with clear sunny weather and bright visibility"),
            ("Rainy / Wet Road", "a photo of a car driving on a wet rainy road with water reflections and rain"),
            ("Dense Fog / Haze", "a photo of heavy fog and misty low visibility conditions on a highway"),
            ("Snow / Ice", "a photo of a winter road with snow, ice, and cold conditions"),
            ("Overcast / Cloudy", "a photo of cloudy overcast sky during driving")
        ]

        self.time_prompts = [
            ("Daylight", "a photo of driving during bright daytime with clear daylight"),
            ("Dusk / Sunset", "a photo of driving at sunset, twilight, or golden hour with warm sky"),
            ("Night / Dark", "a photo of driving at night in the dark with headlights and streetlights")
        ]

        self.road_type_prompts = [
            ("Expressway / Highway", "a photo of an open multi-lane highway or interstate with fast traffic"),
            ("Urban City Arterial", "a photo of busy downtown city streets with traffic lights and buildings"),
            ("Pedestrian Crossing", "a photo of an urban street with crosswalk and pedestrians walking"),
            ("Suburban / Residential", "a photo of a quiet residential neighborhood road with houses"),
            ("Rural / Mountain Road", "a photo of a rural two-lane winding road through countryside or mountains")
        ]

        self.hazard_prompts = [
            ("Low Hazard", "a photo of safe calm driving on an empty open road with good visibility"),
            ("Moderate Hazard", "a photo of moderate driving traffic, dusk lighting, or minor obstacles"),
            ("High Hazard", "a photo of hazardous driving with rain, night, dense traffic, or pedestrians"),
            ("Critical Hazard", "a photo of dangerous driving hazard, slippery ice, severe low visibility, or sudden obstacle")
        ]

    def _score_prompts(self, image: Image.Image, prompt_pairs: list) -> tuple:
        labels = [p[0] for p in prompt_pairs]
        texts = [p[1] for p in prompt_pairs]

        inputs = self.clip_processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # shape: [1, num_prompts]
            probs = F.softmax(logits_per_image, dim=1)[0].cpu().numpy()

        scored = list(zip(labels, probs))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0], round(float(scored[0][1]), 3), {label: round(float(p), 3) for label, p in scored}

    def analyze(self, image_input) -> dict:
        """
        Runs comprehensive driving scenario and risk analysis.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.fromarray(image_input).convert("RGB")

        # 1. Weather Analysis
        weather_label, weather_conf, weather_all = self._score_prompts(image, self.weather_prompts)

        # 2. Lighting / Time of Day
        time_label, time_conf, time_all = self._score_prompts(image, self.time_prompts)

        # 3. Road / Topology
        road_label, road_conf, road_all = self._score_prompts(image, self.road_type_prompts)

        # 4. Direct Hazard Classification
        hazard_label, hazard_conf, hazard_all = self._score_prompts(image, self.hazard_prompts)

        # 5. Autonomous Safety Risk Assessment Engine
        risk_score = 15  # baseline
        risk_factors = []
        action_recommendations = []

        if "Rainy" in weather_label:
            risk_score += 25
            risk_factors.append("Wet road surface: friction coefficient reduced (~0.4 μ)")
            action_recommendations.append("Engage traction control, increase following distance to 3.5s")
        elif "Dense Fog" in weather_label:
            risk_score += 35
            risk_factors.append("Severe atmospheric scattering: optical camera visibility degraded")
            action_recommendations.append("Reduce cruise speed by 30%, activate fog lamps and radar-guided headway")
        elif "Snow" in weather_label:
            risk_score += 40
            risk_factors.append("Snow / ice layer: high slip hazard")
            action_recommendations.append("Limit acceleration torque, activate winter ABS profile")

        if "Night" in time_label:
            risk_score += 20
            risk_factors.append("Low ambient illumination: relies heavily on active headlamps and lidar")
            action_recommendations.append("Switch to high-beam assist, reduce cornering speeds")
        elif "Dusk" in time_label:
            risk_score += 10
            risk_factors.append("Low sun angle / glare potential")

        if "Pedestrian" in road_label:
            risk_score += 25
            risk_factors.append("High pedestrian vulnerability zone detected")
            action_recommendations.append("Pre-charge AEB (Autonomous Emergency Braking), cap velocity to 25 km/h")
        elif "Urban City" in road_label:
            risk_score += 15
            risk_factors.append("Complex multi-agent urban intersection dynamics")
            action_recommendations.append("Activate 360-degree cross-traffic monitoring")
        elif "Expressway" in road_label:
            action_recommendations.append("Standard Highway Autopilot (Level 3 ADAS) permitted up to 100 km/h")

        if "Critical Hazard" in hazard_label and hazard_conf >= 0.45:
            risk_score += 25
            risk_factors.append(f"CLIP multimodal perception flagged critical hazard ({round(hazard_conf * 100, 1)}% confidence)")
        elif "High Hazard" in hazard_label and hazard_conf >= 0.45:
            risk_score += 15
            risk_factors.append(f"CLIP multimodal perception flagged elevated hazard ({round(hazard_conf * 100, 1)}% confidence)")

        if not action_recommendations:
            action_recommendations.append("Normal autonomous cruise operational (target: 65 km/h, headway: 2.0s)")

        # Normalize risk score to 0 - 100%
        risk_score = min(max(risk_score, 10), 98)

        if risk_score < 30:
            hazard_level = "LOW"
            hazard_color = "#28a745"
        elif risk_score < 60:
            hazard_level = "MODERATE"
            hazard_color = "#ffc107"
        elif risk_score < 80:
            hazard_level = "HIGH"
            hazard_color = "#fd7e14"
        else:
            hazard_level = "CRITICAL"
            hazard_color = "#dc3545"

        summary = (
            f"Autonomous driving scenario identified as {weather_label.lower()} conditions during "
            f"{time_label.lower()} on a {road_label.lower()}. Overall driving risk is assessed as "
            f"{hazard_level} ({risk_score}% index)."
        )

        return {
            "model_name": "Autonomous Scenario & Risk Analyzer (Model 2 - Multimodal CLIP)",
            "weather": {
                "detected": weather_label,
                "confidence": weather_conf,
                "breakdown": weather_all
            },
            "time_of_day": {
                "detected": time_label,
                "confidence": time_conf,
                "breakdown": time_all
            },
            "road_type": {
                "detected": road_label,
                "confidence": road_conf,
                "breakdown": road_all
            },
            "hazard_assessment": {
                "level": hazard_level,
                "color": hazard_color,
                "risk_score_pct": risk_score,
                "visual_hazard_signal": {
                    "detected": hazard_label,
                    "confidence": hazard_conf
                },
                "risk_factors": risk_factors if risk_factors else ["Optimal driving conditions detected"],
                "av_recommendations": action_recommendations
            },
            "scenario_summary": summary
        }
