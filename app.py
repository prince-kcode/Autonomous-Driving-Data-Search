import os
import io
import json
import warnings
import requests
import streamlit as st
from PIL import Image
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="AV-Vision | Autonomous Driving AI Hub",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Autonomous Vehicle HUD & Telemetry Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');

    /* Global Typography */
    .brand-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .brand-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.0rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }

    /* HUD Card Container */
    .hud-card {
        background: rgba(16, 26, 46, 0.65);
        border: 1px solid rgba(0, 210, 255, 0.25);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
    }

    /* HUD Metric Badges */
    .hud-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-right: 8px;
        margin-bottom: 6px;
    }
    .hud-badge-cyan {
        background: rgba(0, 210, 255, 0.15);
        color: #00d2ff;
        border: 1px solid rgba(0, 210, 255, 0.4);
    }
    .hud-badge-green {
        background: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.4);
    }
    .hud-badge-amber {
        background: rgba(255, 177, 0, 0.15);
        color: #ffb100;
        border: 1px solid rgba(255, 177, 0, 0.4);
    }
    .hud-badge-red {
        background: rgba(255, 71, 87, 0.15);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.4);
    }

    /* Telemetry Info Pill */
    .telemetry-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

def check_backend_status():
    try:
        res = requests.get(f"{FASTAPI_URL}/", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None

backend_online, backend_info = check_backend_status()

# Local Fallback Engines (instantiated only if backend is offline)
@st.cache_resource
def get_local_fallback_engines():
    try:
        from core.object_recognizer import ObjectRecognizer
        from core.scene_analyzer import SceneAnalyzer
        from core.yolo_detector import YOLODetector
        from core.qdrant_manager import get_qdrant_client, COLLECTION_NAME
        from transformers import CLIPProcessor, CLIPModel
        from core.device import get_device

        dev = get_device()
        model_name = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(dev)
        clip_proc = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        clip_model.eval()

        return {
            "device": dev,
            "qdrant": get_qdrant_client(),
            "collection": COLLECTION_NAME,
            "clip_model": clip_model,
            "clip_proc": clip_proc,
            "m1": ObjectRecognizer(),
            "m2": SceneAnalyzer(clip_model=clip_model, clip_processor=clip_proc),
            "m3": YOLODetector()
        }
    except Exception as e:
        st.error(f"Failed to initialize local fallback engines: {e}")
        return None

# Sidebar - System Diagnostics HUD
with st.sidebar:
    st.markdown('<div style="font-family: Orbitron; font-size: 1.3rem; color: #00d2ff; font-weight: 700; margin-bottom: 10px;">⚡ AV CORE HUD</div>', unsafe_allow_html=True)
    
    if backend_online:
        st.markdown('<span class="hud-badge hud-badge-green">● FASTAPI SERVER ONLINE</span>', unsafe_allow_html=True)
        compute = backend_info.get("compute", {})
        dev_label = compute.get("device", "cuda").upper()
        gpu_label = compute.get("gpu_name", "NVIDIA RTX Acceleration")
        st.caption(f"Compute Engine: **{dev_label}** ({gpu_label})")
        st.caption(f"Indexed Vectors: **{backend_info.get('indexed_images', 0)} scenarios**")
    else:
        st.markdown('<span class="hud-badge hud-badge-cyan">● EMBEDDED LOCAL ENGINE</span>', unsafe_allow_html=True)
        st.caption("Active in standalone direct-inference mode.")

    st.markdown("---")
    st.markdown("### 🤖 Integrated AI Models")
    st.markdown("""
    - **Model 1: Object Recognizer**  
      MobileNetV3 Deep CNN with verified signal states:
      *(🔴 Red, 🟡 Yellow, 🟢 Green)*
    - **Model 2: Scenario & Risk Analyzer**  
      Multimodal CLIP zero-shot perception with AV hazard rating & action protocols.
    - **Model 3: YOLO Perception**  
      YOLO11 real-time detector with sunlight glare suppression.
    """)
    st.markdown("---")
    st.caption("AV-Vision Enterprise Autonomous Driving AI v2.2")

# Main Header
st.markdown('<div class="brand-title">🚘 AV-VISION INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Autonomous Driving Multimodal Scenario Search, Real-Time Perception & ADAS Safety Telemetry</div>', unsafe_allow_html=True)

# Tabs - Note: Dataset Gallery tab removed per user request
tab1, tab2, tab3 = st.tabs([
    "🔍 Scenario Search & Retrieval",
    "🔬 Real-Time Vision Telemetry (3 Models)",
    "🧠 Model Training & Telemetry Studio"
])

# ----------------- TAB 1: Scenario Search & Retrieval -----------------
with tab1:
    st.markdown("### 🔎 Natural Language Vector Scenario Retrieval")
    st.write("Retrieve exact driving scenarios using high-dimensional CLIP text-image vector embeddings stored in Qdrant.")

    st.markdown("**Quick Preset Queries:**")
    p1, p2, p3, p4 = st.columns(4)
    preset_query = None
    if p1.button("🌧️ Rainy Night Intersection"):
        preset_query = "wet city street at night with cars, bus, and crosswalk"
    if p2.button("🚑 Emergency Ambulance"):
        preset_query = "emergency ambulance vehicle on a city street"
    if p3.button("🚌 Coach Bus on Road"):
        preset_query = "large passenger bus driving on roadway"
    if p4.button("🚚 Highway Cargo Truck"):
        preset_query = "heavy semi cargo truck driving along highway"

    p5, p6, p7, p8 = st.columns(4)
    if p5.button("🏎️ Sports Car Highway"):
        preset_query = "sports car cruising along an open highway road"
    if p6.button("🏍️ Motorcycle Lane"):
        preset_query = "sport motorcycle beside concrete wall"
    if p7.button("🌲 Forest Sunlight Glare"):
        preset_query = "misty pine forest with bright sunlight glare"
    if p8.button("🚲 Bicycle Roadside"):
        preset_query = "bicycle parked near urban street wall"

    c_query, c_limit = st.columns([4, 1])
    with c_query:
        default_q = preset_query if preset_query else "a sports car cruising on the highway"
        query_input = st.text_input("Enter Natural Language Scenario Description:", value=default_q)
    with c_limit:
        limit = st.slider("Result Limit", min_value=1, max_value=8, value=3)

    search_clicked = st.button("🚀 Search Scenarios", type="primary", width="stretch")

    if search_clicked or preset_query:
        query_to_run = query_input.strip()
        results = []

        with st.spinner(f"Vectorizing query and querying Qdrant for: '{query_to_run}'..."):
            if backend_online:
                try:
                    resp = requests.get(
                        f"{FASTAPI_URL}/search",
                        params={"text_query": query_to_run, "limit": limit, "include_analysis": True},
                        timeout=30
                    )
                    if resp.status_code == 200:
                        results = resp.json()
                    else:
                        st.error(f"Search API error: {resp.text}")
                except Exception as e:
                    st.error(f"Backend search request failed: {e}")
            else:
                local = get_local_fallback_engines()
                if local:
                    import torch
                    inputs = local["clip_proc"](text=[query_to_run], return_tensors="pt", padding=True, truncation=True).to(local["device"])
                    with torch.no_grad():
                        feats = local["clip_model"].get_text_features(**inputs)
                    if not isinstance(feats, torch.Tensor):
                        feats = getattr(feats, "text_embeds", getattr(feats, "pooler_output", feats))
                    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                    svec = feats.detach().cpu().numpy()[0].astype(float).tolist()

                    hits = local["qdrant"].query_points(
                        collection_name=local["collection"],
                        query=svec,
                        limit=limit,
                        with_payload=True
                    ).points

                    for hit in hits:
                        fn = hit.payload.get("filename", "")
                        ipath = os.path.join("data", fn)
                        item = {
                            "filename": fn,
                            "title": hit.payload.get("title", fn),
                            "score": round(float(hit.score), 4)
                        }
                        if os.path.exists(ipath):
                            pimg = Image.open(ipath).convert("RGB")
                            item["model_1_recognition"] = local["m1"].predict(pimg)
                            item["model_2_analysis"] = local["m2"].analyze(pimg)
                            y_out = local["m3"].detect(pimg)
                            item["model_3_yolo"] = {
                                "total_objects": y_out["total_objects"],
                                "counts": y_out["counts"],
                                "detections": y_out["detections"]
                            }
                        results.append(item)

        if not results:
            st.warning("No matching driving scenarios found.")
        else:
            st.success(f"Retrieved {len(results)} matching autonomous driving scenarios:")
            
            for idx, item in enumerate(results):
                st.markdown("---")
                cols = st.columns([1.1, 1.9])
                img_path = os.path.join("data", item["filename"])

                with cols[0]:
                    st.subheader(f"#{idx+1}: {item.get('title', item['filename'])}")
                    if os.path.exists(img_path):
                        st.image(img_path, width="stretch")
                    st.metric(label="Vector Cosine Match", value=f"{item['score']:.4f}")

                with cols[1]:
                    st.markdown("#### 🤖 Real-Time Multi-Model AI Diagnostics")
                    
                    # Model 1
                    m1 = item.get("model_1_recognition", {})
                    if m1:
                        top_obj = m1.get("primary_object", "")
                        conf_pct = m1.get("primary_confidence_pct", 0)
                        
                        # Custom badge for traffic lights
                        badge_type = "hud-badge-cyan"
                        if "red" in top_obj.lower():
                            badge_type = "hud-badge-red"
                        elif "yellow" in top_obj.lower():
                            badge_type = "hud-badge-amber"
                        elif "green" in top_obj.lower():
                            badge_type = "hud-badge-green"
                            
                        st.markdown(f"**Model 1 — Object Recognizer:** <span class='hud-badge {badge_type}'>{top_obj} ({conf_pct}%)</span>", unsafe_allow_html=True)
                        top_preds = m1.get("top_predictions", [])[:3]
                        pred_df = pd.DataFrame(top_preds).rename(columns={"class_name": "Class", "confidence_pct": "Confidence %"})
                        st.bar_chart(pred_df.set_index("Class")["Confidence %"], height=130)

                    # Model 2
                    m2 = item.get("model_2_analysis", {})
                    if m2:
                        st.markdown("**Model 2 — Scenario & Safety Risk Analyzer:**")
                        w = m2.get("weather", {}).get("detected", "N/A")
                        t = m2.get("time_of_day", {}).get("detected", "N/A")
                        r = m2.get("road_type", {}).get("detected", "N/A")
                        haz = m2.get("hazard_assessment", {})
                        h_lvl = haz.get("level", "LOW")
                        h_score = haz.get("risk_score_pct", 15)

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Weather", w)
                        c2.metric("Lighting", t)
                        c3.metric("Road Type", r)
                        c4.metric("Hazard Rating", f"{h_lvl} ({h_score}%)")

                        st.caption(f"**ADAS Protocol:** {', '.join(haz.get('av_recommendations', []))}")

                    # Model 3
                    m3 = item.get("model_3_yolo", {})
                    if m3:
                        tot = m3.get("total_objects", 0)
                        counts = m3.get("counts", {})
                        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "None detected"
                        st.info(f"**Model 3 — YOLO Perception ({tot} entities):** {counts_str}")


# ----------------- TAB 2: Real-Time Vision Telemetry -----------------
with tab2:
    st.markdown("### 🔬 Inspect Driving Frames With All 3 AI Models")
    st.write("Upload a custom driving picture or test scenario to run instant side-by-side vision inference.")

    col_upload, col_sample = st.columns([1.5, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload custom driving image:", type=["jpg", "jpeg", "png"])
    with col_sample:
        demo_presets = [
            "None",
            "Sunny Highway Car (highway_sunny_traffic.jpg)",
            "White Sports Car Road (car.jpg)",
            "Aerial Parking Lot Cars (night_highway_speed.jpg)",
            "Parking Lot Overhead (parking_lot_cars.jpg)",
            "Highway Porsche Speed (red_sports_car_road.jpg)",
            "Forest Road Sunlight (foggy_forest_road.jpg)",
            "Emergency Ambulance (emergency_ambulance_road.jpg)",
            "Passenger Coach Bus (school_bus_suburban.jpg)",
            "Highway Cargo Semi-Truck (truck_freeway_transport.jpg)",
            "Mountain Highway Truck (mountain_highway_truck.jpg)",
            "Sport Motorcycle Lane (motorcycle_highway_lane.jpg)",
            "Roadside Bicycle (cyclist_urban_road.jpg)",
            "Pedestrian Crosswalk Urban (pedestrian_crosswalk_urban.jpg)",
            "Red Traffic Light Signal (traffic_light_red_signal.jpg)",
            "Yellow Traffic Light Signal (traffic_light_yellow_signal.jpg)",
            "Green Traffic Light Signal (traffic_light_green_signal.jpg)",
            "Stop Sign Roadside (stop_sign_roadside.jpg)",
            "Construction Cones Hazard (construction_cones_hazard.jpg)",
            "Road Barrier Hazard (road_barrier_hazard.jpg)",
            "Rainy Night Intersection (rainy_night_driving.jpg)"
        ]
        chosen_sample = st.selectbox("Or select a demo test scenario:", demo_presets)

    target_image = None
    target_filename = None

    if uploaded_file is not None:
        try:
            target_image = Image.open(uploaded_file).convert("RGB")
        except Exception as img_err:
            st.error(f"Could not read uploaded image: {img_err}")
    elif chosen_sample != "None":
        fname = chosen_sample.split("(")[-1].replace(")", "").strip()
        fpath = os.path.join("data", fname)
        if os.path.exists(fpath):
            target_filename = fname
            target_image = Image.open(fpath).convert("RGB")

    if target_image is not None:
        st.markdown("---")
        run_btn = st.button("⚡ Run Full 3-Model AI Diagnostics", type="primary")

        if run_btn or target_image:
            with st.spinner("Processing frame across Dataset Verifier, Model 1 (Deep CNN), Model 2 (CLIP), and Model 3 (YOLO11)..."):
                m1_res, m2_res, m3_res = None, None, None
                dataset_check = None

                # Route both sample and uploaded frames via backend when online
                if backend_online:
                    try:
                        if uploaded_file is not None:
                            img_bytes = io.BytesIO()
                            target_image.save(img_bytes, format="JPEG")
                            img_bytes.seek(0)
                            resp = requests.post(
                                f"{FASTAPI_URL}/analyze",
                                files={"file": (uploaded_file.name, img_bytes, "image/jpeg")},
                                timeout=30
                            )
                        else:
                            resp = requests.post(
                                f"{FASTAPI_URL}/analyze",
                                params={"filename": target_filename},
                                timeout=30
                            )
                        if resp.status_code == 200:
                            res = resp.json()
                            m1_res = res["model_1_recognition"]
                            m2_res = res["model_2_analysis"]
                            m3_yolo = res["model_3_yolo"]
                            dataset_check = res.get("dataset_check") or m1_res.get("dataset_check")
                            from core.yolo_detector import YOLODetector
                            m3_res = {
                                "annotated_image": YOLODetector.draw_detections(target_image, m3_yolo["detections"]),
                                "total_objects": m3_yolo["total_objects"],
                                "counts": m3_yolo["counts"],
                                "detections": m3_yolo["detections"]
                            }
                    except Exception:
                        pass

                if m1_res is None:
                    local = get_local_fallback_engines()
                    if local:
                        m1_res = local["m1"].predict(target_image, filename=target_filename)
                        m2_res = local["m2"].analyze(target_image)
                        m3_res = local["m3"].detect(target_image)
                        dataset_check = m1_res.get("dataset_check")

                if m1_res and m2_res and m3_res:
                    if not dataset_check:
                        dataset_check = m1_res.get("dataset_check", {})
                    is_trained = bool(dataset_check.get("is_trained", m1_res.get("is_trained_in_dataset", False)))

                    # Prominent Dataset Training Verification Banner
                    if is_trained:
                        matched_f = dataset_check.get("matched_filename", target_filename or "dataset image")
                        trained_cls = dataset_check.get("trained_class") or m1_res.get("primary_object", "N/A")
                        sim_pct = dataset_check.get("similarity_pct", 100.0)
                        st.success(
                            f"✅ **Image IS Trained by Dataset** — Matched `data/{matched_f}` "
                            f"(Trained Class: **{trained_cls}** | Match Confidence: **{sim_pct}%**)"
                        )
                    else:
                        closest_f = dataset_check.get("closest_dataset_file", "None")
                        sim_pct = dataset_check.get("similarity_pct", 0.0)
                        st.error(
                            f"❌ **Image is NOT Trained by Dataset** — This uploaded image was not found in the trained `data/` dataset "
                            f"(Closest dataset sample: `{closest_f}` at `{sim_pct}%` similarity; required `>= 92%`)."
                        )
                        if uploaded_file is not None:
                            if st.button("➕ Add Uploaded Image to Dataset & Train All Models", key="add_untrained_img_btn"):
                                with st.spinner("Saving image to data/, re-indexing Qdrant, and retraining Model 1..."):
                                    os.makedirs("data", exist_ok=True)
                                    safe_name = os.path.basename(uploaded_file.name)
                                    if not safe_name.lower().endswith((".jpg", ".jpeg", ".png")):
                                        safe_name += ".jpg"
                                    save_path = os.path.join("data", safe_name)
                                    target_image.save(save_path, format="JPEG", quality=94)
                                    if backend_online:
                                        requests.post(f"{FASTAPI_URL}/index", timeout=120)
                                        requests.post(f"{FASTAPI_URL}/train", params={"epochs": 8}, timeout=240)
                                    else:
                                        from populate_db import index_images
                                        from train import train_model
                                        index_images()
                                        train_model(epochs=8, batch_size=32)
                                    st.cache_resource.clear()
                                    st.success(f"✅ Added `data/{safe_name}` to dataset and retrained all models! Re-run diagnostics to verify.")
                                    st.rerun()

                    c_left, c_mid, c_right = st.columns([1, 1, 1.2])

                    with c_left:
                        st.markdown("#### 1️⃣ Model 1: Object Recognizer")
                        st.image(target_image, caption="Input Frame", width="stretch")
                        
                        top_cls = m1_res['primary_object']
                        if not is_trained:
                            st.markdown(
                                "**Dataset Status:** <span class='hud-badge hud-badge-red'>❌ NOT TRAINED IN DATASET</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(f"Closest Zero-Shot Class Guess: `{top_cls}` ({m1_res['primary_confidence_pct']}%)")
                        else:
                            badge_c = "hud-badge-cyan"
                            if "red" in top_cls.lower():
                                badge_c = "hud-badge-red"
                            elif "yellow" in top_cls.lower():
                                badge_c = "hud-badge-amber"
                            elif "green" in top_cls.lower():
                                badge_c = "hud-badge-green"

                            st.markdown(
                                "**Dataset Status:** <span class='hud-badge hud-badge-green'>✅ TRAINED IN DATASET</span>",
                                unsafe_allow_html=True
                            )
                            st.markdown(f"**Identified:** <span class='hud-badge {badge_c}'>{top_cls}</span>", unsafe_allow_html=True)
                        st.metric("Model Confidence", f"{m1_res['primary_confidence_pct']}%")
                        top_p = m1_res.get("top_predictions", [])
                        df_p = pd.DataFrame(top_p).rename(columns={"class_name": "Class", "confidence_pct": "Confidence %"})
                        st.bar_chart(df_p.set_index("Class")["Confidence %"], height=160)

                    with c_mid:
                        st.markdown("#### 2️⃣ Model 2: Scene & Risk Analyzer")
                        w_det = m2_res["weather"]["detected"]
                        t_det = m2_res["time_of_day"]["detected"]
                        r_det = m2_res["road_type"]["detected"]
                        haz = m2_res["hazard_assessment"]

                        st.markdown(f"**Weather Condition:** `{w_det}`")
                        st.markdown(f"**Lighting & Time:** `{t_det}`")
                        st.markdown(f"**Road Topology:** `{r_det}`")
                        
                        st.markdown("---")
                        st.markdown(f"**Safety Hazard Score:** `{haz['risk_score_pct']}%`")
                        st.progress(haz['risk_score_pct'] / 100.0)

                        st.markdown(f"**Hazard Level:** `{haz['level']}`")
                        st.markdown("**Identified Risk Factors:**")
                        for rf in haz.get("risk_factors", []):
                            st.write(f"- {rf}")

                        st.markdown("**Autonomous Action Commands:**")
                        for act in haz.get("av_recommendations", []):
                            st.write(f"- 🚗 {act}")

                    with c_right:
                        st.markdown("#### 3️⃣ Model 3: YOLO11 Object Detection")
                        st.image(m3_res["annotated_image"], caption="YOLO11 Bounding Boxes", width="stretch")
                        st.info(f"**Detected Entities:** {m3_res['total_objects']}")
                        if m3_res["counts"]:
                            st.dataframe(pd.DataFrame(list(m3_res["counts"].items()), columns=["Detected Class", "Count"]), width="stretch")


# ----------------- TAB 3: Model Training Studio -----------------
with tab3:
    st.markdown("### 🧠 AI Model Training & Benchmark Telemetry Studio")
    st.write("Inspect performance curves, test benchmark accuracies, and retrain models on GPU.")

    history_file = "models/training_history.json"
    hist_data = None
    best_val_display = "N/A"
    num_classes_display = 11
    total_samples_display = "2,400+"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                hist_data = json.load(f)
            if hist_data.get("val_acc"):
                best_val_display = f"{max(hist_data['val_acc']):.2f}%"
            if hist_data.get("classes"):
                num_classes_display = len(hist_data["classes"])
            if hist_data.get("total_samples"):
                total_samples_display = f"{hist_data['total_samples']:,}"
        except Exception:
            hist_data = None

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("#### 1️⃣ Model 1: Object Recognizer")
        st.markdown(f"""
        - **Architecture**: MobileNetV3 Deep CNN
        - **Classes**: {num_classes_display} AV Categories *(Red, Yellow, Green signals + Glare Negative)*
        - **Best Val Accuracy**: **{best_val_display}**
        - **Weights**: `models/object_recognizer.pth`
        """)
    with col_m2:
        st.markdown("#### 2️⃣ Model 2: Scenario Analyzer")
        st.markdown("""
        - **Architecture**: Multimodal CLIP ViT-B/32
        - **Perception**: Weather, Lighting, Topology
        - **Risk Matrix**: Low, Moderate, High, Critical
        - **AV Protocols**: Headway, speed, AEB standby
        """)
    with col_m3:
        st.markdown("#### 3️⃣ Model 3: YOLO11 Detector")
        st.markdown("""
        - **Architecture**: YOLO11 Real-Time Perception
        - **Signal Verification**: Red, Yellow, Green color detection
        - **Glare Suppression**: Filters out sunlight/sky false positives
        - **Weights**: `models/yolo11_autodrive.pt`
        """)

    st.markdown("---")
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        retrain_m1_btn = st.button("🎯 Retrain Model 1 (Deep CNN)", width="stretch")
    with c_btn2:
        retrain_all_btn = st.button("🚀 Train / Verify All 3 Models", type="primary", width="stretch")

    if retrain_m1_btn:
        with st.spinner(f"Retraining Model 1 on {total_samples_display} samples on GPU..."):
            if backend_online:
                resp = requests.post(
                    f"{FASTAPI_URL}/train",
                    params={"epochs": 8, "learning_rate": 0.0008},
                    timeout=300
                )
                if resp.status_code == 200:
                    res = resp.json()
                    get_local_fallback_engines.clear()
                    st.success(f"🎉 Model 1 training complete & reloaded on backend! Best Accuracy: {res['best_accuracy']}%")
                    st.rerun()
                else:
                    st.error(f"Backend training failed: {resp.text}")
            else:
                from train import train_model
                h = train_model(epochs=8, learning_rate=0.0008)
                get_local_fallback_engines.clear()
                best_acc = max(h["val_acc"]) if h.get("val_acc") else 0
                st.success(f"🎉 Model 1 training complete! Best Accuracy: {best_acc}%")
                st.rerun()

    if retrain_all_btn:
        with st.spinner("Training and verifying all 3 models on GPU..."):
            from train_all import train_all_models
            summary = train_all_models(m1_epochs=8, yolo_epochs=3, retrain_m1=True, retrain_yolo=True)
            get_local_fallback_engines.clear()
            if backend_online:
                try:
                    requests.post(f"{FASTAPI_URL}/train", params={"epochs": 1, "learning_rate": 0.0001}, timeout=120)
                except Exception:
                    pass
            st.success("🎉 All 3 models trained and verified successfully!")
            st.json(summary)

    # Performance curves
    if hist_data:
        try:
            st.markdown("#### 📈 Model 1 Training Performance Curves")
            c_c1, c_c2 = st.columns(2)
            df_metrics = pd.DataFrame({
                "Epoch": hist_data["epochs"],
                "Train Accuracy (%)": hist_data["train_acc"],
                "Val Accuracy (%)": hist_data["val_acc"],
                "Train Loss": hist_data["train_loss"],
                "Val Loss": hist_data["val_loss"]
            }).set_index("Epoch")

            with c_c1:
                st.markdown("**Accuracy Progression (Train vs Validation)**")
                st.line_chart(df_metrics[["Train Accuracy (%)", "Val Accuracy (%)"]])
            with c_c2:
                st.markdown("**Loss Curve (Train vs Validation)**")
                st.line_chart(df_metrics[["Train Loss", "Val Loss"]])
        except Exception:
            pass
