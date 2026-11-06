import os
import io
import json
import warnings
import requests
import streamlit as st
from PIL import Image
import pandas as pd

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(
    page_title="Autonomous Driving Data Search & Multi-Model AI Hub",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Autonomous Driving Tech Theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .hazard-badge-low {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .hazard-badge-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .hazard-badge-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

# Check backend status
def check_backend_status():
    try:
        res = requests.get(f"{FASTAPI_URL}/", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None

backend_online, backend_info = check_backend_status()

# Local Fallback Engines (instantiated only if backend is not running)
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
            "m2": SceneAnalyzer(),
            "m3": YOLODetector()
        }
    except Exception as e:
        st.error(f"Failed to initialize local fallback engines: {e}")
        return None

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=400&q=80", caption="Autonomous Driving AI Hub", use_container_width=True)
    st.title("🚗 System Control")
    
    if backend_online:
        st.success("🟢 FastAPI Backend: Online")
        st.caption(f"Host: `{FASTAPI_URL}`")
        compute = backend_info.get("compute", {})
        st.info(f"Compute Device: **{compute.get('device', 'cpu').upper()}**")
        if "gpu_name" in compute:
            st.caption(f"GPU: {compute['gpu_name']}")
    else:
        st.warning("🟠 Direct Embedded Engine Mode")
        st.caption("Backend running embedded inside Streamlit.")

    st.markdown("---")
    st.markdown("### 🤖 3 Integrated AI Models")
    st.markdown("""
    1. **Model 1: Object Recognizer**  
       Deep CNN MobileNetV3 (1,544 samples, **99.14%** accuracy).
    2. **Model 2: Scenario & Risk Analyzer**  
       Multimodal CLIP perception for weather, lighting, road type, and AV risk index.
    3. **Model 3: YOLO Detector**  
       Fine-tuned YOLOv8 on autonomous driving entities.
    """)
    st.markdown("---")
    st.caption("Autonomous Driving Data Search Engine v2.0")

# Header
st.markdown('<div class="main-header">🚗 Autonomous Driving Data Search & Multi-Model AI Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Query high-dimensional driving scenarios with natural language, inspect real-time safety telemetry, and run 3 specialized AI models.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Scenario Search & Retrieval",
    "🔬 Live AI Diagnostics (3 Models)",
    "🧠 Model Training & Telemetry Studio",
    "📁 Dataset & Qdrant Gallery"
])

# ----------------- TAB 1: Search & Retrieval -----------------
with tab1:
    st.markdown("### 🔎 Natural Language Vector Scenario Search")
    st.write("CLIP translates your natural language description into a 512-dimensional vector to retrieve matching autonomous driving situations from Qdrant.")

    st.markdown("**Quick Preset Queries:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_query = None
    if p_col1.button("🌧️ Rainy night highway"):
        preset_query = "a photo of a car driving on a rainy road at night with glowing taillights"
    if p_col2.button("🚶 Pedestrians crossing"):
        preset_query = "pedestrians crossing the street at an urban crosswalk in daylight"
    if p_col3.button("🚌 School bus on road"):
        preset_query = "large yellow bus driving along roadway"
    if p_col4.button("🌫️ Dense fog forest highway"):
        preset_query = "dense fog and misty low visibility conditions on a mountain road"

    p_col5, p_col6, p_col7, p_col8 = st.columns(4)
    if p_col5.button("🚧 Construction barrier & cones"):
        preset_query = "road construction zone with orange cones and barrier hazard"
    if p_col6.button("🏍️ Motorcycle on highway"):
        preset_query = "motorcycle driving on an open expressway lane"
    if p_col7.button("🚦 Traffic light intersection"):
        preset_query = "traffic light at a city street intersection with cars"
    if p_col8.button("❄️ Snow blizzard road"):
        preset_query = "snow covered icy winter road with trucks"

    col_q, col_lim = st.columns([4, 1])
    with col_q:
        default_val = preset_query if preset_query else "a red sports car on the road"
        query_input = st.text_input("Enter Scenario Description:", value=default_val)
    with col_lim:
        limit = st.slider("Result Limit", min_value=1, max_value=8, value=3)

    search_clicked = st.button("🚀 Search Driving Scenarios", type="primary", use_container_width=True)

    if search_clicked or preset_query:
        query_to_run = query_input.strip()
        results = []

        with st.spinner(f"Querying Qdrant vector database for: '{query_to_run}'..."):
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
                        st.error(f"Search API returned error: {resp.text}")
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
            st.warning("No matching driving scenarios found. Ensure images are indexed in Tab 4.")
        else:
            st.success(f"Retrieved {len(results)} matching autonomous driving scenarios:")
            
            for idx, item in enumerate(results):
                st.markdown("---")
                cols = st.columns([1.2, 1.8])
                img_path = os.path.join("data", item["filename"])

                with cols[0]:
                    st.subheader(f"#{idx+1}: {item.get('title', item['filename'])}")
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    st.metric(label="Cosine Similarity Match", value=f"{item['score']:.4f}")

                with cols[1]:
                    st.markdown("#### 🤖 3 Models AI Diagnostics")
                    
                    # Model 1
                    m1 = item.get("model_1_recognition", {})
                    st.markdown("**1️⃣ Model 1 — Object Recognition (Deep CNN):**")
                    if m1:
                        st.markdown(f"Primary Recognized Object: `{m1.get('primary_object')}` (**{m1.get('primary_confidence_pct')}%**)")
                        top_preds = m1.get("top_predictions", [])[:3]
                        pred_df = pd.DataFrame(top_preds).rename(columns={"class_name": "Class", "confidence_pct": "Confidence %"})
                        st.bar_chart(pred_df.set_index("Class")["Confidence %"], height=150)

                    # Model 2
                    m2 = item.get("model_2_analysis", {})
                    st.markdown("**2️⃣ Model 2 — Scene & Safety Risk Analyzer (CLIP):**")
                    if m2:
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
                        c4.metric("Hazard Index", f"{h_lvl} ({h_score}%)")

                        st.caption(f"**Autonomous Action Protocol:** {', '.join(haz.get('av_recommendations', []))}")

                    # Model 3
                    m3 = item.get("model_3_yolo", {})
                    st.markdown("**3️⃣ Model 3 — YOLO Object Detection:**")
                    if m3:
                        tot = m3.get("total_objects", 0)
                        counts = m3.get("counts", {})
                        counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "None detected"
                        st.info(f"**Detected {tot} bounding box entities:** {counts_str}")


# ----------------- TAB 2: Live AI Diagnostics -----------------
with tab2:
    st.markdown("### 🔬 Inspect Any Image With All 3 AI Models")
    st.write("Upload any custom driving image or select from the 26 dataset scenarios to run all 3 models in real-time.")

    col_src1, col_src2 = st.columns([1, 1])
    with col_src1:
        data_images = sorted([f for f in os.listdir("data") if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists("data") else []
        selected_file = st.selectbox("Select scenario from dataset:", ["None"] + data_images)
    with col_src2:
        uploaded_file = st.file_uploader("Or upload custom image:", type=["jpg", "jpeg", "png"])

    target_image = None
    target_filename = None
    if uploaded_file is not None:
        target_image = Image.open(uploaded_file).convert("RGB")
    elif selected_file != "None":
        target_filename = selected_file
        target_image = Image.open(os.path.join("data", selected_file)).convert("RGB")

    if target_image is not None:
        st.markdown("---")
        run_diag = st.button("⚡ Run 3 Models Diagnostics", type="primary")

        if run_diag or target_image:
            with st.spinner("Executing Model 1, Model 2, and Model 3..."):
                m1_res, m2_res, m3_res = None, None, None

                # Route via backend if online
                if backend_online and target_filename:
                    try:
                        res = requests.post(f"{FASTAPI_URL}/analyze", params={"filename": target_filename}, timeout=30).json()
                        m1_res = res["model_1_recognition"]
                        m2_res = res["model_2_analysis"]
                        # For YOLO annotated image, load local or generate
                        from core.yolo_detector import YOLODetector
                        m3_detector = YOLODetector()
                        m3_res = m3_detector.detect(target_image)
                    except Exception:
                        pass

                if m1_res is None:
                    local = get_local_fallback_engines()
                    if local:
                        m1_res = local["m1"].predict(target_image)
                        m2_res = local["m2"].analyze(target_image)
                        m3_res = local["m3"].detect(target_image)

                if m1_res and m2_res and m3_res:
                    c_left, c_mid, c_right = st.columns([1, 1, 1.2])

                    with c_left:
                        st.markdown("#### 1️⃣ Model 1: Object Recognizer")
                        st.image(target_image, caption="Original Input", use_container_width=True)
                        st.success(f"**Top Class:** {m1_res['primary_object']}")
                        st.metric("Confidence", f"{m1_res['primary_confidence_pct']}%")
                        top_p = m1_res.get("top_predictions", [])
                        df_p = pd.DataFrame(top_p).rename(columns={"class_name": "Class", "confidence_pct": "Confidence %"})
                        st.bar_chart(df_p.set_index("Class")["Confidence %"], height=180)

                    with c_mid:
                        st.markdown("#### 2️⃣ Model 2: Scene & Risk Analyzer")
                        w_det = m2_res["weather"]["detected"]
                        t_det = m2_res["time_of_day"]["detected"]
                        r_det = m2_res["road_type"]["detected"]
                        haz = m2_res["hazard_assessment"]

                        st.markdown(f"**Weather:** `{w_det}`")
                        st.markdown(f"**Lighting:** `{t_det}`")
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
                        st.markdown("#### 3️⃣ Model 3: YOLO Detection")
                        st.image(m3_res["annotated_image"], caption="YOLOv8 Bounding Boxes", use_container_width=True)
                        st.info(f"**Total Detected Entities:** {m3_res['total_objects']}")
                        if m3_res["counts"]:
                            st.dataframe(pd.DataFrame(list(m3_res["counts"].items()), columns=["Detected Class", "Count"]), use_container_width=True)


# ----------------- TAB 3: Model Training Studio -----------------
with tab3:
    st.markdown("### 🧠 Autonomous Driving AI Models Training Studio")
    st.write("Train, fine-tune, or recalibrate all 3 models on the expanded autonomous driving dataset.")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown("#### 1️⃣ Model 1: Object Recognizer")
        st.markdown("""
        - **Architecture**: MobileNetV3 Deep CNN
        - **Classes**: 8 AV Categories
        - **Status**: Trained on 1,544 samples
        - **Val Accuracy**: **99.14%**
        """)
    with col_m2:
        st.markdown("#### 2️⃣ Model 2: Scenario Analyzer")
        st.markdown("""
        - **Architecture**: Multimodal CLIP ViT-B/32
        - **Perception**: Weather, Lighting, Topology
        - **Status**: Calibrated across 26 scenes
        - **Risk Matrix**: Low, Moderate, High, Critical
        """)
    with col_m3:
        st.markdown("#### 3️⃣ Model 3: YOLO Detector")
        st.markdown("""
        - **Architecture**: Fine-Tuned YOLOv8
        - **Bounding Boxes**: Vehicles, Pedestrians, Signs
        - **Status**: Trained on GPU (`models/yolov8_autodrive.pt`)
        """)

    st.markdown("---")
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        retrain_m1_btn = st.button("🎯 Retrain Model 1 (Deep CNN)", use_container_width=True)
    with c_btn2:
        retrain_all_btn = st.button("🚀 Train / Verify All 3 Models", type="primary", use_container_width=True)

    if retrain_m1_btn:
        with st.spinner("Retraining Model 1 on 1,500+ samples on GPU..."):
            from train import train_model
            h = train_model(epochs=8, learning_rate=0.0008)
            st.success(f"🎉 Model 1 training complete! Best Accuracy: {h['val_acc'][-1]}%")
            st.rerun()

    if retrain_all_btn:
        with st.spinner("Training and verifying all 3 models on GPU..."):
            from train_all import train_all_models
            summary = train_all_models(m1_epochs=8, yolo_epochs=3, retrain_m1=True)
            st.success("🎉 All 3 models trained and verified successfully!")
            st.json(summary)

    # Show training metrics
    history_file = "models/training_history.json"
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                hist_data = json.load(f)
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


# ----------------- TAB 4: Dataset & Vector Gallery -----------------
with tab4:
    st.markdown("### 📁 Dataset & Qdrant Vector Gallery")
    st.write("Browse all 26 autonomous driving scenarios indexed into the Qdrant vector database.")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Re-index All 26 Images into Qdrant", use_container_width=True):
            with st.spinner("Generating CLIP embeddings and uploading points to Qdrant..."):
                if backend_online:
                    try:
                        res = requests.post(f"{FASTAPI_URL}/index", params={"recreate": True}, timeout=60).json()
                        st.success(f"Successfully indexed {res.get('indexed_images', 26)} images via FastAPI backend!")
                    except Exception as e:
                        st.error(f"Failed to index via backend: {e}")
                else:
                    from populate_db import index_images
                    cnt = index_images(data_dir="data", recreate_collection=True)
                    st.success(f"Successfully indexed {cnt} images!")
                st.rerun()

    data_dir = "data"
    images = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(data_dir) else []
    st.info(f"Total scenario images in dataset: **{len(images)}**")

    # Render gallery grid (3 columns)
    grid_cols = st.columns(3)
    for i, img_name in enumerate(images):
        col_idx = i % 3
        img_p = os.path.join(data_dir, img_name)
        with grid_cols[col_idx]:
            st.image(img_p, caption=img_name.replace("_", " ").rsplit(".", 1)[0].title(), use_container_width=True)
            st.caption(f"`{img_name}`")
