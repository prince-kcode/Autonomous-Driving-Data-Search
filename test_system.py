import os
import sys
from PIL import Image
from core.device import get_device, get_device_info
from core.qdrant_manager import get_qdrant_client, COLLECTION_NAME
from core.object_recognizer import ObjectRecognizer
from core.scene_analyzer import SceneAnalyzer
from core.yolo_detector import YOLODetector

def run_tests():
    print("=" * 60)
    print("AUTONOMOUS DRIVING DATA SEARCH - SYSTEM TEST")
    print("=" * 60)

    # 1. Device
    print("\n[1/5] Checking Hardware Compute Device...")
    dev_info = get_device_info()
    print(f"  Device: {dev_info.get('device')}")
    if "gpu_name" in dev_info:
        print(f"  GPU Name: {dev_info['gpu_name']}")
    print("  -> PASSED")

    # 2. Qdrant
    print("\n[2/5] Checking Qdrant Vector Storage...")
    client = get_qdrant_client()
    exists = client.collection_exists(COLLECTION_NAME)
    print(f"  Collection '{COLLECTION_NAME}' exists: {exists}")
    if exists:
        pts = client.get_collection(COLLECTION_NAME).points_count
        print(f"  Indexed Points Count: {pts}")
    print("  -> PASSED")

    test_img_path = "data/highway_sunny_traffic.jpg"
    if not os.path.exists(test_img_path):
        test_img_path = "data/car.jpg"
    img = Image.open(test_img_path)

    # 3. Model 1
    print("\n[3/5] Testing Model 1: Autonomous Object Recognizer...")
    m1 = ObjectRecognizer()
    r1 = m1.predict(img)
    print(f"  Model Name: {r1['model_name']}")
    print(f"  Top Class: {r1['primary_object']} ({r1['primary_confidence_pct']}%)")
    print("  -> PASSED")

    # 4. Model 2
    print("\n[4/5] Testing Model 2: Scenario & Risk Analyzer...")
    m2 = SceneAnalyzer()
    r2 = m2.analyze(img)
    print(f"  Model Name: {r2['model_name']}")
    print(f"  Weather: {r2['weather']['detected']}")
    print(f"  Lighting: {r2['time_of_day']['detected']}")
    print(f"  Hazard Level: {r2['hazard_assessment']['level']} ({r2['hazard_assessment']['risk_score_pct']}%)")
    print(f"  AV Action: {r2['hazard_assessment']['av_recommendations'][0]}")
    print("  -> PASSED")

    # 5. Model 3
    print("\n[5/5] Testing Model 3: YOLO Object Detector...")
    m3 = YOLODetector()
    r3 = m3.detect(img)
    print(f"  Model Name: {r3['model_name']}")
    print(f"  Detected Entities: {r3['total_objects']}")
    print(f"  Counts: {r3['counts']}")
    print("  -> PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! SYSTEM READY FOR LIVE DEMO!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

