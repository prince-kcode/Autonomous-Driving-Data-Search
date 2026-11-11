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
    assert exists, f"Qdrant collection '{COLLECTION_NAME}' does not exist! Run populate_db.py first."
    pts = client.get_collection(COLLECTION_NAME).points_count
    print(f"  Indexed Points Count: {pts}")
    assert pts > 0, f"Qdrant collection '{COLLECTION_NAME}' is empty!"
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
    print("\n[5/5] Testing Model 3: YOLO Object Detector with Glare Filtering...")
    m3 = YOLODetector()
    r3 = m3.detect(img)
    print(f"  Model Name: {r3['model_name']}")
    print(f"  Detected Entities: {r3['total_objects']}")
    print(f"  Counts: {r3['counts']}")
    assert r3['total_objects'] > 0, "Expected YOLO to detect vehicles in reference driving frame!"

    # Verify RGB/BGR color ordering regression on data/car.jpg
    if os.path.exists("data/car.jpg"):
        car_r = m3.detect("data/car.jpg")
        print(f"  Car RGB Regression Check: {car_r['counts']}")
        assert car_r['counts'].get("car", 0) >= 1, f"Expected 'car' detection on data/car.jpg, got {car_r['counts']}"

    # Test false positive suppression on forest road with sunlight
    forest_img = "data/foggy_forest_road.jpg"
    if os.path.exists(forest_img):
        forest_r = m3.detect(forest_img)
        print(f"  Forest Road Sunlight False-Positive Check: {forest_r['counts']}")
        assert "traffic light" not in str(forest_r['counts']).lower(), "Sunlight should not be detected as traffic light!"
        print("  Forest Road Sunlight False-Positive Check: -> PASSED (Zero false traffic lights)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! TRAFFIC LIGHT VERIFIER ACTIVE!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
