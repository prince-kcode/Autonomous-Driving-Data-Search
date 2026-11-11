import os
import urllib.request

ADDITIONAL_IMAGES = [
    {
        "filename": "motorcycle_highway_lane.jpg",
        "url": "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=800&q=80",
        "desc": "Sport motorcycle parked beside urban wall"
    },
    {
        "filename": "school_bus_suburban.jpg",
        "url": "https://images.unsplash.com/photo-1570125909232-eb263c188f7e?w=800&q=80",
        "desc": "Large passenger coach bus on open asphalt"
    },
    {
        "filename": "red_sports_car_road.jpg",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800&q=80",
        "desc": "Sports sedan cruising at speed on multi-lane highway"
    },
    {
        "filename": "modern_electric_vehicle.jpg",
        "url": "https://images.unsplash.com/photo-1563720223185-11003d516935?w=800&q=80",
        "desc": "Modern black SUV parked on urban driveway"
    },
    {
        "filename": "foggy_forest_road.jpg",
        "url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80",
        "desc": "Sunlight glare filtering through misty pine forest foliage (negative traffic-light check)"
    }
]

def download_additional():
    os.makedirs("data", exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    count = 0
    for item in ADDITIONAL_IMAGES:
        dest_path = os.path.join("data", item["filename"])
        if os.path.exists(dest_path):
            print(f"Skipping {item['filename']} (exists)")
            continue
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, 'wb') as f:
                f.write(resp.read())
            print(f"Downloaded: {item['filename']} - {item['desc']}")
            count += 1
        except Exception as e:
            print(f"Failed {item['filename']}: {e}")
    print(f"Successfully downloaded {count} new driving scenario images!")

if __name__ == "__main__":
    download_additional()
