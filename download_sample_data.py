import os
import urllib.request

DATASET_IMAGES = [
    {
        "filename": "highway_sunny_traffic.jpg",
        "url": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?w=800&q=80",
        "desc": "Sports sedan cruising on a palm-lined daytime street"
    },
    {
        "filename": "rainy_night_driving.jpg",
        "url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800&q=80",
        "desc": "Night driving on wet city street with cars, bus, and crosswalk"
    },
    {
        "filename": "cyclist_urban_road.jpg",
        "url": "https://images.unsplash.com/photo-1517649763962-0c623266ddc0?w=800&q=80",
        "desc": "Bicycle parked along an urban concrete backdrop"
    },
    {
        "filename": "foggy_rural_highway.jpg",
        "url": "https://images.unsplash.com/photo-1487621167305-5d248087c724?w=800&q=80",
        "desc": "Dense fog and misty forest low-visibility atmosphere"
    },
    {
        "filename": "city_downtown_traffic.jpg",
        "url": "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800&q=80",
        "desc": "Driver POV behind steering wheel in downtown city traffic"
    },
    {
        "filename": "night_highway_speed.jpg",
        "url": "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=800&q=80",
        "desc": "Overhead view of multi-lane roadway and parking lot vehicles"
    },
    {
        "filename": "truck_freeway_transport.jpg",
        "url": "https://images.unsplash.com/photo-1601584115197-04ecc0da31d7?w=800&q=80",
        "desc": "Heavy cargo truck driving along a rural two-lane highway"
    }
]

def download_data():
    os.makedirs("data", exist_ok=True)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for item in DATASET_IMAGES:
        dest_path = os.path.join("data", item["filename"])
        if os.path.exists(dest_path):
            print(f"Skipping {item['filename']} (already exists)")
            continue
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, 'wb') as out_file:
                out_file.write(resp.read())
            print(f"Downloaded: {item['filename']} - {item['desc']}")
        except Exception as e:
            print(f"Failed to download {item['filename']}: {e}")

if __name__ == "__main__":
    download_data()

