import os
import urllib.request

ADDITIONAL_IMAGES = [
    {
        "filename": "motorcycle_highway_lane.jpg",
        "url": "https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=800&q=80",
        "desc": "Motorcycle driving on an open road"
    },
    {
        "filename": "school_bus_suburban.jpg",
        "url": "https://images.unsplash.com/photo-1570125909232-eb263c188f7e?w=800&q=80",
        "desc": "Large bus traveling on a roadway"
    },
    {
        "filename": "construction_road_barrier.jpg",
        "url": "https://images.unsplash.com/photo-1541888946425-d0fbb186c5f7?w=800&q=80",
        "desc": "Road construction zone with obstacles and barricades"
    },
    {
        "filename": "pedestrian_night_rain.jpg",
        "url": "https://images.unsplash.com/photo-1514565131-fce0801e5785?w=800&q=80",
        "desc": "Pedestrian in rain at night near roadway"
    },
    {
        "filename": "tunnel_driving_lighting.jpg",
        "url": "https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=800&q=80",
        "desc": "Vehicle driving through an illuminated underground highway tunnel"
    },
    {
        "filename": "sunset_coastal_highway.jpg",
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80",
        "desc": "Golden hour sunset lighting on an open scenic highway"
    },
    {
        "filename": "heavy_traffic_jam_city.jpg",
        "url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=800&q=80",
        "desc": "Dense urban traffic congestion with vehicles"
    },
    {
        "filename": "red_sports_car_road.jpg",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=800&q=80",
        "desc": "Red sports car cruising on asphalt road"
    },
    {
        "filename": "modern_electric_vehicle.jpg",
        "url": "https://images.unsplash.com/photo-1563720223185-11003d516935?w=800&q=80",
        "desc": "Modern smart electric car driving autonomously"
    },
    {
        "filename": "stop_sign_crossroad.jpg",
        "url": "https://images.unsplash.com/photo-1584438784894-089d6a62b8fa?w=800&q=80",
        "desc": "Stop sign posted at an urban street junction"
    },
    {
        "filename": "foggy_forest_road.jpg",
        "url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800&q=80",
        "desc": "Winding road through foggy misty mountain forest"
    },
    {
        "filename": "night_city_junction.jpg",
        "url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=800&q=80",
        "desc": "Illuminated nighttime street junction with vehicle lights"
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
