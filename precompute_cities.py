import json
import sys
from pathlib import Path

# Add project root to path to ensure imports work
BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

try:
    from eco_route_engine import download_and_fuse_bbox, build_emission_graph
except ImportError as e:
    print(f"[ERROR] Could not import eco_route_engine: {e}")
    sys.exit(1)

def main():
    saved_points_file = BASE_DIR / "saved_points.json"
    if not saved_points_file.exists():
        print(f"[ERROR] saved_points.json not found at {saved_points_file}")
        sys.exit(1)

    with open(saved_points_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # We precompute Delhi, Noida, Mumbai, and Bangalore.
    # Kolkata is already precomputed.
    target_cities = ["delhi", "noida", "mumbai", "bangalore"]

    print("=" * 60)
    print("  Carbon-Efficient Routing Map Precomputation Pipeline")
    print("=" * 60)

    for city in target_cities:
        if city not in config:
            print(f"[WARN] City '{city}' not found in saved_points.json. Skipping.")
            continue

        city_data = config[city]
        city_name = city_data["city_name"]
        center = city_data["center"]
        buf = city_data["buffer_degrees"]
        
        print(f"\n>>> Processing city: {city_name} ({city_data['state_name']})")
        print(f"    Center: Lat {center['lat']}, Lon {center['lon']}")
        print(f"    Buffer size: {buf}° (Approx. {buf*111:.1f} km)")

        # Calculate olat, olon, dlat, dlon such that when download_and_fuse_bbox
        # applies its 0.015 buffer, the final bounding box size is exactly center +/- buf.
        # lat_min = min(olat, dlat) - 0.015 = center - buf  => min(olat, dlat) = center - buf + 0.015
        # lat_max = max(olat, dlat) + 0.015 = center + buf  => max(olat, dlat) = center + buf - 0.015
        offset = buf - 0.015
        if offset < 0:
            print(f"[ERROR] Buffer size {buf} is too small for download_and_fuse_bbox offset calculation.")
            continue

        olat = center["lat"] - offset
        dlat = center["lat"] + offset
        olon = center["lon"] - offset
        dlon = center["lon"] + offset

        city_dir = BASE_DIR / city
        print(f"    Target directory: {city_dir}")

        # Check if already precomputed
        gpickle_path = city_dir / "fused_roads.gpickle"
        geojson_path = city_dir / "fused_roads.geojson"
        
        if gpickle_path.exists() and geojson_path.exists():
            print(f"    [INFO] Precomputed files already exist for {city_name}. Skipping download.")
            continue

        try:
            print(f"    [1/2] Downloading & fusing map features (OSM network, buildings, vegetation)...")
            download_and_fuse_bbox(olat, olon, dlat, dlon, city_name=city)
            
            print(f"    [2/2] Rebuilding and compiling NetworkX graph to binary (.gpickle)...")
            G = build_emission_graph(city=city, use_ml=True, use_csv_emission=False)
            
            if G and G.number_of_nodes() > 0:
                print(f"    [SUCCESS] Successfully precomputed {city_name}!")
                print(f"              Nodes: {G.number_of_nodes():,}, Edges: {G.number_of_edges():,}")
            else:
                print(f"    [ERROR] Compiled graph for {city_name} is empty.")
                
        except Exception as e:
            print(f"    [ERROR] Failed to precompute {city_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  Precomputation pipeline execution completed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
