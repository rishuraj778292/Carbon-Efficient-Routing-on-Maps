import osmnx as ox
import time
import sys

print("Testing OSMnx downloads for Noida...")
sys.stdout.flush()

lat_center, lon_center = 28.5600, 77.3450
buf = 0.04 # let's test a smaller buffer first (approx 4.4km radius, i.e., 9km x 9km)

lat_min = lat_center - buf
lat_max = lat_center + buf
lon_min = lon_center - buf
lon_max = lon_center + buf

print(f"Bbox: N={lat_max}, S={lat_min}, E={lon_max}, W={lon_min}")
sys.stdout.flush()

t0 = time.time()
try:
    print("[1] Downloading road network...")
    sys.stdout.flush()
    try:
        G = ox.graph_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), network_type="drive")
    except TypeError:
        G = ox.graph_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, network_type="drive")
    print(f"Road network downloaded successfully in {time.time() - t0:.2f} seconds.")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed to download road network: {e}")
    sys.stdout.flush()

t1 = time.time()
try:
    print("[2] Downloading building footprints...")
    sys.stdout.flush()
    tags_b = {"building": True}
    try:
        buildings = ox.features_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), tags=tags_b)
    except TypeError:
        buildings = ox.features_from_bbox(north=lat_max, south=lat_min, east=lon_max, west=lon_min, tags=tags_b)
    print(f"Buildings downloaded successfully in {time.time() - t1:.2f} seconds.")
    print(f"Buildings count: {len(buildings)}")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed to download buildings: {e}")
    sys.stdout.flush()

print("Finished diagnostic test.")
