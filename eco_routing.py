from pathlib import Path
import argparse
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

import geopandas as gpd
import networkx as nx
import osmnx as ox
from heapq import heappop, heappush

def load_edges_dataset() -> gpd.GeoDataFrame:
    for candidate in [
        Path("fused_roads_with_sumo_with_emissions.geojson"),
        Path("fused_roads_with_emissions.geojson"),
        Path("fused_roads_with_sumo.geojson"),
        Path("fused_roads.geojson"),
    ]:
        if candidate.exists():
            return gpd.read_file(candidate)

    raise FileNotFoundError(
        "No fused road dataset found. Expected fused_roads_with_sumo.geojson, "
        "fused_roads_with_emissions.geojson, or fused_roads.geojson."
    )


_EDGES_CACHE: Optional[gpd.GeoDataFrame] = None


def get_edges_dataset() -> gpd.GeoDataFrame:
    """Lazy-load and cache the fused dataset so imports don't fail."""
    global _EDGES_CACHE
    if _EDGES_CACHE is None:
        _EDGES_CACHE = load_edges_dataset()
    return _EDGES_CACHE

def build_carbon_graph():
    """Create network graph with carbon cost as edge weight"""
    edges = get_edges_dataset()
    G = nx.MultiDiGraph()
    
    for idx, row in edges.iterrows():
        u = row['u']  # start node
        v = row['v']  # end node

        emission_cost = row.get('emission_cost_g')
        carbon_cost = row.get('carbon_cost_with_emissions', row['carbon_cost'])
        avg_speed_kmph = row.get('avg_speed', row.get('avg_speed_kmph', row.get('avg_speed_kmh')))
        
        # Edge attributes
        edge_data = {
            'length': row['length'],
            'carbon_cost': carbon_cost,
            'emission_cost_g': emission_cost,
            'vehicle_count': row['vehicle_count'],
            'avg_speed': avg_speed_kmph,
            'building_density': row['building_density'],
            'vegetation_score': row['vegetation_score'],
            'time': row['length'] / (avg_speed_kmph * 1000 / 3600),  # time in seconds
            'geometry': row.get('geometry')
        }
        
        G.add_edge(u, v, **edge_data)
    
    return G

def route_shortest_distance(G, origin, destination):
    """Find shortest distance route"""
    route = nx.shortest_path(G, origin, destination, weight='length')
    return route

def route_fastest_time(G, origin, destination):
    """Find fastest time route"""
    route = nx.shortest_path(G, origin, destination, weight='time')
    return route

def route_lowest_carbon(G, origin, destination):
    """Find route with lowest carbon footprint"""
    route = nx.shortest_path(G, origin, destination, weight='carbon_cost')
    return route


def route_lowest_emissions(G, origin, destination):
    """Find route with lowest emissions (vehicle-aware)"""
    route = nx.shortest_path(G, origin, destination, weight='carbon_cost')
    return route

def route_balanced(G, origin, destination, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Multi-objective routing
    alpha: weight for carbon cost
    beta: weight for time
    gamma: weight for distance
    """
    # Create composite weight
    for u, v, data in G.edges(data=True):
        # Normalize values (assume max values)
        norm_carbon = data['carbon_cost'] / 5000  # adjust based on your data
        norm_time = data['time'] / 300  # adjust based on your data
        norm_distance = data['length'] / 1000  # adjust based on your data
        
        # Composite score
        data['composite_weight'] = (
            alpha * norm_carbon + 
            beta * norm_time + 
            gamma * norm_distance
        )
    
    route = nx.shortest_path(G, origin, destination, weight='composite_weight')
    return route

def calculate_route_stats(G, route):
    """Calculate statistics for a given route"""
    total_distance = 0
    total_time = 0
    total_carbon = 0
    total_emissions = 0
    total_vehicles = 0
    avg_speed_list = []
    
    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        
        # Get edge with minimum carbon cost if multiple edges exist
        edge_data = min(G[u][v].values(), key=lambda x: x['carbon_cost'])
        
        total_distance += edge_data['length']
        total_time += edge_data['time']
        total_carbon += edge_data['carbon_cost']
        if edge_data.get('emission_cost_g') is not None:
            total_emissions += edge_data['emission_cost_g']
        total_vehicles += edge_data.get('vehicle_count', 0)
        avg_speed_list.append(edge_data['avg_speed'])
    
    return {
        'distance_km': total_distance / 1000,
        'time_minutes': total_time / 60,
        'carbon_footprint': total_carbon,
        'emission_cost_g': total_emissions,
        'avg_vehicle_density': total_vehicles / len(route) if route else 0,
        'avg_speed_kmh': sum(avg_speed_list) / len(avg_speed_list) if avg_speed_list else 0,
        'num_segments': len(route) - 1
    }

def _route_edges_with_geometry(G: nx.MultiDiGraph, route: List[int]) -> List:
    geometries = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        edge_data = min(G[u][v].values(), key=lambda x: x['carbon_cost'])
        geom = edge_data.get('geometry')
        if geom is not None:
            geometries.append(geom)
    return geometries


def _plot_route_comparison(
    routes: Dict[str, List[int]],
    output_path: str,
    show_plot: bool
):
    fig, ax = plt.subplots(figsize=(10, 10))
    try:
        edges = get_edges_dataset()
        edges.plot(ax=ax, color="lightgray", linewidth=0.5, alpha=0.6)
    except Exception:
        pass

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (label, route) in enumerate(routes.items()):
        geoms = _route_edges_with_geometry(build_carbon_graph(), route)
        if not geoms:
            continue
        gdf = gpd.GeoSeries(geoms)
        gdf.plot(ax=ax, linewidth=2.5, color=colors[idx % len(colors)], label=label)

    ax.set_title("Eco Routing Comparison")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def compare_routes_data(origin: int, destination: int) -> Dict[str, object]:
    """Return route comparison results without printing."""
    G = build_carbon_graph()
    strategies = [
        ("Shortest Distance", route_shortest_distance),
        ("Fastest Time", route_fastest_time),
        ("Lowest Carbon", route_lowest_carbon),
        ("Lowest Emissions (Vehicle-Aware)", route_lowest_emissions),
        ("Balanced (50% carbon, 30% time, 20% distance)",
         lambda g, o, d: route_balanced(g, o, d, 0.5, 0.3, 0.2))
    ]

    results = []
    routes_for_plot: Dict[str, List[int]] = {}

    for strategy_name, route_func in strategies:
        try:
            route = route_func(G, origin, destination)
            stats = calculate_route_stats(G, route)
            results.append({
                'strategy': strategy_name,
                **stats
            })
            routes_for_plot[strategy_name] = route
        except nx.NetworkXNoPath:
            continue

    best_eco = min(results, key=lambda x: x['carbon_footprint']) if results else None

    return {
        "origin": origin,
        "destination": destination,
        "results": results,
        "best_eco": best_eco,
        "routes_for_plot": routes_for_plot,
    }


def compare_routes(origin, destination, plot_path: Optional[str] = None, show_plot: bool = False):
    """Compare different routing strategies"""
    payload = compare_routes_data(origin, destination)

    print(f"\n{'='*60}")
    print(f"Route Comparison: Origin {origin} → Destination {destination}")
    print(f"{'='*60}\n")

    for item in payload["results"]:
        print(f"📍 {item['strategy']}")
        print(f"   Distance: {item['distance_km']:.2f} km")
        print(f"   Time: {item['time_minutes']:.2f} minutes")
        print(f"   Carbon Footprint: {item['carbon_footprint']:.2f}")
        if item['emission_cost_g'] > 0:
            print(f"   Emission Cost: {item['emission_cost_g']:.2f} g")
        print(f"   Avg Speed: {item['avg_speed_kmh']:.2f} km/h")
        print(f"   Segments: {item['num_segments']}")
        print()

    if payload["best_eco"] is not None:
        best_eco = payload["best_eco"]
        print(f"🌱 RECOMMENDED ECO-ROUTE: {best_eco['strategy']}")
        print(
            f"   Saves {payload['results'][0]['carbon_footprint'] - best_eco['carbon_footprint']:.2f} carbon units"
        )

    if plot_path:
        _plot_route_comparison(payload["routes_for_plot"], plot_path, show_plot)
        print(f"\nSaved route comparison plot: {plot_path}")

    return payload["results"]

if __name__ == "__main__":
    print("Loading road network...")

    parser = argparse.ArgumentParser(description="Compare eco-routing strategies")
    parser.add_argument("--origin", type=int, help="Origin node ID")
    parser.add_argument("--dest", type=int, help="Destination node ID")
    parser.add_argument("--plot", type=str, help="Save route comparison plot (PNG path)")
    parser.add_argument("--show-plot", action="store_true", help="Show plot window")
    args = parser.parse_args()

    if args.origin is not None and args.dest is not None:
        compare_routes(args.origin, args.dest, plot_path=args.plot, show_plot=args.show_plot)
        sys.exit(0)

    G = build_carbon_graph()
    nodes = list(G.nodes())[:10]

    print(f"\nSample nodes available: {nodes[:5]}...")
    print("\nTo use this script:")
    print("1. Find origin and destination node IDs from your fused_roads.geojson")
    print("2. Call: compare_routes(origin_node_id, destination_node_id)")
    print("\nExample:")
    print(f"results = compare_routes({nodes[0]}, {nodes[4]})")
    print("\nCLI example:")
    print(f"python eco_routing.py --origin {nodes[0]} --dest {nodes[4]}")
