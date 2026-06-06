# 🌿 Eco-Routing Engine
**Dynamic Urban Navigation via Machine Learning & Computer Vision**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)](https://flask.palletsprojects.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange.svg)](https://catboost.ai/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer_Vision-yellow.svg)](https://ultralytics.com/yolov8)

An advanced web-based routing engine designed to minimize the carbon footprint of urban commutes. Unlike traditional routing algorithms (like Google Maps) that strictly optimize for **Distance** or **Time**, this engine mathematically optimizes for the **Lowest Emission Factor**.

By integrating real-world infrastructure data, environmental APIs, CatBoost regression models, and YOLOv8 real-time computer vision, the platform proactively routes drivers away from congestion and "urban canyons" where pollutants get trapped.

---

## ✨ Key Features

1. **CatBoost Emission Prediction Model**
   - Replaces static math formulas with a highly trained ML regressor.
   - Evaluates 44 specific features per road segment (including speed limits, highway classes, and weather descriptions).
   - Non-linear modeling accurately captures how low-speed idling drastically spikes fuel consumption and emissions.

2. **Real-Time Computer Vision Re-routing (YOLOv8)**
   - Traffic is dynamic, and so is our graph.
   - Users or traffic cameras can upload images of specific road segments directly to the UI.
   - A YOLOv8 model instantly detects the vehicle mix (cars, trucks, buses). If a massive traffic jam is detected, the engine scales up the edge-weight and **dynamically re-routes** the driver to bypass the bottleneck.

3. **"Urban Canyon" Awareness**
   - Environmental GIS data is factored into every route.
   - Roads with high `building_density` and low `wind_speed` trap exhaust gases, drastically raising the localized emission factor. The algorithm actively favors open, tree-lined, breezy streets to disperse pollutants.

4. **Multi-City Scalability**
   - Ingests raw `.geojson` data. Easily scalable to any city (Delhi, Kolkata, Mumbai) by simply uploading OpenStreetMap data arrays.

5. **SUMO Integration**
   - Ready to hook into **Simulation of Urban MObility (SUMO)** for deep, microscopic traffic flow modeling.

---

## 🏗️ Detailed System Architecture & Data Flow

This project operates on a multi-stage pipeline, blending geospatial data, machine learning, and computer vision to calculate and route vehicles optimally.

### Phase 1: Real-World Data Sourcing & Graph Construction
To ensure accurate routing on Indian streets, the system first constructs a base directed graph (NetworkX) where every node is an intersection and every edge is a road segment.
* **OSMnx / Overpass API**: Extracts the physical road topology, length, speed limits, lane counts, and highway types (e.g., `primary`, `residential`).
* **OpenWeatherMap API**: Provides real-time localized weather conditions. Crucially, it fetches `wind_speed_mps` which determines how quickly pollutants disperse.
* **OpenAQ / AQICN**: Supplies baseline Air Quality Index (AQI) and pollutant concentrations (PM2.5, NO₂, etc.) across city zones.

### Phase 2: Machine Learning Emission Prediction (CatBoost)
Instead of using static physics formulas, every single edge in the road graph is passed through a pre-trained **CatBoost Regression Model**.
* **Inputs (44 Features)**: The model takes the road's topology (length, max speed), environmental context (building density, vegetation score, wind speed), temporal context (time of day cyclical encoding), and baseline traffic estimates.
* **Non-Linear Dynamics**: CatBoost accurately predicts that a vehicle dropping from 40 km/h to 10 km/h doesn't just increase emissions linearly; it causes an exponential spike due to idling and stop-and-go inefficiencies.
* **Output**: A specific `emission_factor` (in grams of CO₂) is assigned as the weight of that specific road segment.

### Phase 3: Dynamic Computer Vision Overrides (YOLOv8)
Historical and baseline traffic data cannot account for sudden accidents or unpredictable traffic jams.
1. **User Interaction**: Through the Flask dashboard, users (or traffic cameras) upload an image of a specific road segment.
2. **Object Detection**: A YOLOv8 model processes the image in real-time, detecting and counting specific classes (`car`, `bus`, `truck`, `motorcycle`).
3. **Graph Scaling**: If the detected vehicle count massively exceeds the baseline, the engine dynamically recalculates the `emission_factor` for that specific edge using the CatBoost base scaled by the new traffic density.
4. **Re-Routing**: The graph is instantly updated, and Dijkstra's algorithm is re-run, seamlessly physically wrapping the new route around the congested bottleneck to find a greener path.

### Phase 4: Multi-Objective Routing
With the fully weighted graph established, the engine runs Dijkstra's algorithm four separate times to calculate and compare different strategies:
* **Shortest Distance**: Minimizes edge `length` (ignores traffic/emissions entirely).
* **Fastest Time**: Minimizes `time` (distance / average speed).
* **Lowest Emission**: Minimizes the CatBoost `emission_factor` (avoids traffic, low speeds, and "urban canyons").
* **Balanced**: Minimizes a weighted composite score (0.5 Emission + 0.3 Time + 0.2 Distance).

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* Windows/Linux/MacOS

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/eco-routing-engine.git
cd eco-routing-engine
```

### 2. Install Dependencies
Install all required Python packages (including CatBoost, Flask, GeoPandas, OSMnx, etc.):
```bash
pip install -r requirements.txt
```

### 3. Run the Server
Launch the Flask backend engine:
```bash
python serve.py
```

### 4. Access the Dashboard
Open your web browser and navigate to:
```text
http://localhost:5000
```

---

## 🗺️ How to Use the Dashboard

1. **Select Your City**: Choose Kolkata, Delhi, or Mumbai from the dropdown panel.
2. **Set Origin & Destination**: Click the "Click map" buttons, then click directly on the Leaflet map to drop your start and end pins.
3. **Adjust the Time Slider**: Move the slider between 0 (Midnight) and 23 (11 PM). The engine will dynamically adjust baseline traffic counts (e.g., heavily penalizing routes at 10:00 AM rush hour).
4. **Find Base Routes**: Click the **Find Base Routes** button. The engine will calculate and visualize 4 competing strategies:
   - *Shortest Distance* (Blue)
   - *Fastest Time* (Orange)
   - *Lowest Emission* (Green)
   - *Balanced* (Purple)
5. **Apply Camera Overrides**: Scroll down the sidebar to view the top 30 highest-emission bottlenecks on your route. Upload an image of a traffic jam to any segment, and click **Apply Data & Re-Route** to watch the Eco-Route dynamically wrap around the congestion!

---

## 🛠️ Technology Stack
* **Backend Framework**: Python, Flask
* **Machine Learning**: CatBoost Regressor, scikit-learn
* **Computer Vision**: Ultralytics YOLOv8
* **Geospatial / Graphs**: GeoPandas, NetworkX, Shapely
* **Frontend UI**: Vanilla JS, Leaflet.js, HTML5/CSS3

---

## 📄 License
This project is licensed under the MIT License.
