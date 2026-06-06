"""
serve.py
---------
Flask server that:
  - Serves index.html (the frontend)
  - POST /api/route      -> runs eco_route_engine and returns JSON + plot
  - POST /api/detect     -> YOLO vehicle detection on uploaded road image
  - GET  /api/plot       -> streams the saved route_compare PNG
  - GET  /api/default_points -> returns demo origin/destination

YOLO integration:
  - Upload a road image via /api/detect
  - Returns vehicle counts + emission estimate per km
  - The route endpoint accepts optional vehicle_override from YOLO results
  - If no image is uploaded, fallback random values are used automatically
"""

import base64
import io
import json
import os
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    import _io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Graceful Flask import ─────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not found.  Run:  pip install flask flask-cors")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

# ── Server-side state ───────────────────────────────────────────────────────────────
# SUMO tasks: { task_id: {"done": bool, "status": str, "output": [lines], "plot_b64": str|None} }
_sumo_tasks: dict = {}
# Segment detection cache (optional): { segment_id: detection_result }
_segment_cache: dict = {}

# ── Default points per city ────────────────────────────────────────────────────
DEFAULT_POINTS = {
    "kolkata": {
        "origin": {"lat": 22.5680, "lon": 88.3512, "label": "Esplanade / BBD Bagh"},
        "destination": {"lat": 22.5756, "lon": 88.3697, "label": "Sealdah Station"}
    },
    "delhi": {
        "origin": {"lat": 28.6139, "lon": 77.2090, "label": "Connaught Place"},
        "destination": {"lat": 28.5921, "lon": 77.2250, "label": "India Gate"}
    },
    "mumbai": {
        "origin": {"lat": 18.9220, "lon": 72.8347, "label": "Gateway of India"},
        "destination": {"lat": 18.9696, "lon": 72.8223, "label": "Marine Drive"}
    },
    "noida": {
        "origin": {"lat": 28.5355, "lon": 77.3910, "label": "Sector 18"},
        "destination": {"lat": 28.5562, "lon": 77.3601, "label": "Okhla Bird Sanctuary"}
    },
    "bangalore": {
        "origin": {"lat": 12.9716, "lon": 77.5946, "label": "Vidhana Soudha"},
        "destination": {"lat": 12.9352, "lon": 77.6245, "label": "Koramangala"}
    }
}

# ── YOLO model singleton (loaded once) ────────────────────────────────────────
_yolo_model = None

def _get_yolo_model():
    """Lazy-load the YOLO model."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        # Look for local model first
        model_paths = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt"]
        for mp in model_paths:
            if Path(mp).exists():
                _yolo_model = YOLO(str(mp))
                print(f"[YOLO] Loaded model: {mp}")
                return _yolo_model
            full = BASE_DIR / mp
            if full.exists():
                _yolo_model = YOLO(str(full))
                print(f"[YOLO] Loaded model: {full}")
                return _yolo_model
        # Try default download
        _yolo_model = YOLO("yolov8n.pt")
        print("[YOLO] Downloaded and loaded yolov8n.pt")
        return _yolo_model
    except Exception as e:
        print(f"[YOLO] Could not load model: {e}")
        return None

# COCO vehicle class IDs
VEHICLE_COCO_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Emissions per vehicle type (g CO2 per km)
EMISSIONS_G_PER_KM = {
    "car":        120,
    "motorcycle":  72,
    "bus":        822,
    "truck":      900,
    "van":        200,
    "bicycle":      0,
}


def _run_yolo_detection(img_bytes: bytes) -> dict:
    """
    Core detection logic shared by /api/detect and /api/detect_segment.
    Returns a result dict (same schema as api_detect).
    """
    if len(img_bytes) < 100:
        return _fallback_vehicle_response("Image too small", as_dict=True)

    model = _get_yolo_model()
    if model is None:
        return _fallback_vehicle_response("YOLO model not available", as_dict=True)

    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return _fallback_vehicle_response("Could not decode image", as_dict=True)

        results = model(frame, verbose=False, conf=0.35)

        vehicle_counts = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_COCO_IDS:
                label = VEHICLE_COCO_IDS[cls_id]
                vehicle_counts[label] = vehicle_counts.get(label, 0) + 1

        total = sum(vehicle_counts.values())
        co2_per_km = sum(
            cnt * EMISSIONS_G_PER_KM.get(vtype, 120)
            for vtype, cnt in vehicle_counts.items()
        )

        annotated = results[0].plot()
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buf).decode()

        return {
            "vehicle_counts": vehicle_counts,
            "total_vehicles": total,
            "co2_per_km_g":   co2_per_km,
            "annotated_b64":  annotated_b64,
            "source":         "yolo_detection",
            "message":        f"Detected {total} vehicles via YOLOv8",
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return _fallback_vehicle_response(f"Detection error: {e}", as_dict=True)


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Accept a road image upload, run YOLO vehicle detection.
    Returns vehicle counts + emission estimate per km.
    If no image or YOLO fails, returns fallback random values.
    """
    try:
        if "image" not in request.files:
            return _fallback_vehicle_response("No image uploaded")
        file = request.files["image"]
        if file.filename == "":
            return _fallback_vehicle_response("Empty filename")
        img_bytes = file.read()
        result = _run_yolo_detection(img_bytes)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return _fallback_vehicle_response(f"Detection error: {str(e)}")


@app.route("/api/detect_segment", methods=["POST"])
def api_detect_segment():
    """
    Same as /api/detect but for a named road segment.
    Accepts:
      image       : image file (multipart/form-data)
      segment_id  : "u_v" edge identifier string
      segment_name: optional human-readable road name
    Returns same schema as /api/detect + segment_id field.
    """
    segment_id   = request.form.get("segment_id", "unknown")
    segment_name = request.form.get("segment_name", "")
    try:
        if "image" not in request.files:
            result = _fallback_vehicle_response("No image", as_dict=True)
        else:
            file = request.files["image"]
            img_bytes = file.read()
            result = _run_yolo_detection(img_bytes)

        result["segment_id"]   = segment_id
        result["segment_name"] = segment_name

        # Cache the result so the frontend can retrieve it later if needed
        _segment_cache[segment_id] = result

        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        fb = _fallback_vehicle_response(f"Error: {e}", as_dict=True)
        fb["segment_id"] = segment_id
        return jsonify(fb)


def _fallback_vehicle_response(reason: str, as_dict: bool = False):
    """Generate random realistic vehicle counts as fallback."""
    import random
    counts = {
        "car":        random.randint(20, 80),
        "motorcycle": random.randint(5, 30),
        "bus":        random.randint(1, 8),
        "truck":      random.randint(2, 12),
    }
    total = sum(counts.values())
    co2 = sum(cnt * EMISSIONS_G_PER_KM.get(v, 120) for v, cnt in counts.items())
    data = {
        "vehicle_counts": counts,
        "total_vehicles": total,
        "co2_per_km_g":   co2,
        "annotated_b64":  None,
        "source":         "random_fallback",
        "message":        f"Using random values ({reason}). "
                          f"Upload a road image for real YOLO detection.",
    }
    return data if as_dict else jsonify(data)


@app.route("/api/route", methods=["POST"])
def api_route():
    """
    Expects JSON body:
    {
      "city":       str     (optional, default "kolkata"),
      "origin_lat": float,  "origin_lon": float,
      "dest_lat":   float,  "dest_lon":   float,
      "hour":       int     (optional, default 8),
      "vehicle_override":  int  (optional, legacy global scale),
      "route_overrides":   dict (optional, per-route YOLO upload),
      "segment_overrides": dict (optional, per-segment upload, {"u_v": vehicle_count})
    }
    Returns JSON with stats, route coordinates, per-route segments, and base64 plot.
    """
    try:
        body = request.get_json(force=True) or {}
        city = body.get("city", "kolkata").lower()
        defaults = DEFAULT_POINTS.get(city, DEFAULT_POINTS["kolkata"])
        
        olat = float(body.get("origin_lat", defaults["origin"]["lat"]))
        olon = float(body.get("origin_lon", defaults["origin"]["lon"]))
        dlat = float(body.get("dest_lat",   defaults["destination"]["lat"]))
        dlon = float(body.get("dest_lon",   defaults["destination"]["lon"]))
        hour = int(body.get("hour", 8))
        vehicle_override  = body.get("vehicle_override",  None)
        route_overrides   = body.get("route_overrides",   None)
        segment_overrides = body.get("segment_overrides", None)

        # Convert segment_overrides values to int (JSON may send floats)
        if segment_overrides:
            segment_overrides = {k: int(v) for k, v in segment_overrides.items()}

        from eco_route_engine import run_eco_routing
        plot_path = str(BASE_DIR / f"route_compare_{city}.png")
        result = run_eco_routing(
            olat, olon, dlat, dlon,
            hour=hour,
            city=city,
            plot_path=plot_path,
            vehicle_override=vehicle_override,
            route_overrides=route_overrides,
            segment_overrides=segment_overrides,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # Embed plot as base64
        if Path(plot_path).exists():
            with open(plot_path, "rb") as f:
                result["plot_b64"] = base64.b64encode(f.read()).decode()

        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/plot")
def api_plot():
    city = request.args.get("city", "kolkata").lower()
    plot_path = str(BASE_DIR / f"route_compare_{city}.png")
    if not Path(plot_path).exists():
        return jsonify({"error": "No plot yet. Call /api/route first."}), 404
    return send_file(plot_path, mimetype="image/png")


@app.route("/api/default_points")
def api_default_points():
    """Return the default demo origin/destination for a given city."""
    city = request.args.get("city", "kolkata").lower()
    defaults = DEFAULT_POINTS.get(city, DEFAULT_POINTS["kolkata"])
    return jsonify({
        "origin": defaults["origin"],
        "destination": defaults["destination"],
        "city": city.title(),
    })


@app.route("/api/upload_city", methods=["POST"])
def api_upload_city():
    """
    Accepts a zip file with city data (fused_roads.geojson, etc.)
    Extracts it to a directory named after the city.
    """
    try:
        import zipfile
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        city_name = request.form.get("city_name", "").strip().lower()
        if not city_name:
            return jsonify({"error": "No city name provided"}), 400
            
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
            
        city_dir = BASE_DIR / city_name
        city_dir.mkdir(parents=True, exist_ok=True)
        
        # Save zip to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            
        # Extract
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            # security: prevent zip slip
            for member in zip_ref.namelist():
                if member.startswith("/") or ".." in member:
                    continue
                zip_ref.extract(member, str(city_dir))
            
        os.unlink(tmp_path)
        
        return jsonify({"success": True, "message": f"Successfully uploaded and extracted data for {city_name.title()}."})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── SUMO endpoints ────────────────────────────────────────────────────────────────────
@app.route("/api/sumo", methods=["POST"])
def api_sumo():
    """
    Start a SUMO eco-routing simulation in a background thread.
    Accepts JSON: { city, steps, vehicles, hour, skip_sumo }
    Returns: { task_id }
    """
    body       = request.get_json(force=True) or {}
    city       = body.get("city",      "kolkata")
    steps      = int(body.get("steps",     300))
    vehicles   = int(body.get("vehicles",  200))
    hour       = int(body.get("hour",        8))
    skip_sumo  = bool(body.get("skip_sumo", True))

    task_id = str(uuid.uuid4())[:8]
    _sumo_tasks[task_id] = {
        "done":    False,
        "status":  "running",
        "output":  [],
        "plot_b64": None,
    }

    def _run():
        try:
            cmd = [
                sys.executable, str(BASE_DIR / "run_sumo.py"),
                "--city",     city,
                "--steps",    str(steps),
                "--vehicles", str(vehicles),
                "--hour",     str(hour),
            ]
            if skip_sumo:
                cmd.append("--skip-sumo")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(BASE_DIR),
            )
            for line in proc.stdout:
                stripped = line.rstrip()
                if stripped:
                    _sumo_tasks[task_id]["output"].append(stripped)
            proc.wait()

            # Try to embed the plot image
            city_key = city.lower().replace(" ", "_")
            plot_file = BASE_DIR / f"route_compare_{city_key}.png"
            if plot_file.exists():
                with open(plot_file, "rb") as fh:
                    _sumo_tasks[task_id]["plot_b64"] = base64.b64encode(fh.read()).decode()

            _sumo_tasks[task_id]["status"] = "done" if proc.returncode == 0 else "error"
        except Exception as exc:
            _sumo_tasks[task_id]["output"].append(f"[ERROR] {exc}")
            _sumo_tasks[task_id]["status"] = "error"
        finally:
            _sumo_tasks[task_id]["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({"task_id": task_id})


@app.route("/api/sumo_status/<task_id>")
def api_sumo_status(task_id):
    """Poll SUMO task status. Returns output lines + done flag + optional plot_b64."""
    task = _sumo_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({
        "done":     task["done"],
        "status":   task["status"],
        "output":   task["output"],
        "plot_b64": task.get("plot_b64"),
    })


# subprocess import needed for SUMO
import subprocess


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*55}")
    print(f"  Eco-Routing Server starting on http://localhost:{port}")
    print(f"  Open your browser at: http://localhost:{port}")
    print(f"  YOLO vehicle detection: POST /api/detect")
    print(f"  Segment detection:      POST /api/detect_segment")
    print(f"  City Upload:            POST /api/upload_city")
    print(f"  SUMO simulation:        POST /api/sumo")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
