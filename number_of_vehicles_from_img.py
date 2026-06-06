"""
Download options (do once when online):
        pip install ultralytics
        python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"   # downloads ~6 MB
    Then copy the cached .pt file here.
Usage:
    python number_of_vehicles_from_img.py                        # uses default image + default model
    python number_of_vehicles_from_img.py --source 0             # webcam (real-time)
    python number_of_vehicles_from_img.py --source road.jpg      # custom image
    python number_of_vehicles_from_img.py --model yolov8n.pt     # custom model

# Run on an image (default)
python number_of_vehicles_from_img.py --source derek-lee-C031beKcwdQ-unsplash.jpg

# Run real-time on webcam
python number_of_vehicles_from_img.py --source 0

# Run on a video file
python number_of_vehicles_from_img.py --source traffic.mp4

# Adjust confidence threshold
python number_of_vehicles_from_img.py --source road.jpg --conf 0.5


"""

import sys
import os
import argparse
from pathlib import Path
import cv2

# Source: IPCC / European Environment Agency typical values
EMISSIONS_G_PER_KM = {
    "car":          120,   # average petrol/diesel passenger car
    "motorcycle":    72,   # motorbike / two-wheeler
    "bus":          822,   # city bus (all passengers, ~50 g/km/passenger × 16)
    "truck":        900,   # heavy goods vehicle (HGV)
    "van":          200,   # light commercial van
    "bicycle":        0,   # no tailpipe emissions
    "person":         0,   # pedestrian
}

VEHICLE_CLASS_MAP = {
    "car":         "car",
    "motorbike":   "motorcycle",
    "motorcycle":  "motorcycle",
    "bus":         "bus",
    "truck":       "truck",
    "van":         "van",
    "bicycle":     "bicycle",
    "auto":        "car",
    "minivan":     "van",
    "pickup truck":"truck",
    "lorry":       "truck",
    "heavy truck": "truck",
    "suv":         "car",
    "sedan":       "car",
    "hatchback":   "car",
    "jeep":        "car",
}

VEHICLE_COCO_IDS = {2, 3, 5, 7}  # car=2, motorcycle=3, bus=5, truck=7


def find_local_model(preferred: str) -> str:
    """Return the path to a .pt model that exists locally."""
    candidates = [
        preferred,
        "yolov8n.pt",
        "yolov8s.pt",
        "yolo11n.pt",
        "yolo11s.pt",
    ]
    hub_cache = Path.home() / "AppData" / "Roaming" / "Ultralytics"
    for name in candidates:
        if Path(name).exists():
            return name
        cached = hub_cache / name
        if cached.exists():
            return str(cached)

    print("\n" + "=" * 60)
    print("ERROR: No local YOLO model (.pt) found.")
    print("=" * 60)
    print("To fix this, connect to the internet once and run:")
    print("  python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")
    print("This downloads a ~6 MB model. After that, re-run this script.")
    print("=" * 60)
    sys.exit(1)


def count_vehicles(results, names):
    """
    Given YOLO results and class-name dict, return:
        counts  – {label: count}
        category_counts – {emission_category: count}
    """
    counts = {}
    category_counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = names[cls_id].lower()
        if cls_id not in VEHICLE_COCO_IDS and label not in VEHICLE_CLASS_MAP:
            continue

        counts[label] = counts.get(label, 0) + 1

        category = VEHICLE_CLASS_MAP.get(label, "car")   # default to car if unknown
        category_counts[category] = category_counts.get(category, 0) + 1

    return counts, category_counts


def estimate_emissions(category_counts: dict, distance_km: float = 1.0) -> dict:
    """
    Return per-category and total CO2 emissions (grams) for `distance_km`.
    Default distance = 1 km (per-km snapshot of the road).
    """
    breakdown = {}
    for cat, cnt in category_counts.items():
        factor = EMISSIONS_G_PER_KM.get(cat, 120)  # fallback = car
        breakdown[cat] = {
            "count":        cnt,
            "g_CO2_per_km": factor,
            "total_g_CO2":  cnt * factor * distance_km,
        }
    return breakdown


def print_report(counts: dict, category_counts: dict, emission_breakdown: dict):
    """Pretty-print the vehicle count and emissions report."""
    print("\n" + "=" * 50)
    print("  VEHICLE COUNT REPORT")
    print("=" * 50)

    if not counts:
        print("  No vehicles detected.")
    else:
        print(f"  {'Vehicle Type':<20} {'Count':>5}")
        print("  " + "-" * 26)
        for label, cnt in sorted(counts.items()):
            print(f"  {label:<20} {cnt:>5}")
        print(f"  {'TOTAL':<20} {sum(counts.values()):>5}")

    print("\n" + "=" * 50)
    print("  ESTIMATED CO2 EMISSIONS  (per 1 km of road)")
    print("=" * 50)

    if not emission_breakdown:
        print("  No vehicle emissions to report.")
    else:
        total_co2 = 0
        print(f"  {'Category':<15} {'Count':>5}  {'g CO2/km/veh':>13}  {'Total g CO2':>12}")
        print("  " + "-" * 52)
        for cat, info in sorted(emission_breakdown.items()):
            print(f"  {cat:<15} {info['count']:>5}  {info['g_CO2_per_km']:>13}  {info['total_g_CO2']:>12.1f}")
            total_co2 += info["total_g_CO2"]
        print("  " + "-" * 52)
        print(f"  {'TOTAL CO2':<15}        {'':>13}  {total_co2:>12.1f} g/km")
        print(f"  ≈ {total_co2/1000:.3f} kg CO2 per km of this road segment")

    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# 4. Annotate frame with bounding boxes
# ---------------------------------------------------------------------------
def annotate_frame(frame, results, names, category_counts):
    """Draw bounding boxes on frame, label only vehicle classes."""
    COLORS = {
        "car":        (0, 200, 255),
        "motorcycle": (0, 150, 200),
        "bus":        (50, 50, 255),
        "truck":      (200, 50, 50),
        "van":        (200, 150, 0),
        "bicycle":    (0, 200, 0),
    }

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = names[cls_id].lower()
        if cls_id not in VEHICLE_COCO_IDS and label not in VEHICLE_CLASS_MAP:
            continue

        category = VEHICLE_CLASS_MAP.get(label, "car")
        color = COLORS.get(category, (255, 255, 255))

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # HUD overlay – vehicle counts
    y_off = 28
    cv2.rectangle(frame, (0, 0), (240, 22 + 22 * len(category_counts) + 10),
                  (0, 0, 0), -1)
    cv2.putText(frame, "Vehicle Counts:", (8, y_off - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 200), 1)
    for cat, cnt in sorted(category_counts.items()):
        cv2.putText(frame, f"  {cat}: {cnt}", (8, y_off + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
        y_off += 20

    return frame


def main():
    parser = argparse.ArgumentParser(description="Vehicle counter + emissions estimator")
    parser.add_argument("--source", default="derek-lee-C031beKcwdQ-unsplash.jpg",
                        help="Image path, video path, or webcam index (0, 1, …)")
    parser.add_argument("--model",  default="yolov8n.pt",
                        help="Path to local YOLO .pt model")
    parser.add_argument("--conf",   type=float, default=0.35,
                        help="Confidence threshold (default 0.35)")
    parser.add_argument("--show",   action="store_true", default=True,
                        help="Display annotated output window")
    args = parser.parse_args()

    model_path = find_local_model(args.model)
    print(f"[INFO] Loading model: {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    names = model.names

    source = args.source
    is_webcam = source.isdigit()
    if is_webcam:
        source = int(source)
        print(f"[INFO] Real-time webcam mode (device {source}). Press 'q' to quit.")
    elif not os.path.exists(source):
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)
    else:
        print(f"[INFO] Processing image/video: {source}")

    if is_webcam or str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {source}")
            sys.exit(1)

        print("[INFO] Processing frames… Press 'q' to quit.")
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=args.conf, verbose=False)
            counts, category_counts = count_vehicles(results, names)
            frame = annotate_frame(frame, results, category_counts=category_counts,
                                   names=names)

            if args.show:
                cv2.imshow("Vehicle Counter + Emissions Estimator", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:
                emission_breakdown = estimate_emissions(category_counts)
                print(f"\n[Frame {frame_id}]")
                print_report(counts, category_counts, emission_breakdown)

            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()

    else:
        results = model(source, conf=args.conf)
        counts, category_counts = count_vehicles(results, names)
        emission_breakdown       = estimate_emissions(category_counts)
        print_report(counts, category_counts, emission_breakdown)

        if args.show:
            frame = cv2.imread(source)
            if frame is not None:
                frame = annotate_frame(frame, results, names, category_counts)
                cv2.imshow("Vehicle Counter + Emissions Estimator", frame)
                print("[INFO] Press any key in the image window to close.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()