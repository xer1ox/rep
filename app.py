"""
app.py — Real-time face detection and recognition

- Green box  : Whitelist match (≥80% similarity)
- Yellow box : Celebrity identified via AWS Rekognition (≥80% confidence)
- Red box    : No match

Whitelist setup:
  Flat:        whitelist/<name>.jpg
  Subdirectory: whitelist/<name>/photo1.jpg  (multiple images = better accuracy)

AWS Rekognition (optional):
  Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  or configure ~/.aws/credentials
  If not configured, unknown faces show "No Match" instead.

Press 'q' to quit.
"""

import time
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
WHITELIST_DIR        = "whitelist"
SIMILARITY_THRESHOLD = 0.80   # 0–1: minimum score for whitelist match
CELEBRITY_THRESHOLD  = 0.80   # 0–1: minimum confidence for celebrity match
SCALE_FACTOR         = 0.5    # downscale factor for detection speed
DISTANCE_SCALE       = 0.6    # face_recognition distance that maps to 0% similarity
REKOGNITION_COOLDOWN = 1.0    # seconds between AWS API calls
CACHE_SNAP           = 20     # round bbox coords to this many px for cache key

# BGR colors
COLOR_MATCH     = (0, 255, 0)    # green  — whitelist match
COLOR_CELEBRITY = (0, 215, 255)  # yellow — celebrity
COLOR_NO_MATCH  = (0, 0, 255)    # red    — no match


# ── Whitelist loading ─────────────────────────────────────────────────────────
def load_whitelist(directory: str) -> dict:
    """Load all reference images from `directory` and return
    {name: [encoding, ...]} for each known person."""
    db = defaultdict(list)
    root = Path(directory)

    if not root.exists():
        print(f"[WARN] Whitelist directory '{directory}' not found — creating it.")
        root.mkdir(parents=True)
        return {}

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # Determine display name from path
        if path.parent == root:
            name = path.stem          # flat: whitelist/alice.jpg → "alice"
        else:
            name = path.parent.name   # subdir: whitelist/alice/photo.jpg → "alice"

        image = face_recognition.load_image_file(str(path))
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"[WARN] No face detected in '{path}' — skipping.")
            continue
        db[name].append(encodings[0])
        print(f"  Loaded: {name} ({path.name})")

    return dict(db)


# ── Face matching ─────────────────────────────────────────────────────────────
def match_face(encoding: np.ndarray, whitelist_db: dict) -> tuple:
    """Compare `encoding` against all whitelist entries.
    Returns (best_name, best_score) where score is in [0.0, 1.0]."""
    best_name  = None
    best_score = 0.0

    for name, known_encodings in whitelist_db.items():
        distances = face_recognition.face_distance(known_encodings, encoding)
        min_dist  = float(np.min(distances))
        score     = max(0.0, 1.0 - (min_dist / DISTANCE_SCALE))
        if score > best_score:
            best_score = score
            best_name  = name

    return best_name, best_score


# ── AWS Rekognition (optional) ────────────────────────────────────────────────
def init_rekognition():
    """Try to initialise a boto3 Rekognition client.
    Returns the client or None if boto3/credentials are unavailable."""
    try:
        import boto3
        client = boto3.client("rekognition")
        # Lightweight test to verify credentials are present
        client.list_collections(MaxResults=1)
        print("[INFO] AWS Rekognition enabled.")
        return client
    except Exception as exc:
        print(f"[INFO] AWS Rekognition not available ({exc}) — celebrity lookup disabled.")
        return None


def query_rekognition(client, frame_bgr: np.ndarray) -> list:
    """Send `frame_bgr` to RecognizeCelebrities.
    Returns a list of dicts with keys: name, confidence, bbox (top,right,bottom,left)."""
    _, buf     = cv2.imencode(".jpg", frame_bgr)
    response   = client.recognize_celebrities(Image={"Bytes": buf.tobytes()})
    results    = []
    h, w       = frame_bgr.shape[:2]

    for celeb in response.get("CelebrityFaces", []):
        bb   = celeb["Face"]["BoundingBox"]
        conf = celeb["MatchConfidence"] / 100.0   # AWS returns 0–100
        results.append({
            "name":       celeb["Name"],
            "confidence": conf,
            "bbox": (
                int(bb["Top"]    * h),
                int((bb["Left"] + bb["Width"])  * w),
                int((bb["Top"]  + bb["Height"]) * h),
                int(bb["Left"]  * w),
            ),
        })
    return results


def snap(value: int) -> int:
    """Round value to nearest CACHE_SNAP for stable cache keys despite minor movement."""
    return round(value / CACHE_SNAP) * CACHE_SNAP


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_result(frame, top: int, right: int, bottom: int, left: int,
                label: str, color: tuple) -> None:
    """Draw a colored bounding box with a filled label banner above it."""
    # Box border
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Label background (filled rectangle above the box)
    label_h = 26
    cv2.rectangle(frame, (left, top - label_h), (right, top), color, cv2.FILLED)

    # White label text
    cv2.putText(
        frame, label,
        (left + 4, top - 7),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        cv2.LINE_AA,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Face Detection & Recognition App")
    print("=" * 50)

    # Load whitelist
    print(f"\nLoading whitelist from '{WHITELIST_DIR}/' ...")
    whitelist_db = load_whitelist(WHITELIST_DIR)
    if whitelist_db:
        print(f"Loaded {len(whitelist_db)} person(s): {', '.join(whitelist_db)}")
    else:
        print("Whitelist is empty — all faces will show 'No Match' unless recognised as celebrities.")

    # Init Rekognition (optional)
    rek_client             = init_rekognition()
    last_rek_time          = 0.0
    celebrity_cache: dict  = {}   # (snap_top, snap_right, snap_bottom, snap_left) → {name, confidence}

    # Open webcam
    print("\nOpening camera (index 0) ...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera at index 0. Check that a webcam is connected.")
    print("Camera open. Press 'q' to quit.\n")

    scale_inv = int(1 / SCALE_FACTOR)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to capture frame — retrying ...")
            continue

        # ── Detection on downscaled frame ──
        small     = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, locations)

        # ── Optional: refresh Rekognition cache ──
        now = time.monotonic()
        if rek_client and (now - last_rek_time) >= REKOGNITION_COOLDOWN:
            try:
                rek_results    = query_rekognition(rek_client, frame)
                last_rek_time  = now
                celebrity_cache = {}
                for r in rek_results:
                    t, ri, b, l = r["bbox"]
                    key = (snap(t), snap(ri), snap(b), snap(l))
                    celebrity_cache[key] = {"name": r["name"], "confidence": r["confidence"]}
            except Exception as exc:
                print(f"[WARN] Rekognition error: {exc}")

        # ── Per-face decision & drawing ──
        for (top, right, bottom, left), enc in zip(locations, encodings):
            # Scale coords back to original resolution
            top    *= scale_inv
            right  *= scale_inv
            bottom *= scale_inv
            left   *= scale_inv

            name, score = match_face(enc, whitelist_db)

            if score >= SIMILARITY_THRESHOLD:
                # ── Whitelist match ──
                label = f"{name}  {score:.0%}"
                color = COLOR_MATCH

            else:
                # ── Try celebrity cache ──
                cache_key    = (snap(top), snap(right), snap(bottom), snap(left))
                celeb_entry  = celebrity_cache.get(cache_key)

                # Also check nearby snapped keys (face may drift a little)
                if celeb_entry is None:
                    for key, val in celebrity_cache.items():
                        kt, kr, kb, kl = key
                        if (abs(kt - snap(top))    <= CACHE_SNAP * 2 and
                                abs(kr - snap(right))  <= CACHE_SNAP * 2 and
                                abs(kb - snap(bottom)) <= CACHE_SNAP * 2 and
                                abs(kl - snap(left))   <= CACHE_SNAP * 2):
                            celeb_entry = val
                            break

                if celeb_entry and celeb_entry["confidence"] >= CELEBRITY_THRESHOLD:
                    label = f"\u2605 {celeb_entry['name']}  {celeb_entry['confidence']:.0%}"
                    color = COLOR_CELEBRITY
                else:
                    label = "No Match"
                    color = COLOR_NO_MATCH

            draw_result(frame, top, right, bottom, left, label, color)

        cv2.imshow("Face Recognition  |  q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
