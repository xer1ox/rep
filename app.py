"""
app.py — Real-time face detection and recognition (Windows-compatible)

On first run, DeepFace downloads model weights (~92 MB). Subsequent runs are instant.

Box colours:
  Green  — whitelist match (≥ 80% similarity) — shows name and score
  Red    — no match

Whitelist setup:
  Flat layout   :  whitelist/<name>.jpg   (one image per person)
  Subdirectory  :  whitelist/<name>/photo1.jpg  (multiple images = better accuracy)

Press 'q' to quit.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from deepface import DeepFace

# ── Configuration ─────────────────────────────────────────────────────────────
WHITELIST_DIR        = "whitelist"
MODEL_NAME           = "Facenet512"   # downloads ~92 MB on first run; cached afterwards
DETECTOR_BACKEND     = "opencv"       # fast, ships with opencv-python — no extras needed
SIMILARITY_THRESHOLD = 80.0           # percent — faces scoring ≥ this count as a match
PROCESS_EVERY_N      = 3              # run recognition on every Nth frame (boosts FPS)
MIN_FACE_SIZE        = (80, 80)       # ignore detections smaller than this (px)

# BGR colours (OpenCV uses BGR, not RGB)
COLOR_MATCH    = (0, 220, 0)    # green
COLOR_NO_MATCH = (0, 0, 220)    # red


# ── Whitelist loading ─────────────────────────────────────────────────────────
def load_whitelist(directory: str) -> dict:
    """
    Scan `directory` for images, extract a face embedding from each, and return
    {name: [embedding, ...]} ready for comparison.

    Flat layout   → filename stem becomes the name  (alice.jpg → "alice")
    Subdirectory  → parent folder name becomes the name  (alice/front.jpg → "alice")
    """
    db = defaultdict(list)
    root = Path(directory)

    if not root.exists():
        root.mkdir(parents=True)
        print(f"[INFO] Created '{directory}/' — add face images and restart.")
        return {}

    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    found_any = False

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in supported:
            continue
        found_any = True

        # Derive display name from path structure
        name = path.parent.name if path.parent != root else path.stem

        try:
            results = DeepFace.represent(
                img_path=str(path),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
        except Exception as exc:
            print(f"[WARN] Could not process '{path.name}': {exc}")
            continue

        if not results:
            print(f"[WARN] No face found in '{path.name}' — skipping.")
            continue
        if len(results) > 1:
            print(f"[WARN] Multiple faces in '{path.name}' — using the first one only.")

        db[name].append(results[0]["embedding"])
        print(f"  [OK] {name}  ({path.name})")

    if not found_any:
        print(f"[INFO] '{directory}/' is empty — all faces will show 'No Match'.")

    return dict(db)


# ── Similarity ────────────────────────────────────────────────────────────────
def cosine_similarity_pct(vec_a: list, vec_b: list) -> float:
    """Cosine similarity between two embedding vectors, expressed as a percentage."""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom) * 100.0


def find_match(embedding: list, db: dict) -> tuple:
    """
    Compare `embedding` against every entry in the whitelist db.
    Returns (best_name, best_score_pct, is_matched).
    """
    if not db:
        return None, 0.0, False

    best_name, best_score = None, 0.0
    for name, embeddings in db.items():
        score = max(cosine_similarity_pct(embedding, e) for e in embeddings)
        if score > best_score:
            best_score, best_name = score, name

    return best_name, best_score, best_score >= SIMILARITY_THRESHOLD


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_result(frame, x: int, y: int, w: int, h: int,
                label: str, color: tuple) -> None:
    """Draw a coloured bounding box with a filled label banner below it."""
    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Measure text so the banner fits perfectly
    font, scale, thickness = cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    banner_x1 = x
    banner_y1 = y + h + 1
    banner_x2 = x + text_w + 10
    banner_y2 = y + h + text_h + baseline + 12

    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2),
                  color, cv2.FILLED)
    cv2.putText(frame, label,
                (banner_x1 + 5, banner_y2 - baseline - 3),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Face Detection & Recognition App")
    print("=" * 52)
    print(f"\n  Model      : {MODEL_NAME}")
    print(f"  Threshold  : {SIMILARITY_THRESHOLD}%")
    print(f"\nLoading whitelist — this may take a moment on first run...")

    db = load_whitelist(WHITELIST_DIR)
    names = ", ".join(db) if db else "none"
    print(f"\n  Whitelist  : {len(db)} person(s) — {names}")

    # Open webcam (index 0 = default camera; change to 1, 2, … if needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[ERROR] Cannot open camera. Make sure a webcam is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Haar cascade for fast face detection (included with opencv-python)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Cache the latest recognition results so every displayed frame shows labels
    # even when recognition is skipped (runs every PROCESS_EVERY_N frames).
    cached: list = []   # [(x, y, w, h, label, color), ...]
    frame_n = 0

    print("\nPress 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame — retrying...")
            continue

        frame_n += 1

        if frame_n % PROCESS_EVERY_N == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=MIN_FACE_SIZE,
            )

            cached = []
            for (x, y, w, h) in faces:
                crop = frame[y: y + h, x: x + w]
                label, color = "No Match", COLOR_NO_MATCH

                try:
                    # Use detector_backend="skip" because we already cropped the face
                    res = DeepFace.represent(
                        img_path=crop,
                        model_name=MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=False,
                    )
                    if res:
                        name, score, matched = find_match(res[0]["embedding"], db)
                        if matched:
                            label = f"{name}  {score:.1f}%"
                            color = COLOR_MATCH
                except Exception:
                    pass  # keep "No Match" on any error

                cached.append((x, y, w, h, label, color))

        # Draw cached results on every frame (smooth visuals)
        for (x, y, w, h, label, color) in cached:
            draw_result(frame, x, y, w, h, label, color)

        # HUD
        cv2.putText(
            frame,
            f"People in whitelist: {len(db)}   |   Q to quit",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()
