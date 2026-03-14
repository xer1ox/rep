# CLAUDE.md — AI Assistant Guide

## Project Overview

Real-time face detection and recognition application using a webcam feed. Faces are matched against a local whitelist; unrecognised faces optionally query AWS Rekognition's celebrity API.

**Single entry point:** `app.py`
**Language:** Python 3
**Key libraries:** `face_recognition`, `opencv-python`, `numpy`, `boto3`

---

## Repository Structure

```
rep/
├── app.py              # Main application (all logic lives here)
├── requirements.txt    # Python dependencies (pinned/constrained)
├── README.md           # User-facing quickstart and configuration guide
├── whitelist/          # Reference face images (empty by default, .gitkeep present)
│   ├── <name>.jpg      # Flat layout: one image per person
│   └── <name>/         # Subdirectory layout: multiple images per person
│       └── photo.jpg
└── CLAUDE.md           # This file
```

---

## Architecture

`app.py` is a single-file script with five logical sections:

| Section | Purpose |
|---|---|
| Configuration constants | Thresholds, colors, timing — edit here to tune behavior |
| `load_whitelist()` | Reads `whitelist/` at startup; returns `{name: [encoding, ...]}` |
| `match_face()` | Compares a live encoding against all whitelist entries; returns `(name, score 0–1)` |
| `init_rekognition()` / `query_rekognition()` | Optional AWS celebrity lookup; gracefully degrades if boto3 or credentials are absent |
| `main()` | Camera loop: capture → downscale → detect → match → draw → display |

### Detection pipeline (per frame)

1. Frame captured from webcam (`cv2.VideoCapture`)
2. Downscaled by `SCALE_FACTOR` (default 0.5) for speed
3. `face_recognition.face_locations()` → bounding boxes
4. `face_recognition.face_encodings()` → 128-d vectors
5. Each encoding compared against whitelist via `face_recognition.face_distance()`
6. Score = `max(0, 1 - distance / DISTANCE_SCALE)` — maps distance → similarity %
7. If score ≥ `SIMILARITY_THRESHOLD` → green box (whitelist match)
8. Else check `celebrity_cache` (refreshed every `REKOGNITION_COOLDOWN` seconds) → yellow box
9. Else → red "No Match" box
10. Bounding-box coords scaled back to original resolution before drawing

### Caching strategy

Rekognition results are cached in `celebrity_cache` keyed by `(snap_top, snap_right, snap_bottom, snap_left)` where `snap()` rounds coordinates to the nearest `CACHE_SNAP` pixels (default 20). This tolerates minor face movement without re-calling the API every frame. Nearby keys (within 2×`CACHE_SNAP`) are also checked.

---

## Configuration Constants (`app.py` top section)

| Constant | Default | Effect |
|---|---|---|
| `WHITELIST_DIR` | `"whitelist"` | Path to reference images directory |
| `SIMILARITY_THRESHOLD` | `0.80` | Min score to count as whitelist match |
| `CELEBRITY_THRESHOLD` | `0.80` | Min AWS confidence to show celebrity name |
| `SCALE_FACTOR` | `0.5` | Downscale before detection (lower = faster, less accurate) |
| `DISTANCE_SCALE` | `0.6` | Face distance that maps to 0% similarity |
| `REKOGNITION_COOLDOWN` | `1.0` | Seconds between AWS API calls |
| `CACHE_SNAP` | `20` | Pixel rounding for bbox cache keys |

---

## Development Workflow

### Setup

```bash
# System deps (Ubuntu/Debian — required for dlib which backs face_recognition)
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev

# Python env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running

```bash
python app.py
```

Press `q` to quit. Whitelist is loaded once at startup — restart after adding images.

### Dependencies

```
face_recognition==1.3.0      # dlib-based face encoding (pinned)
opencv-python==4.10.0.84     # webcam capture and drawing (pinned)
numpy>=1.24,<2.0             # array math
boto3>=1.34                  # AWS Rekognition (optional at runtime)
```

`boto3` is always installed but the Rekognition client is only used if valid AWS credentials are present. The app falls back gracefully when they are not.

### No test suite, no CI

There are no automated tests or CI pipelines. Manual testing requires a connected webcam.

---

## Whitelist Management

### Adding people

**Flat (one image):**
```
whitelist/alice.jpg
```

**Subdirectory (multiple images, more robust):**
```
whitelist/alice/front.jpg
whitelist/alice/side.jpg
```

- Display name = filename stem (flat) or directory name (subdirectory)
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Each image must contain exactly one clear, front-facing face
- Images with no detectable face are skipped with a `[WARN]` message
- Restart `app.py` after changes

### Accuracy tips

- 3–5 varied images per person (angles, lighting) improves recall
- `SIMILARITY_THRESHOLD` can be lowered (e.g. `0.70`) for more lenient matching or raised for stricter matching

---

## AWS Rekognition (optional)

The app calls `rekognition.recognize_celebrities()` on the full-resolution frame.

**Credential setup (choose one):**

```bash
# Environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# or use ~/.aws/credentials (standard boto3 credential chain)
```

The client is initialised with a lightweight `list_collections` probe. If that fails for any reason (no credentials, no network, wrong region), Rekognition is silently disabled for the session.

---

## Key Conventions

- **All logic in `app.py`** — do not split into separate modules unless the file grows substantially.
- **Constants at the top** — tunable values live in the configuration block, not scattered through functions.
- **Graceful degradation** — AWS dependency is always optional; never make it required for core functionality.
- **BGR color order** — OpenCV uses BGR, not RGB. All color tuples in the codebase are BGR.
- **RGB for face_recognition** — `face_recognition` expects RGB; frames are converted with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before encoding.
- **Downscale for detection, full-res for drawing** — detect on small frame, scale coords back (`* scale_inv`) before drawing.
- **No GPU dependency** — detection uses the HOG model (`model="hog"`), not CNN, so no CUDA required.

---

## Common Issues

| Symptom | Fix |
|---|---|
| `dlib` build fails during `pip install` | Install `cmake` and `build-essential` first |
| Low FPS | Lower `SCALE_FACTOR` (e.g. `0.25`) |
| Known person not recognised | Add more whitelist images; lower `SIMILARITY_THRESHOLD` |
| Camera not found | Try `cv2.VideoCapture(1)` or higher index |
| AWS credentials error | Run `aws sts get-caller-identity` to verify |
| All faces show "No Match" | Whitelist is empty; add images to `whitelist/` and restart |
