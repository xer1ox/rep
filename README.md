# Face Detection & Recognition App

Real-time webcam app that detects faces and checks them against a personal whitelist.
**No cmake, no build tools — installs with plain `pip` on Windows, macOS, and Linux.**

## Box colour key

| Colour | Meaning |
|--------|---------|
| **Green** | Whitelist match (≥ 80% similarity) — shows name and score |
| **Red** | No match |

---

## Quick start (Windows)

### 1. Create and activate a virtual environment

```bat
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bat
pip install -r requirements.txt
```

> **Note:** On the very first run `deepface` downloads the Facenet512 model weights
> (~92 MB). This is a one-time download; subsequent runs are instant.

Verify the installation:

```bat
python -c "import deepface, cv2; print('All imports OK')"
```

### 3. Add people to the whitelist

Place one or more clear, front-facing photos in the `whitelist/` folder.

**Option A — one image per person (simplest):**

```
whitelist/
├── alice.jpg
└── bob.png
```

The filename stem (`alice`, `bob`) becomes the display name shown on screen.

**Option B — multiple images per person (more accurate):**

```
whitelist/
├── alice/
│   ├── front.jpg
│   └── angle.jpg
└── bob/
    └── passport.jpg
```

Rules for reference images:
- Exactly **one** clearly visible face per image.
- Good lighting, front-facing — passport-style photos work best.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Restart `app.py` after adding or changing images (whitelist loads at startup).

### 4. Run

```bat
python app.py
```

Press **`q`** to quit.

---

## macOS / Linux

Same steps — no system packages required.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Configuration

Edit the constants at the top of `app.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `80.0` | Minimum score (%) for a match |
| `PROCESS_EVERY_N` | `3` | Run recognition every N frames (higher = faster but less responsive) |
| `MIN_FACE_SIZE` | `(80, 80)` | Minimum face size in pixels to detect |
| `MODEL_NAME` | `"Facenet512"` | DeepFace model — see [available models](https://github.com/serengil/deepface#face-recognition-models) |

---

## Troubleshooting

**Camera not found:**
Edit `cv2.VideoCapture(0)` in `app.py` — try index `1`, `2`, etc.

**Low FPS:**
Increase `PROCESS_EVERY_N` (e.g. `5`) or reduce camera resolution in `app.py`.

**Known person not recognised:**
Add more reference photos with varied angles and lighting. Ensure the whitelist image has exactly one face.

**`No module named 'deepface'`:**
Make sure your virtual environment is activated before running `pip install` and `python app.py`.
