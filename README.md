# Face Detection & Recognition App

Real-time webcam app that detects faces and identifies them against a personal whitelist,
with an optional fallback to AWS Rekognition for celebrity recognition.

## Color key

| Box color | Meaning |
|---|---|
| **Green** | Whitelist match (≥ 80% similarity) — shows name and score |
| **Yellow** | Celebrity identified via AWS Rekognition (≥ 80% confidence) — shows ★ name and score |
| **Red** | No match |

---

## Quick start

### 1. Install system dependencies (required for dlib)

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev
```

**macOS (Homebrew):**
```bash
brew install cmake
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Verify:
```bash
python -c "import face_recognition, cv2, boto3; print('All imports OK')"
```

### 3. Add people to the whitelist

**Option A — one image per person (flat):**
```
whitelist/
├── alice.jpg
└── bob.png
```

**Option B — multiple images per person (more robust):**
```
whitelist/
├── alice/
│   ├── front.jpg
│   └── angle.jpg
└── bob/
    └── photo.jpg
```

Rules:
- The filename stem (Option A) or directory name (Option B) becomes the display name.
- Each image should contain exactly one clear, front-facing face.
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Restart `app.py` after adding images — the whitelist is loaded once at startup.

### 4. Run

```bash
python app.py
```

Press **`q`** to quit.

---

## AWS Rekognition setup (optional)

When a face doesn't match your whitelist, the app can query AWS Rekognition's
`RecognizeCelebrities` API to identify public figures.

### Configure credentials (choose one method)

**Environment variables:**
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

**AWS credentials file (`~/.aws/credentials`):**
```ini
[default]
aws_access_key_id = your_key
aws_secret_access_key = your_secret
region = us-east-1
```

If credentials are not configured, the app runs normally — unknown faces just show "No Match".

**Free tier:** AWS offers 5,000 Rekognition image analyses per month for the first 12 months.

---

## Configuration

Edit the constants at the top of `app.py`:

| Constant | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.80` | Minimum score (0–1) for a whitelist match to count |
| `CELEBRITY_THRESHOLD` | `0.80` | Minimum AWS confidence (0–1) to show a celebrity name |
| `SCALE_FACTOR` | `0.5` | Resize factor for detection (lower = faster but less accurate) |
| `REKOGNITION_COOLDOWN` | `1.0` | Minimum seconds between AWS API calls |

---

## Troubleshooting

**`dlib` build fails:**
Make sure `cmake` and `build-essential` are installed before running `pip install`.

**Low FPS:**
Lower `SCALE_FACTOR` (e.g. `0.25`) or reduce your camera resolution.

**False negatives (known person not recognised):**
Add more reference images with varied angles and lighting conditions.

**Camera not found:**
Check that a webcam is connected. Edit `cv2.VideoCapture(0)` in `app.py` to try a different index (1, 2, ...).

**AWS credentials error:**
Run `aws sts get-caller-identity` to verify your credentials are configured correctly.
