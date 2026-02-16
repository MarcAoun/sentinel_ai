# Sentinel AI - Smart Security System
## Raspberry Pi Doorbell Camera with Face Recognition

---

## Features
- **Face Recognition** — Recognizes known people and greets them by name
- **Door Unlock** — Automatically "unlocks" the door for known faces
- **Face Enrollment** — Asks unknown visitors their name and remembers them
- **Object Detection** — Detects packages and sends notifications
- **Live Video Feed** — Real-time camera view with overlays

---

## Quick Setup

### Step 1: Copy project to Pi
```bash
cd ~
# If you already have the sentinel_ai folder, skip this
mkdir -p ~/sentinel_ai
```

### Step 2: Run setup script
```bash
cd ~/sentinel_ai
chmod +x setup.sh
./setup.sh
```
⚠️ **dlib takes 15-30 minutes to compile on a Raspberry Pi. Be patient!**

### Step 3: Download object detection model
```bash
cd ~/sentinel_ai
source venv/bin/activate
python3 download_model.py
```

### Step 4: Run Sentinel AI
```bash
cd ~/sentinel_ai
source venv/bin/activate

# With USB webcam
python3 main.py

# With Pi Camera
python3 main.py --picamera

# Without object detection (faster)
python3 main.py --no-objects
```

---

## Controls (while running)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `e` | Enroll an unknown face |
| `l` | Show event log |
| `s` | Show status |

---

## How It Works

1. **Camera starts** and begins processing frames
2. **Faces detected** → compared against known database
3. **Known face** → Greets by name + unlocks door (terminal message)
4. **Unknown face** → Alerts you + prompts for enrollment
5. **Press 'e'** → Type their name → Face saved to database
6. **Package detected** → Sends notification (terminal message)

---

## Project Structure
```
sentinel_ai/
├── main.py              # Main application
├── config.py            # All settings in one place
├── setup.sh             # Installation script
├── download_model.py    # Download object detection model
├── README.md            # This file
├── modules/
│   ├── camera.py        # Camera handling
│   ├── face_manager.py  # Face recognition & enrollment
│   ├── object_detector.py  # Package/object detection
│   └── notifier.py      # Notifications & alerts
├── data/
│   ├── known_faces/     # Stored face images
│   └── encodings.pkl    # Face encoding database
└── models/              # MobileNet SSD model files
```
