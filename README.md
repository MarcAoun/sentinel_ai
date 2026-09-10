# Sentinel AI

Raspberry Pi doorbell cam. Recognizes faces, unlocks for people it knows, flags packages.

## Setup
```bash
mkdir -p ~/sentinel_ai && cd ~/sentinel_ai
chmod +x setup.sh && ./setup.sh   # dlib compile takes 15-30 min, be patient
source venv/bin/activate
python3 download_model.py
python3 main.py                   # or --picamera / --no-objects
```

## Controls
`q` quit, `e` enroll a face, `l` event log, `s` status

## How it works
Camera checks faces against the known-faces database. Recognized faces get greeted and "unlock" the door, unknown ones get flagged and can be enrolled on the spot with `e`. Packages trigger a notification.

## Structure
```
sentinel_ai/
├── main.py / config.py / setup.sh / download_model.py
├── modules/  (camera, face_manager, object_detector, notifier)
├── data/     (known_faces, encodings.pkl)
└── models/
```
