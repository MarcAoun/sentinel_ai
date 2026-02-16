#!/usr/bin/env python3
"""
Sentinel AI - Desktop Security Application
A PyQt5-based smart security system for Raspberry Pi
"""

import sys
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image as PILImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QTabWidget, QGroupBox,
    QFormLayout, QSpinBox, QComboBox, QCheckBox, QLineEdit, QSlider,
    QDialog, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect,
    QProgressBar, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QLinearGradient, QPainter
import os
import subprocess
import time
import logging
import sounddevice as sd

# Voice assistant
import config as sentinel_config
from modules.voice_assistant import VoiceAssistant

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
CONFIDENCE_THRESHOLD = 0.5
KNOWN_FACES_DIR = "known_faces"

ENCODINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_encodings.pkl")

# =============================================================================
# COLOR PALETTE  (Apple-style light mode)
# =============================================================================
BG_MAIN = "#f2f2f7"
BG_CARD = "#ffffff"
BG_CARD_LIGHT = "#f9f9f9"
BG_INPUT = "#f2f2f7"
BG_CAMERA = "#1a1a2e"
BORDER = "#e5e5ea"
BORDER_LIGHT = "#d1d1d6"
ACCENT = "#007AFF"
ACCENT_DIM = "#0071e3"
GREEN = "#34C759"
GREEN_DIM = "#28a745"
RED = "#FF3B30"
RED_DIM = "#d63031"
ORANGE = "#FF9500"
PURPLE = "#8B5CF6"
TEXT = "#1c1c1e"
TEXT_DIM = "#8e8e93"
TEXT_MUTED = "#aeaeb2"


def load_known_faces():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        # Backward compat: add door_unlock list if missing
        if "door_unlock" not in data:
            data["door_unlock"] = [False] * len(data["names"])
        # Ensure list lengths match
        while len(data["door_unlock"]) < len(data["names"]):
            data["door_unlock"].append(False)
        return data
    return {"names": [], "encodings": [], "door_unlock": []}


def save_known_faces(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


# =============================================================================
# CAMERA THREAD
# =============================================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    detection_event = pyqtSignal(str, str)
    enrollment_finished = pyqtSignal(str)
    door_unlocked = pyqtSignal(str)  # emits person name when door unlocks

    def __init__(self, camera_index=DEFAULT_CAMERA_INDEX):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.detection_enabled = True
        self.face_recognition_enabled = True
        self.package_detection_enabled = True
        self.enrolling = False
        self.enroll_name = ""
        self.enroll_frames = []
        self.enroll_target = 10
        self.known_faces = load_known_faces()
        self.tolerance = CONFIDENCE_THRESHOLD
        self.frame_count = 0
        self.last_faces = []
        self.last_raw_frame = None
        self._door_unlock_cooldown = {}  # name -> last unlock timestamp

    def run(self):
        try:
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
            )
            picam2.configure(config)
            picam2.start()
        except Exception as e:
            self.detection_event.emit("error", f"Failed to open camera: {e}")
            return

        self.running = True
        self.detection_event.emit("system", "Camera started")

        while self.running:
            # Use PIL capture - convert to RGB to strip any alpha/padding channel
            pil_img = picam2.capture_image("main").convert("RGB")
            frame = np.array(pil_img)

            self.frame_count += 1
            self.last_raw_frame = frame
            display_frame = frame.copy()

            # Face detection every 3rd frame
            if self.detection_enabled and self.face_recognition_enabled and self.frame_count % 3 == 0:
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                face_locations = face_recognition.face_locations(small, model="hog")
                face_encodings = face_recognition.face_encodings(small, face_locations)

                faces = []
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2

                    name = "Unknown"
                    confidence = 0.0
                    best_idx = None
                    if self.known_faces["encodings"]:
                        distances = face_recognition.face_distance(
                            self.known_faces["encodings"], encoding
                        )
                        if len(distances) > 0:
                            best_idx = np.argmin(distances)
                            best_dist = distances[best_idx]
                            confidence = max(0.0, 1.0 - best_dist)
                            if best_dist < self.tolerance:
                                name = self.known_faces["names"][best_idx]

                    faces.append((top, right, bottom, left, name, confidence, encoding))

                    # Door unlock check for privileged faces
                    if name != "Unknown" and best_idx is not None:
                        if (len(self.known_faces.get("door_unlock", [])) > best_idx
                                and self.known_faces["door_unlock"][best_idx]):
                            now_ts = time.time()
                            last_unlock = self._door_unlock_cooldown.get(name, 0)
                            if now_ts - last_unlock > 30:  # 30s cooldown
                                self._door_unlock_cooldown[name] = now_ts
                                self.door_unlocked.emit(name)

                    if self.enrolling and name == "Unknown":
                        self.enroll_frames.append(encoding)
                        self.detection_event.emit(
                            "enroll",
                            f"Capturing... {len(self.enroll_frames)}/{self.enroll_target}"
                        )
                        if len(self.enroll_frames) >= self.enroll_target:
                            self._finish_enrollment()

                self.last_faces = faces

            # Draw face boxes (RGB colors since frame is RGB)
            for (top, right, bottom, left, name, confidence, _) in list(self.last_faces):
                if name != "Unknown":
                    color = (0, 230, 118)  # green
                    label_bg = (0, 180, 90)
                    label_text = f"{name} {confidence:.0%}"
                else:
                    color = (255, 145, 0)  # orange
                    label_bg = (200, 110, 0)
                    label_text = "Unknown"
                # Draw box with thicker line
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                # Corner accents
                corner_len = 15
                cv2.line(display_frame, (left, top), (left + corner_len, top), color, 3)
                cv2.line(display_frame, (left, top), (left, top + corner_len), color, 3)
                cv2.line(display_frame, (right, top), (right - corner_len, top), color, 3)
                cv2.line(display_frame, (right, top), (right, top + corner_len), color, 3)
                cv2.line(display_frame, (left, bottom), (left + corner_len, bottom), color, 3)
                cv2.line(display_frame, (left, bottom), (left, bottom - corner_len), color, 3)
                cv2.line(display_frame, (right, bottom), (right - corner_len, bottom), color, 3)
                cv2.line(display_frame, (right, bottom), (right, bottom - corner_len), color, 3)
                # Name label
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                label_w = max(right - left, tw + 12)
                label_h = 28
                cv2.rectangle(display_frame, (left, top - label_h), (left + label_w, top), label_bg, -1)
                cv2.putText(display_frame, label_text, (left + 6, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Enrollment overlay
            if self.enrolling:
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                cv2.putText(display_frame, f"ENROLLING: {self.enroll_name}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 212, 255), 2)
                pct = int(len(self.enroll_frames) / self.enroll_target * 100)
                bar_w = int((FRAME_WIDTH - 30) * pct / 100)
                cv2.rectangle(display_frame, (15, 42), (FRAME_WIDTH - 15, 56), (40, 50, 70), -1)
                cv2.rectangle(display_frame, (15, 42), (15 + bar_w, 56), (0, 212, 255), -1)
                cv2.putText(display_frame, f"{pct}%", (FRAME_WIDTH - 60, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, FRAME_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 160, 180), 1)

            self.frame_ready.emit(display_frame)
            self.msleep(int(1000 / FPS))

        picam2.stop()
        picam2.close()
        self.detection_event.emit("system", "Camera stopped")

    def _finish_enrollment(self):
        avg_encoding = np.mean(self.enroll_frames, axis=0)
        self.known_faces["names"].append(self.enroll_name)
        self.known_faces["encodings"].append(avg_encoding)
        if "door_unlock" not in self.known_faces:
            self.known_faces["door_unlock"] = [False] * (len(self.known_faces["names"]) - 1)
        self.known_faces["door_unlock"].append(False)
        save_known_faces(self.known_faces)
        self.detection_event.emit("enroll", f"Enrollment complete for: {self.enroll_name}")
        self.enrollment_finished.emit(self.enroll_name)
        self.enrolling = False
        self.enroll_frames = []
        self.enroll_name = ""

    def stop(self):
        self.running = False
        self.wait()

    def start_enrollment(self, name):
        self.enroll_name = name
        self.enroll_frames = []
        self.enrolling = True
        self.detection_event.emit("enroll", f"Starting enrollment for: {name}")

    def stop_enrollment(self):
        if self.enroll_frames:
            self._finish_enrollment()
        else:
            self.enrolling = False
            self.enroll_frames = []
            self.detection_event.emit("enroll", "Enrollment cancelled")
            self.enroll_name = ""


# =============================================================================
# VOICE ASSISTANT SIGNAL BRIDGE (thread-safe Qt <-> voice assistant)
# =============================================================================
class VoiceBridge(QObject):
    transcript_received = pyqtSignal(str)
    user_transcript_received = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    enroll_completed = pyqtSignal(str)


# =============================================================================
# ENROLL DIALOG
# =============================================================================
class EnrollDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enroll New Face")
        self.setFixedSize(420, 280)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("New Face Enrollment")
        title.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        layout.addWidget(title)

        desc = QLabel("Position the person in front of the camera.\nWe'll capture 10 frames to learn their face.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 0;")
        layout.addWidget(desc)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter person's name...")
        self.name_input.setFixedHeight(42)
        self.name_input.setFont(QFont("Helvetica Neue", 12))
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                padding: 0 14px;
                border: 1px solid {BORDER};
                border-radius: 10px;
                background-color: {BG_INPUT};
                color: {TEXT};
                font-size: 13px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        layout.addWidget(self.name_input)

        layout.addSpacing(8)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedHeight(40)
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_DIM};
                border: none;
                border-radius: 10px;
                padding: 0 20px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {BORDER}; color: {TEXT}; }}
        """)
        btn_cancel.clicked.connect(self.reject)

        btn_start = QPushButton("Start Enrollment")
        btn_start.setFixedHeight(40)
        btn_start.setCursor(Qt.PointingHandCursor)
        btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0 24px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_DIM}; }}
        """)
        btn_start.clicked.connect(self.validate_and_accept)

        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_start)
        layout.addLayout(btn_layout)

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 14px;
            }}
        """)

    def validate_and_accept(self):
        if self.name_input.text().strip():
            self.accept()
        else:
            self.name_input.setStyleSheet(f"""
                QLineEdit {{
                    padding: 0 14px;
                    border: 1px solid {RED};
                    border-radius: 10px;
                    background-color: {BG_INPUT};
                    color: {TEXT};
                }}
            """)

    def get_name(self):
        return self.name_input.text().strip()


# =============================================================================
# MAIN APP
# =============================================================================
class SentinelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sentinel AI")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 900)
        self.camera_thread = None

        # Voice assistant
        self.voice_bridge = VoiceBridge()
        self.voice_assistant = VoiceAssistant(
            on_transcript=self.voice_bridge.transcript_received.emit,
            on_user_transcript=self.voice_bridge.user_transcript_received.emit,
            on_status=self.voice_bridge.status_changed.emit,
            on_enroll=self._do_voice_enroll,
            on_remove=self._do_voice_remove,
            on_rename=self._do_voice_rename,
        )

        # Voice auto-greet / debounce state
        self._voice_stable_name = None
        self._voice_stable_since = 0
        self._voice_last_known_time = 0
        self._voice_greeted_name = None
        self._voice_greeted_time = 0
        self._voice_pending_greet = None  # (person_type, name) queued for after connect
        self._pending_unknown_encoding = None  # saved encoding for auto-enroll
        self._voice_enroll_suppress_until = 0  # suppress greetings after enrollment

        self.setup_ui()
        self.apply_theme()
        self.refresh_known_faces()

        # Connect voice signals
        self.voice_bridge.transcript_received.connect(self._on_voice_transcript)
        self.voice_bridge.user_transcript_received.connect(self._on_user_transcript)
        self.voice_bridge.status_changed.connect(self._on_voice_status)
        self.voice_bridge.enroll_completed.connect(self._on_voice_enroll_ui)

    def _card(self, layout_type="v"):
        card = QWidget()
        if layout_type == "v":
            lay = QVBoxLayout(card)
        else:
            lay = QHBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        card.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 16px;
            }}
        """)
        return card, lay

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # === HEADER BAR ===
        header = QWidget()
        header.setFixedHeight(52)
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_CARD};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 20, 0)

        h_layout.addStretch()

        # Center: title
        brand = QLabel("Sentinel AI")
        brand.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        brand.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        h_layout.addWidget(brand)

        h_layout.addStretch()

        # Right: status indicators
        self.header_status = QLabel("Offline")
        self.header_status.setFont(QFont("Helvetica Neue", 10))
        self.header_status.setStyleSheet(f"""
            color: {TEXT_MUTED};
            background-color: {BG_INPUT};
            border: none;
            border-radius: 12px;
            padding: 4px 14px;
        """)
        h_layout.addWidget(self.header_status)

        root.addWidget(header)

        # === MAIN CONTENT ===
        content = QWidget()
        content.setStyleSheet(f"background-color: {BG_MAIN}; border: none;")
        main_layout = QHBoxLayout(content)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # --- LEFT: Camera Panel ---
        cam_card, cam_layout = self._card()

        cam_header = QHBoxLayout()
        cam_title = QLabel("Live Feed")
        cam_title.setFont(QFont("Helvetica Neue", 14, QFont.Bold))
        cam_title.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        cam_header.addWidget(cam_title)
        cam_header.addStretch()

        self.btn_camera = QPushButton("Start")
        self.btn_camera.setFixedSize(100, 40)
        self.btn_camera.setCursor(Qt.PointingHandCursor)
        self.btn_camera.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
        self.btn_camera.clicked.connect(self.start_camera)
        self._style_btn(self.btn_camera, ACCENT)
        cam_header.addWidget(self.btn_camera)

        self.btn_camera_voice = QPushButton("Start + Voice")
        self.btn_camera_voice.setFixedSize(160, 40)
        self.btn_camera_voice.setCursor(Qt.PointingHandCursor)
        self.btn_camera_voice.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
        self.btn_camera_voice.clicked.connect(self._start_camera_with_voice)
        self._style_btn(self.btn_camera_voice, PURPLE)
        cam_header.addWidget(self.btn_camera_voice)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedSize(100, 40)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
        self.btn_stop.clicked.connect(self.stop_camera)
        self._style_btn(self.btn_stop, RED)
        self.btn_stop.setVisible(False)
        cam_header.addWidget(self.btn_stop)
        cam_layout.addLayout(cam_header)

        # Camera feed
        self.camera_feed = QLabel()
        self.camera_feed.setAlignment(Qt.AlignCenter)
        self.camera_feed.setMinimumSize(640, 480)
        self.camera_feed.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_feed.setText("Camera Off")
        self.camera_feed.setFont(QFont("Helvetica Neue", 14))
        self.camera_feed.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_CAMERA};
                border: 1px solid {BORDER};
                border-radius: 12px;
                color: {TEXT_MUTED};
            }}
        """)
        cam_layout.addWidget(self.camera_feed)

        # Status row under camera
        status_row = QHBoxLayout()
        status_row.setContentsMargins(6, 0, 0, 0)
        status_row.setSpacing(8)

        self.face_count_label = QLabel("Faces Detected: 0")
        self.face_count_label.setFont(QFont("Helvetica Neue", 10))
        self.face_count_label.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none;")
        status_row.addWidget(self.face_count_label)

        status_row.addStretch()
        cam_layout.addLayout(status_row)

        main_layout.addWidget(cam_card, stretch=3)

        # --- RIGHT: Sidebar ---
        self.sidebar = QTabWidget()
        self.sidebar.setMinimumWidth(340)
        self.sidebar.setMaximumWidth(420)

        # == HOME TAB ==
        home_tab = QWidget()
        home_tab.setStyleSheet(f"background-color: {BG_CARD}; border: none;")
        home_layout = QVBoxLayout(home_tab)
        home_layout.setContentsMargins(16, 16, 16, 16)
        home_layout.setSpacing(14)

        home_title = QLabel("Welcome to Sentinel AI")
        home_title.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        home_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        home_layout.addWidget(home_title)

        home_desc = QLabel("Your smart security assistant.\nMonitor, recognize, and interact.")
        home_desc.setWordWrap(True)
        home_desc.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 2px 0;")
        home_layout.addWidget(home_desc)

        # Dashboard cards
        def _home_stat(label_text, value_text, color):
            card = QWidget()
            card.setStyleSheet(f"""
                QWidget {{
                    background-color: {BG_INPUT};
                    border: 1px solid {BORDER};
                    border-radius: 12px;
                }}
            """)
            card_lay = QHBoxLayout(card)
            card_lay.setContentsMargins(14, 10, 14, 10)
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background-color: {color}; border-radius: 5px; border: none;")
            card_lay.addWidget(dot)
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Helvetica Neue", 12))
            lbl.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; border: none;")
            card_lay.addWidget(lbl)
            card_lay.addStretch()
            val = QLabel(value_text)
            val.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
            val.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
            card_lay.addWidget(val)
            return card, val

        cam_card_w, self.home_cam_status = _home_stat("Camera", "Offline", ORANGE)
        home_layout.addWidget(cam_card_w)

        faces_data = load_known_faces()
        n_enrolled = len(set(faces_data["names"]))
        faces_card_w, self.home_faces_val = _home_stat("Enrolled Faces", str(n_enrolled), GREEN)
        home_layout.addWidget(faces_card_w)

        voice_card_w, self.home_voice_status = _home_stat("Voice Assistant", "Inactive", PURPLE)
        home_layout.addWidget(voice_card_w)

        home_layout.addStretch()
        self.sidebar.addTab(home_tab, "Home")

        # == VOICE TAB ==
        voice_tab = QWidget()
        voice_tab.setStyleSheet(f"background-color: {BG_CARD}; border: none;")
        voice_layout = QVBoxLayout(voice_tab)
        voice_layout.setContentsMargins(16, 16, 16, 16)
        voice_layout.setSpacing(10)

        voice_title = QLabel("Voice Assistant")
        voice_title.setFont(QFont("Helvetica Neue", 15, QFont.Bold))
        voice_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        voice_layout.addWidget(voice_title)

        voice_desc = QLabel("Talk to Sentinel via your microphone.")
        voice_desc.setWordWrap(True)
        voice_desc.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 2px 0;")
        voice_layout.addWidget(voice_desc)

        # Voice status (hidden row)
        voice_status_row = QHBoxLayout()
        self.voice_status_dot = QLabel()
        self.voice_status_dot.setFixedSize(8, 8)
        self.voice_status_dot.setStyleSheet(f"background-color: {TEXT_MUTED}; border-radius: 4px; border: none;")
        voice_status_row.addWidget(self.voice_status_dot)
        self.voice_status_label = QLabel("Inactive")
        self.voice_status_label.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none; font-size: 11px;")
        voice_status_row.addWidget(self.voice_status_label)
        voice_status_row.addStretch()
        voice_layout.addLayout(voice_status_row)

        # Current voice & persona info
        voice_info_row = QHBoxLayout()
        voice_info_row.setSpacing(8)
        self.voice_info_label = QLabel(
            f"Voice: {sentinel_config.VOICE_VOICE.capitalize()}  |  "
            f"Persona: {sentinel_config.VOICE_PERSONA.replace('_', ' ').title()}"
        )
        self.voice_info_label.setStyleSheet(
            f"color: {TEXT_DIM}; background: transparent; border: none; font-size: 11px;"
        )
        voice_info_row.addWidget(self.voice_info_label)
        voice_info_row.addStretch()
        voice_layout.addLayout(voice_info_row)

        # Start/stop button
        self.btn_voice = QPushButton("Start Voice Session")
        self.btn_voice.setFixedHeight(44)
        self.btn_voice.setCursor(Qt.PointingHandCursor)
        self.btn_voice.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
        self.btn_voice.clicked.connect(self.toggle_voice)
        self._style_btn(self.btn_voice, PURPLE)
        voice_layout.addWidget(self.btn_voice)

        # Conversation area
        chat_label = QLabel("Conversation")
        chat_label.setFont(QFont("Helvetica Neue", 12, QFont.Bold))
        chat_label.setStyleSheet(f"color: {TEXT}; background: transparent; margin-top: 6px;")
        voice_layout.addWidget(chat_label)

        self.voice_chat = QTextEdit()
        self.voice_chat.setReadOnly(True)
        self.voice_chat.setFont(QFont("Helvetica Neue", 10))
        self.voice_chat.setPlaceholderText("No recent messages.")
        self.voice_chat.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_INPUT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 12px;
            }}
        """)
        voice_layout.addWidget(self.voice_chat)

        # Bottom action buttons
        voice_actions = QHBoxLayout()
        voice_actions.setSpacing(6)

        _action_style = f"""
            QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 4px 10px;
                font-size: 11px;
                outline: none;
            }}
            QPushButton:hover {{ background-color: {BORDER}; color: {TEXT}; }}
            QPushButton:focus {{ outline: none; border: 1px solid {BORDER}; }}
        """

        btn_clear = QPushButton("\u2716  Clear")
        btn_clear.setFixedHeight(32)
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.setStyleSheet(_action_style)
        btn_clear.clicked.connect(lambda: self.voice_chat.clear())
        voice_actions.addWidget(btn_clear)

        self.btn_mute = QPushButton("\U0001F3A4  Mute Mic")
        self.btn_mute.setFixedHeight(32)
        self.btn_mute.setCursor(Qt.PointingHandCursor)
        self.btn_mute.setStyleSheet(_action_style)
        self.btn_mute.clicked.connect(self._toggle_mic_mute)
        voice_actions.addWidget(self.btn_mute)

        voice_layout.addLayout(voice_actions)

        self.sidebar.addTab(voice_tab, "Voice")

        # == LOGS TAB ==
        log_tab = QWidget()
        log_tab.setStyleSheet(f"background-color: {BG_CARD}; border: none;")
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(16, 16, 16, 16)

        log_header = QHBoxLayout()
        log_title = QLabel("Detection Log")
        log_title.setFont(QFont("Helvetica Neue", 13, QFont.Bold))
        log_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        log_header.addWidget(log_title)
        log_header.addStretch()
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(60, 28)
        btn_clear.setCursor(Qt.PointingHandCursor)
        btn_clear.clicked.connect(self.clear_log)
        btn_clear.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {BORDER}; color: {TEXT}; }}
        """)
        log_header.addWidget(btn_clear)
        log_layout.addLayout(log_header)

        self.detection_log = QTextEdit()
        self.detection_log.setReadOnly(True)
        self.detection_log.setFont(QFont("SF Mono", 10))
        self.detection_log.setStyleSheet(f"""
            QTextEdit {{
                background-color: {BG_INPUT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 10px;
            }}
        """)
        log_layout.addWidget(self.detection_log)
        self.sidebar.addTab(log_tab, "Logs")

        # == ENROLLMENT TAB ==
        enroll_tab = QWidget()
        enroll_tab.setStyleSheet(f"background-color: {BG_CARD}; border: none;")
        enroll_layout = QVBoxLayout(enroll_tab)
        enroll_layout.setContentsMargins(16, 16, 16, 16)
        enroll_layout.setSpacing(12)

        enroll_title = QLabel("Face Enrollment")
        enroll_title.setFont(QFont("Helvetica Neue", 13, QFont.Bold))
        enroll_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        enroll_layout.addWidget(enroll_title)

        enroll_desc = QLabel("Register faces so Sentinel can recognize\npeople and greet them by name.")
        enroll_desc.setWordWrap(True)
        enroll_desc.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; padding: 2px 0;")
        enroll_layout.addWidget(enroll_desc)

        self.btn_enroll = QPushButton("Enroll New Face")
        self.btn_enroll.setFixedHeight(44)
        self.btn_enroll.setCursor(Qt.PointingHandCursor)
        self.btn_enroll.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
        self.btn_enroll.clicked.connect(self.start_enrollment)
        self._style_btn(self.btn_enroll, GREEN, text_color="#000")
        enroll_layout.addWidget(self.btn_enroll)

        self.btn_stop_enroll = QPushButton("Stop Enrollment")
        self.btn_stop_enroll.setFixedHeight(44)
        self.btn_stop_enroll.setVisible(False)
        self.btn_stop_enroll.setCursor(Qt.PointingHandCursor)
        self.btn_stop_enroll.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
        self.btn_stop_enroll.clicked.connect(self.stop_enrollment)
        self._style_btn(self.btn_stop_enroll, RED)
        enroll_layout.addWidget(self.btn_stop_enroll)

        known_label = QLabel("Registered Faces")
        known_label.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
        known_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent; margin-top: 8px;")
        enroll_layout.addWidget(known_label)

        self.known_faces_list = QListWidget()
        self.known_faces_list.setMinimumHeight(120)
        self.known_faces_list.setMaximumHeight(200)
        self.known_faces_list.setFont(QFont("Helvetica Neue", 11))
        self.known_faces_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {BG_INPUT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 12px;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background-color: {ACCENT};
                color: white;
            }}
            QListWidget::item:hover {{
                background-color: {BORDER};
            }}
        """)
        enroll_layout.addWidget(self.known_faces_list)

        enroll_btn_row = QHBoxLayout()
        enroll_btn_row.setSpacing(8)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedHeight(34)
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.clicked.connect(self.refresh_known_faces)
        btn_refresh.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {BORDER}; color: {TEXT}; }}
        """)
        enroll_btn_row.addWidget(btn_refresh)

        self.btn_toggle_access = QPushButton("Toggle Door Access")
        self.btn_toggle_access.setFixedHeight(34)
        self.btn_toggle_access.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_access.clicked.connect(self._toggle_door_access)
        self.btn_toggle_access.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 14px;
                outline: none;
            }}
            QPushButton:hover {{ background-color: {ACCENT_DIM}; }}
            QPushButton:focus {{ outline: none; }}
        """)
        enroll_btn_row.addWidget(self.btn_toggle_access)

        self.btn_remove_face = QPushButton("Remove")
        self.btn_remove_face.setFixedHeight(34)
        self.btn_remove_face.setCursor(Qt.PointingHandCursor)
        self.btn_remove_face.clicked.connect(self.remove_enrolled_face)
        self.btn_remove_face.setStyleSheet(f"""
            QPushButton {{
                background-color: {RED};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 14px;
                outline: none;
            }}
            QPushButton:hover {{ background-color: {RED_DIM}; }}
            QPushButton:focus {{ outline: none; }}
        """)
        enroll_btn_row.addWidget(self.btn_remove_face)

        enroll_layout.addLayout(enroll_btn_row)
        enroll_layout.addStretch()
        self.sidebar.addTab(enroll_tab, "Enrollment")

        # == SETTINGS TAB ==
        settings_tab = QWidget()
        settings_tab.setStyleSheet(f"background-color: {BG_CARD}; border: none;")
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(16, 16, 16, 16)
        settings_layout.setSpacing(10)

        s_title = QLabel("Settings")
        s_title.setFont(QFont("Helvetica Neue", 13, QFont.Bold))
        s_title.setStyleSheet(f"color: {TEXT}; background: transparent;")
        settings_layout.addWidget(s_title)

        det_group = QGroupBox("Detection")
        det_group.setStyleSheet(self._group_style())
        det_form = QFormLayout(det_group)
        det_form.setSpacing(8)

        self.chk_face_recognition = QCheckBox("Face Recognition")
        self.chk_face_recognition.setChecked(True)
        self.chk_face_recognition.setStyleSheet(self._check_style())
        det_form.addRow(self.chk_face_recognition)

        self.chk_package_detection = QCheckBox("Package Detection")
        self.chk_package_detection.setChecked(True)
        self.chk_package_detection.setStyleSheet(self._check_style())
        det_form.addRow(self.chk_package_detection)

        self.confidence_spinner = QSpinBox()
        self.confidence_spinner.setRange(10, 100)
        self.confidence_spinner.setValue(int(CONFIDENCE_THRESHOLD * 100))
        self.confidence_spinner.setSuffix("%")
        self.confidence_spinner.setMinimumWidth(100)
        self.confidence_spinner.setStyleSheet(self._spin_style())
        det_form.addRow("Confidence:", self.confidence_spinner)
        settings_layout.addWidget(det_group)

        notif_group = QGroupBox("Notifications")
        notif_group.setStyleSheet(self._group_style())
        notif_form = QFormLayout(notif_group)

        self.chk_notifications = QCheckBox("Enable Notifications")
        self.chk_notifications.setChecked(True)
        self.chk_notifications.setStyleSheet(self._check_style())
        notif_form.addRow(self.chk_notifications)

        self.chk_sound = QCheckBox("Play Sounds")
        self.chk_sound.setChecked(True)
        self.chk_sound.setStyleSheet(self._check_style())
        notif_form.addRow(self.chk_sound)
        settings_layout.addWidget(notif_group)

        # Speaker volume
        vol_group = QGroupBox("Speaker")
        vol_group.setStyleSheet(self._group_style())
        vol_form = QFormLayout(vol_group)
        vol_form.setSpacing(8)

        voice_sel_row = QHBoxLayout()
        voice_sel_row.setSpacing(6)
        self.voice_combo = QComboBox()
        voices = ["alloy", "ash", "coral", "echo", "fable", "nova", "shimmer"]
        self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText(sentinel_config.VOICE_VOICE)
        self.voice_combo.setStyleSheet(self._combo_style())
        self.voice_combo.currentTextChanged.connect(self._on_voice_changed)
        voice_sel_row.addWidget(self.voice_combo)

        btn_test_voice = QPushButton("Test")
        btn_test_voice.setFixedSize(50, 28)
        btn_test_voice.setCursor(Qt.PointingHandCursor)
        btn_test_voice.clicked.connect(self._test_voice)
        btn_test_voice.setStyleSheet(f"""
            QPushButton {{
                background-color: {PURPLE};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                outline: none;
            }}
            QPushButton:hover {{ background-color: #7C3AED; }}
            QPushButton:focus {{ outline: none; }}
        """)
        voice_sel_row.addWidget(btn_test_voice)
        vol_form.addRow("Voice:", voice_sel_row)

        # Persona selector
        persona_sel_row = QHBoxLayout()
        persona_sel_row.setSpacing(6)
        self.persona_combo = QComboBox()
        persona_labels = {
            "default": "Default (Chill)",
            "robot": "Robot",
            "pirate": "Pirate",
            "british_butler": "British Butler",
            "drill_sergeant": "Drill Sergeant",
            "professional": "Professional",
            "nonchalant": "Nonchalant",
        }
        for key, label in persona_labels.items():
            self.persona_combo.addItem(label, key)
        self.persona_combo.setCurrentIndex(0)
        self.persona_combo.setStyleSheet(self._combo_style())
        self.persona_combo.currentIndexChanged.connect(self._on_persona_changed)
        persona_sel_row.addWidget(self.persona_combo)

        btn_test_persona = QPushButton("Test")
        btn_test_persona.setFixedSize(50, 28)
        btn_test_persona.setCursor(Qt.PointingHandCursor)
        btn_test_persona.clicked.connect(self._test_persona)
        btn_test_persona.setStyleSheet(f"""
            QPushButton {{
                background-color: {PURPLE};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                outline: none;
            }}
            QPushButton:hover {{ background-color: #7C3AED; }}
            QPushButton:focus {{ outline: none; }}
        """)
        persona_sel_row.addWidget(btn_test_persona)
        vol_form.addRow("Persona:", persona_sel_row)

        vol_row = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(80)
        self.volume_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {BG_INPUT};
                height: 6px;
                border-radius: 3px;
                border: 1px solid {BORDER};
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
        """)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        vol_row.addWidget(self.volume_slider)
        self.volume_label = QLabel("80%")
        self.volume_label.setFixedWidth(46)
        self.volume_label.setStyleSheet(f"color: {TEXT_DIM}; background: transparent;")
        vol_row.addWidget(self.volume_label)
        vol_form.addRow("Volume:", vol_row)
        settings_layout.addWidget(vol_group)

        self.btn_apply = QPushButton("Apply Settings")
        self.btn_apply.setFixedHeight(40)
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.setFont(QFont("Helvetica Neue", 11, QFont.Bold))
        self.btn_apply.clicked.connect(self.apply_settings)
        self._style_btn(self.btn_apply, ACCENT)
        settings_layout.addWidget(self.btn_apply)
        settings_layout.addStretch()
        self.sidebar.addTab(settings_tab, "Settings")

        # Tab styling
        self.sidebar.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 16px;
                border-top-left-radius: 0;
            }}
            QTabBar::tab {{
                background-color: transparent;
                color: {TEXT_MUTED};
                padding: 10px 16px;
                border: none;
                border-bottom: 2px solid transparent;
                font-weight: 600;
                font-size: 12px;
            }}
            QTabBar::tab:selected {{
                color: {TEXT};
                border-bottom: 2px solid {ACCENT};
            }}
            QTabBar::tab:hover {{ color: {TEXT_DIM}; }}
        """)

        main_layout.addWidget(self.sidebar, stretch=1)
        root.addWidget(content)

        # === BOTTOM NAVIGATION BAR ===
        nav_bar = QWidget()
        nav_bar.setFixedHeight(90)
        nav_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_CARD};
                border-top: 1px solid {BORDER};
            }}
        """)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(10, 4, 10, 8)
        nav_layout.setSpacing(0)

        # Nav items: (icon, label, tab_index)
        nav_items = [
            ("\u2302", "Home", 0),
            ("\U0001F916", "Voice", 1),
            ("\U0001F4CB", "Logs", 2),
            ("\u25C9", "Enroll", 3),
            ("\u2699", "Settings", 4),
        ]
        self._nav_buttons = []
        for icon_char, label, tab_idx in nav_items:
            btn_container = QWidget()
            btn_vbox = QVBoxLayout(btn_container)
            btn_vbox.setSpacing(2)
            btn_vbox.setContentsMargins(0, 6, 0, 6)
            btn_vbox.setAlignment(Qt.AlignCenter)

            is_active = (label == "Home")
            color = ACCENT if is_active else TEXT_MUTED

            icon_lbl = QLabel(icon_char)
            icon_lbl.setAlignment(Qt.AlignCenter)
            icon_lbl.setFont(QFont("Helvetica Neue", 24))
            icon_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            btn_vbox.addWidget(icon_lbl)

            text_lbl = QLabel(label)
            text_lbl.setAlignment(Qt.AlignCenter)
            text_lbl.setFont(QFont("Helvetica Neue", 11, QFont.Bold if is_active else QFont.Normal))
            text_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            btn_vbox.addWidget(text_lbl)

            btn_container.setCursor(Qt.PointingHandCursor)
            btn_container.setProperty("tab_idx", tab_idx)
            btn_container.setProperty("nav_label", label)
            btn_container.setProperty("icon_lbl", icon_lbl)
            btn_container.setProperty("text_lbl", text_lbl)
            btn_container.mousePressEvent = lambda e, idx=tab_idx, c=btn_container: self._on_nav_click(idx, c)
            self._nav_buttons.append(btn_container)
            nav_layout.addWidget(btn_container)

        root.addWidget(nav_bar)

        # Sync bottom nav when tabs change from any source
        self.sidebar.currentChanged.connect(self._on_tab_changed)

        # Status bar (hidden — bottom nav replaces it)
        self.statusBar().hide()

    # =========================================================================
    # Navigation
    # =========================================================================
    def _on_nav_click(self, tab_idx, _clicked_container):
        self.sidebar.setCurrentIndex(tab_idx)

    def _on_tab_changed(self, tab_idx):
        for container in self._nav_buttons:
            is_active = (container.property("tab_idx") == tab_idx)
            color = ACCENT if is_active else TEXT_MUTED
            icon_lbl = container.property("icon_lbl")
            text_lbl = container.property("text_lbl")
            if icon_lbl:
                icon_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
            if text_lbl:
                text_lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
                text_lbl.setFont(QFont("Helvetica Neue", 11, QFont.Bold if is_active else QFont.Normal))

    # =========================================================================
    # Camera Controls
    # =========================================================================
    def _start_camera_with_voice(self):
        self.start_camera()
        self.start_voice()

    def start_camera(self):
        self.camera_thread = CameraThread(DEFAULT_CAMERA_INDEX)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.detection_event.connect(self.log_event)
        self.camera_thread.enrollment_finished.connect(self.on_enrollment_finished)
        self.camera_thread.door_unlocked.connect(self._on_door_unlocked)
        self.camera_thread.start()

        self.btn_camera.setVisible(False)
        self.btn_camera_voice.setVisible(False)
        self.btn_stop.setVisible(True)
        self.header_status.setText("Live")
        self.header_status.setStyleSheet(f"""
            color: {GREEN};
            background-color: rgba(52, 199, 89, 0.12);
            border: none;
            border-radius: 12px;
            padding: 4px 14px;
            font-size: 10px;
        """)
        self.home_cam_status.setText("Live")

    def stop_camera(self):
        # Stop voice if running
        if self.voice_assistant.is_active:
            self.stop_voice()

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.camera_feed.setText("Camera Off")
        self.camera_feed.setStyleSheet(f"""
            QLabel {{
                background-color: {BG_CAMERA};
                border: 1px solid {BORDER};
                border-radius: 12px;
                color: {TEXT_MUTED};
            }}
        """)
        self.btn_stop.setVisible(False)
        self.btn_camera.setVisible(True)
        self.btn_camera_voice.setVisible(True)
        self.face_count_label.setText("Faces Detected: 0")
        self.home_cam_status.setText("Offline")
        self.header_status.setText("Offline")
        self.header_status.setStyleSheet(f"""
            color: {TEXT_MUTED};
            background-color: {BG_INPUT};
            border: none;
            border-radius: 12px;
            padding: 4px 14px;
            font-size: 10px;
        """)

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = qt_image.scaled(
            self.camera_feed.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_feed.setPixmap(QPixmap.fromImage(scaled))
        # Update face count and voice context
        if self.camera_thread:
            n = len(self.camera_thread.last_faces)
            self.face_count_label.setText(f"Faces Detected: {n}")
            if self.voice_assistant.is_active:
                self._update_voice_context()

    # =========================================================================
    # Log
    # =========================================================================
    @pyqtSlot(str, str)
    def log_event(self, event_type, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "face": GREEN,
            "package": ORANGE,
            "system": ACCENT,
            "enroll": PURPLE,
            "error": RED,
        }
        color = color_map.get(event_type, TEXT_DIM)
        self.detection_log.append(
            f'<span style="color: {TEXT_MUTED};">[{timestamp}]</span> '
            f'<span style="color: {color};">{message}</span>'
        )
        scrollbar = self.detection_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(str)
    def _on_door_unlocked(self, name):
        self.log_event("face", f"Door unlocked for {name}")

    def clear_log(self):
        self.detection_log.clear()

    # =========================================================================
    # Enrollment
    # =========================================================================
    def start_enrollment(self):
        if not self.camera_thread or not self.camera_thread.running:
            QMessageBox.warning(self, "Camera Required", "Start the camera first.")
            return
        dialog = EnrollDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.get_name()
            self.camera_thread.start_enrollment(name)
            self.btn_enroll.setVisible(False)
            self.btn_stop_enroll.setVisible(True)

    def stop_enrollment(self):
        if self.camera_thread:
            self.camera_thread.stop_enrollment()
        self.btn_enroll.setVisible(True)
        self.btn_stop_enroll.setVisible(False)

    def on_enrollment_finished(self, name):
        self.btn_enroll.setVisible(True)
        self.btn_stop_enroll.setVisible(False)
        self.refresh_known_faces()
        QMessageBox.information(self, "Enrollment Complete", f"Successfully enrolled: {name}")

    def refresh_known_faces(self):
        data = load_known_faces()
        # Build unique names with their privilege status
        name_access = {}
        for i, n in enumerate(data["names"]):
            has_access = data.get("door_unlock", [False] * len(data["names"]))[i] if i < len(data.get("door_unlock", [])) else False
            # If any encoding for this name has access, they have access
            if n not in name_access:
                name_access[n] = has_access
            elif has_access:
                name_access[n] = True
        names = sorted(name_access.keys())
        self.known_faces_list.clear()
        if names:
            for n in names:
                label = f"{n}  [Door Access]" if name_access[n] else n
                self.known_faces_list.addItem(label)
        else:
            item = QListWidgetItem("No faces enrolled")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.known_faces_list.addItem(item)
        self.home_faces_val.setText(str(len(names)))

    def _toggle_door_access(self):
        item = self.known_faces_list.currentItem()
        if not item:
            return
        raw_text = item.text()
        if raw_text == "No faces enrolled":
            return
        name = raw_text.replace("  [Door Access]", "").strip()
        data = load_known_faces()
        # Toggle all encodings for this name
        toggled = False
        for i, n in enumerate(data["names"]):
            if n == name:
                current = data["door_unlock"][i]
                data["door_unlock"][i] = not current
                toggled = True
                new_state = not current
        if toggled:
            save_known_faces(data)
            if self.camera_thread:
                self.camera_thread.known_faces = load_known_faces()
            self.refresh_known_faces()
            status = "granted" if new_state else "revoked"
            self.log_event("system", f"Door access {status} for {name}")

    def remove_enrolled_face(self):
        item = self.known_faces_list.currentItem()
        if not item:
            return
        raw_text = item.text()
        if raw_text == "No faces enrolled":
            return
        name = raw_text.replace("  [Door Access]", "").strip()
        reply = QMessageBox.question(
            self, "Remove Face",
            f"Remove all encodings for \"{name}\"?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        data = load_known_faces()
        keep = [(n, e, d) for n, e, d in zip(data["names"], data["encodings"], data["door_unlock"]) if n != name]
        data["names"] = [n for n, _, _ in keep]
        data["encodings"] = [e for _, e, _ in keep]
        data["door_unlock"] = [d for _, _, d in keep]
        save_known_faces(data)
        # Also update the camera thread's in-memory copy
        if self.camera_thread:
            self.camera_thread.known_faces = load_known_faces()
        self.refresh_known_faces()
        self.log_event("enroll", f"Removed face: {name}")

    # =========================================================================
    # Settings
    # =========================================================================
    def apply_settings(self):
        if self.camera_thread:
            self.camera_thread.face_recognition_enabled = self.chk_face_recognition.isChecked()
            self.camera_thread.package_detection_enabled = self.chk_package_detection.isChecked()
            self.camera_thread.tolerance = self.confidence_spinner.value() / 100.0
        self.log_event("system", f"Settings applied (tolerance: {self.confidence_spinner.value()}%)")

        # Flash the button green to confirm
        self.btn_apply.setText("Settings Applied!")
        self._style_btn(self.btn_apply, GREEN)
        QTimer.singleShot(1500, self._reset_apply_btn)

    def _reset_apply_btn(self):
        self.btn_apply.setText("Apply Settings")
        self._style_btn(self.btn_apply, ACCENT)

    def _on_volume_changed(self, value):
        self.volume_label.setText(f"{value}%")
        self.voice_assistant.set_volume(value / 100.0)
        # Debounce amixer: only run once the slider stops moving
        if not hasattr(self, '_vol_timer'):
            self._vol_timer = QTimer()
            self._vol_timer.setSingleShot(True)
            self._vol_timer.timeout.connect(self._apply_hw_volume)
        self._vol_timer.start(300)

    def _apply_hw_volume(self):
        hw_vol = min(self.volume_slider.value(), 100)
        try:
            subprocess.Popen(
                ["amixer", "sset", "Master", f"{hw_vol}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            pass

    def _on_voice_changed(self, voice_name):
        sentinel_config.VOICE_VOICE = voice_name
        self._update_voice_info_label()
        self.log_event("system", f"Voice changed to: {voice_name}")
        # Restart session so the new voice takes effect
        if self.voice_assistant.is_active:
            self.stop_voice()
            self._wait_and_restart_voice()
        else:
            # Session not active — notify user the change is saved
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.voice_chat.append(
                f'<span style="color: {TEXT_MUTED};">[{timestamp}]</span> '
                f'<span style="color: {TEXT_DIM};">Voice set to {voice_name} — will apply on next session.</span>'
            )

    def _on_persona_changed(self, index):
        key = self.persona_combo.itemData(index)
        sentinel_config.VOICE_PERSONA = key
        self._update_voice_info_label()
        self.log_event("system", f"Persona set to: {self.persona_combo.currentText()}")
        # Restart session so AI starts fresh with new persona (no old history)
        if self.voice_assistant.is_active:
            self.stop_voice()
            self._wait_and_restart_voice()
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.voice_chat.append(
                f'<span style="color: {TEXT_MUTED};">[{timestamp}]</span> '
                f'<span style="color: {TEXT_DIM};">Persona set to {self.persona_combo.currentText()} — will apply on next session.</span>'
            )

    def _update_voice_info_label(self):
        self.voice_info_label.setText(
            f"Voice: {sentinel_config.VOICE_VOICE.capitalize()}  |  "
            f"Persona: {sentinel_config.VOICE_PERSONA.replace('_', ' ').title()}"
        )

    def _wait_and_restart_voice(self):
        """Poll until the old voice session is fully stopped, then start a new one."""
        if self.voice_assistant.is_active:
            QTimer.singleShot(200, self._wait_and_restart_voice)
        else:
            QTimer.singleShot(300, self.start_voice)

    def _test_voice(self):
        """Play a test phrase using OpenAI TTS API with the selected voice."""
        voice_name = self.voice_combo.currentText()
        phrase = f"Hi, I'm {voice_name}. I'll be your Sentinel voice."

        if not sentinel_config.OPENAI_API_KEY:
            self.log_event("error", "No OpenAI API key — set OPENAI_API_KEY env variable")
            return

        # Capture values on main thread before spawning background work
        vol = self.volume_slider.value() / 100.0
        api_key = sentinel_config.OPENAI_API_KEY
        speaker_dev = sentinel_config.VOICE_SPEAKER_DEVICE

        self.log_event("system", f"Testing voice: {voice_name}...")

        def _play_tts():
            try:
                import json as _json
                from urllib.request import Request, urlopen

                payload = _json.dumps({
                    "model": "tts-1",
                    "input": phrase,
                    "voice": voice_name,
                    "response_format": "pcm",
                }).encode("utf-8")

                req = Request(
                    "https://api.openai.com/v1/audio/speech",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                resp = urlopen(req, timeout=15)
                pcm_data = resp.read()

                audio = np.frombuffer(pcm_data, dtype=np.int16)
                audio = np.clip(
                    audio.astype(np.float32) * vol, -32768, 32767
                ).astype(np.int16)

                sd.play(audio, samplerate=24000, device=speaker_dev)
                sd.wait()
            except Exception as e:
                # Log error back on the main thread via signal
                QTimer.singleShot(0, lambda: self.log_event("error", f"Voice test failed: {e}"))

        import threading
        threading.Thread(target=_play_tts, daemon=True).start()

    def _test_persona(self):
        """Play a sample phrase in the selected persona's style."""
        persona_key = self.persona_combo.itemData(self.persona_combo.currentIndex())
        persona_phrases = {
            "default": "Yo what's good! Looking fresh today, not gonna lie.",
            "robot": "SCANNING. HUMAN DETECTED. THREAT LEVEL: MINIMAL. BEEP BOOP.",
            "pirate": "Arrr! Ahoy there matey! State yer business or walk the plank!",
            "british_butler": "Ah, splendid. One has arrived. Might I say, your attire is quite... a choice.",
            "drill_sergeant": "ATTENTION! IDENTIFY YOURSELF IMMEDIATELY! DROP AND GIVE ME TWENTY!",
            "professional": "Good afternoon. Welcome. How may I assist you today?",
            "nonchalant": "Oh. Hey. Sup.",
        }
        phrase = persona_phrases.get(persona_key, "Hello, testing persona.")

        if not sentinel_config.OPENAI_API_KEY:
            self.log_event("error", "No OpenAI API key — set OPENAI_API_KEY env variable")
            return

        voice_name = self.voice_combo.currentText()
        vol = self.volume_slider.value() / 100.0
        api_key = sentinel_config.OPENAI_API_KEY
        speaker_dev = sentinel_config.VOICE_SPEAKER_DEVICE

        self.log_event("system", f"Testing persona: {self.persona_combo.currentText()}...")

        def _play_tts():
            try:
                import json as _json
                from urllib.request import Request, urlopen

                payload = _json.dumps({
                    "model": "tts-1",
                    "input": phrase,
                    "voice": voice_name,
                    "response_format": "pcm",
                }).encode("utf-8")

                req = Request(
                    "https://api.openai.com/v1/audio/speech",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                resp = urlopen(req, timeout=15)
                pcm_data = resp.read()

                audio = np.frombuffer(pcm_data, dtype=np.int16)
                audio = np.clip(
                    audio.astype(np.float32) * vol, -32768, 32767
                ).astype(np.int16)

                sd.play(audio, samplerate=24000, device=speaker_dev)
                sd.wait()
            except Exception as e:
                QTimer.singleShot(0, lambda: self.log_event("error", f"Persona test failed: {e}"))

        import threading
        threading.Thread(target=_play_tts, daemon=True).start()

    def _toggle_mic_mute(self):
        if self.voice_assistant.is_mic_muted:
            self.voice_assistant.unmute_mic()
            self.btn_mute.setText("\U0001F3A4  Mute Mic")
            self.btn_mute.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BG_INPUT};
                    color: {TEXT_DIM};
                    border: 1px solid {BORDER};
                    border-radius: 8px;
                    padding: 4px 10px;
                    font-size: 11px;
                    outline: none;
                }}
                QPushButton:hover {{ background-color: {BORDER}; color: {TEXT}; }}
                QPushButton:focus {{ outline: none; border: 1px solid {BORDER}; }}
            """)
        else:
            self.voice_assistant.mute_mic()
            self.btn_mute.setText("\U0001F507  Unmute")
            self.btn_mute.setStyleSheet(f"""
                QPushButton {{
                    background-color: {RED};
                    color: white;
                    border: 1px solid {RED};
                    border-radius: 8px;
                    padding: 4px 10px;
                    font-size: 11px;
                    outline: none;
                }}
                QPushButton:hover {{ background-color: {RED_DIM}; }}
                QPushButton:focus {{ outline: none; border: 1px solid {RED}; }}
            """)

    # =========================================================================
    # Voice Assistant
    # =========================================================================
    def toggle_voice(self):
        if self.voice_assistant.is_active:
            self.stop_voice()
        else:
            self.start_voice()

    def start_voice(self):
        self._update_voice_context()
        self.voice_assistant.start_session()
        self.btn_voice.setText("Stop Voice Session")
        self._style_btn(self.btn_voice, RED)
        self.voice_status_dot.setStyleSheet(f"background-color: {PURPLE}; border-radius: 4px; border: none;")
        self.voice_status_label.setText("Starting...")
        self.voice_status_label.setStyleSheet(f"color: {PURPLE}; background: transparent; border: none; font-size: 11px;")
        self.home_voice_status.setText("Active")
        self.log_event("system", "Voice session starting")

    def stop_voice(self):
        self.voice_assistant.stop_session()
        self.btn_voice.setText("Start Voice Session")
        self._style_btn(self.btn_voice, PURPLE)
        self.voice_status_dot.setStyleSheet(f"background-color: {TEXT_MUTED}; border-radius: 4px; border: none;")
        self.voice_status_label.setText("Inactive")
        self.voice_status_label.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none; font-size: 11px;")
        self.home_voice_status.setText("Inactive")
        self.log_event("system", "Voice session stopped")

    def _update_voice_context(self):
        """Push face detection info to voice assistant with debouncing and auto-greet."""
        if not self.camera_thread:
            return

        now = time.time()

        # Snapshot shared data to avoid race with CameraThread
        faces = list(self.camera_thread.last_faces)
        raw_frame = self.camera_thread.last_raw_frame

        if faces:
            face = faces[0]  # primary face
            name = face[4]

            if name != "Unknown":
                self._voice_last_known_time = now
                current_person = name
                person_type = "known"
                self.voice_assistant.update_context(person="known", name=name)
            else:
                # Debounce: if we saw a known face very recently, ignore the unknown blip
                if now - self._voice_last_known_time < sentinel_config.VOICE_UNKNOWN_DEBOUNCE:
                    if self.voice_assistant.is_active:
                        self.voice_assistant.reset_silence_timeout()
                    return
                current_person = "unknown"
                person_type = "unknown"
                self.voice_assistant.update_context(person="unknown", name="unidentified")

            # Send frame for vision analysis (respects internal cooldown)
            if raw_frame is not None:
                self.voice_assistant.update_visual_context(
                    raw_frame,
                    person_name=name if name != "Unknown" else "unknown",
                )

            # Track stable identity (debounce rapid changes)
            if current_person != self._voice_stable_name:
                self._voice_stable_name = current_person
                self._voice_stable_since = now

            stable_for = now - self._voice_stable_since

            # Auto-greet after face is stable for 1.5 seconds
            # Skip if we just enrolled someone (camera needs time to catch up)
            if stable_for >= 1.5 and now > self._voice_enroll_suppress_until:
                should_greet = (
                    current_person != self._voice_greeted_name
                    or now - self._voice_greeted_time > sentinel_config.VOICE_GREET_COOLDOWN
                )
                if should_greet:
                    greet_name = current_person if person_type == "known" else "unidentified"
                    # Save unknown encoding now so auto-enroll can use it later
                    if person_type == "unknown":
                        for f in faces:
                            if f[4] == "Unknown":
                                self._pending_unknown_encoding = f[6]
                                break
                    if self.voice_assistant.is_active:
                        self._voice_greeted_name = current_person
                        self._voice_greeted_time = now
                        self.voice_assistant.trigger_greeting(person_type, greet_name)
                    elif sentinel_config.VOICE_AUTO_ENGAGE:
                        self._voice_pending_greet = (person_type, greet_name)
                        self.start_voice()

            # Keep session alive while a face is visible
            if self.voice_assistant.is_active:
                self.voice_assistant.reset_silence_timeout()
        else:
            self._voice_stable_name = None
            self.voice_assistant.update_context()

    @pyqtSlot(str)
    def _on_voice_transcript(self, text):
        """Display final AI response text in the chat area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.voice_chat.append(
            f'<span style="color: {TEXT_MUTED};">[{timestamp}]</span> '
            f'<span style="color: {PURPLE}; font-weight: bold;">Sentinel:</span> '
            f'<span style="color: {TEXT};">{text}</span>'
        )
        scrollbar = self.voice_chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(str)
    def _on_user_transcript(self, text):
        """Display user's speech in the chat area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.voice_chat.append(
            f'<span style="color: {TEXT_MUTED};">[{timestamp}]</span> '
            f'<span style="color: {ACCENT}; font-weight: bold;">User:</span> '
            f'<span style="color: {TEXT};">{text}</span>'
        )
        scrollbar = self.voice_chat.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(str)
    def _on_voice_status(self, status):
        """Update voice status indicator."""
        self.voice_status_label.setText(status)
        if "error" in status.lower() or "ended" in status.lower() or "timeout" in status.lower():
            self.voice_status_dot.setStyleSheet(f"background-color: {TEXT_MUTED}; border-radius: 4px; border: none;")
            self.voice_status_label.setStyleSheet(f"color: {TEXT_MUTED}; background: transparent; border: none; font-size: 11px;")
            self.btn_voice.setText("Start Voice Session")
            self._style_btn(self.btn_voice, PURPLE)
        elif "listening" in status.lower() or "connected" in status.lower():
            self.voice_status_dot.setStyleSheet(f"background-color: {GREEN}; border-radius: 4px; border: none;")
            self.voice_status_label.setStyleSheet(f"color: {GREEN}; background: transparent; border: none;")
            # Send pending greeting after connection is ready
            if self._voice_pending_greet:
                person_type, greet_name = self._voice_pending_greet
                self._voice_pending_greet = None
                self._voice_greeted_name = greet_name if person_type == "known" else "unknown"
                self._voice_greeted_time = time.time()
                QTimer.singleShot(
                    500,
                    lambda pt=person_type, gn=greet_name:
                        self.voice_assistant.trigger_greeting(pt, gn),
                )
        elif "processing" in status.lower():
            self.voice_status_dot.setStyleSheet(f"background-color: {ORANGE}; border-radius: 4px; border: none;")
            self.voice_status_label.setStyleSheet(f"color: {ORANGE}; background: transparent; border: none; font-size: 11px;")
        self.log_event("system", f"Voice: {status}")

    # =========================================================================
    # Voice Auto-Enrollment
    # =========================================================================
    def _do_voice_enroll(self, name):
        """Called from voice assistant thread. Saves face encoding, returns result string."""
        # Try to grab the unknown face encoding from the current frame
        encoding = None
        if self.camera_thread:
            faces = list(self.camera_thread.last_faces)
            for face in faces:
                if face[4] == "Unknown":
                    encoding = face[6]
                    break

        # Fall back to the encoding we saved when we first greeted this person
        if encoding is None and self._pending_unknown_encoding is not None:
            encoding = self._pending_unknown_encoding

        if encoding is None:
            return "No face visible. Ask them to look at the camera and tell you their name again."

        # Save to known faces
        data = load_known_faces()
        data["names"].append(name)
        data["encodings"].append(encoding)
        data["door_unlock"].append(False)
        save_known_faces(data)

        # Update camera thread's in-memory copy
        self.camera_thread.known_faces = load_known_faces()

        # Clear the saved encoding
        self._pending_unknown_encoding = None

        # Update voice context immediately so the AI knows this person is now registered
        self.voice_assistant.update_context(person="known", name=name)

        # Suppress ALL greetings for 10s while camera transitions from Unknown → name
        self._voice_greeted_name = name
        self._voice_greeted_time = time.time()
        self._voice_enroll_suppress_until = time.time() + 10

        # Signal UI update on Qt thread
        self.voice_bridge.enroll_completed.emit(name)

        return f"Successfully registered {name}. You'll recognize them from now on."

    @pyqtSlot(str)
    def _on_voice_enroll_ui(self, name):
        """Called on Qt thread to update UI after voice enrollment."""
        self.refresh_known_faces()
        self.log_event("enroll", f"Auto-enrolled via voice: {name}")

    def _do_voice_remove(self, name):
        """Called from voice assistant thread. Removes a person's face, returns result string."""
        data = load_known_faces()
        # Find all entries matching this name (case-insensitive)
        indices = [i for i, n in enumerate(data["names"]) if n.lower() == name.lower()]
        if not indices:
            return f"No person named {name} found in the system."

        actual_name = data["names"][indices[0]]
        keep = [(n, e, d) for i, (n, e, d) in enumerate(
            zip(data["names"], data["encodings"], data["door_unlock"])
        ) if i not in indices]
        data["names"] = [n for n, _, _ in keep]
        data["encodings"] = [e for _, e, _ in keep]
        data["door_unlock"] = [d for _, _, d in keep]
        save_known_faces(data)

        if self.camera_thread:
            self.camera_thread.known_faces = load_known_faces()

        self.voice_assistant.update_context(person="unknown", name="unidentified")
        self.voice_bridge.enroll_completed.emit(actual_name)
        return f"Successfully removed {actual_name} from the system."

    def _do_voice_rename(self, old_name, new_name):
        """Called from voice assistant thread. Renames a person, returns result string."""
        data = load_known_faces()
        # Find all entries matching old name (case-insensitive)
        found = False
        for i, n in enumerate(data["names"]):
            if n.lower() == old_name.lower():
                data["names"][i] = new_name
                found = True

        if not found:
            return f"No person named {old_name} found in the system."

        save_known_faces(data)

        if self.camera_thread:
            self.camera_thread.known_faces = load_known_faces()

        self.voice_assistant.update_context(person="known", name=new_name)
        self.voice_bridge.enroll_completed.emit(new_name)
        return f"Successfully updated name from {old_name} to {new_name}."

    # =========================================================================
    # Theme
    # =========================================================================
    def apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {BG_MAIN}; }}
            QWidget {{ background-color: transparent; color: {TEXT}; }}
            QLabel {{ color: {TEXT}; }}
            QMessageBox {{
                background-color: {BG_CARD};
            }}
            QMessageBox QLabel {{
                color: {TEXT};
            }}
            QMessageBox QPushButton {{
                background-color: {ACCENT};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 20px;
                font-weight: 600;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{ background-color: {ACCENT_DIM}; }}
            QScrollBar:vertical {{
                background: transparent;
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER_LIGHT};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

    @staticmethod
    def _style_btn(btn, color, text_color="white"):
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {text_color};
                border: none;
                border-radius: 10px;
                padding: 6px 20px;
                font-weight: 600;
                outline: none;
            }}
            QPushButton:focus {{
                outline: none;
                border: none;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
            QPushButton:pressed {{ opacity: 0.7; }}
        """)

    @staticmethod
    def _group_style():
        return f"""
            QGroupBox {{
                color: {TEXT_DIM};
                border: 1px solid {BORDER};
                border-radius: 12px;
                margin-top: 14px;
                padding: 14px 12px 10px 12px;
                font-weight: 600;
                font-size: 11px;
                background-color: {BG_CARD_LIGHT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                padding: 0 8px;
                color: {TEXT_DIM};
            }}
        """

    @staticmethod
    def _combo_style():
        return f"""
            QComboBox {{
                padding: 6px 10px;
                background-color: {BG_INPUT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background-color: {BG_CARD};
                color: {TEXT};
                selection-background-color: {ACCENT};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """

    @staticmethod
    def _spin_style():
        return f"""
            QSpinBox {{
                padding: 6px;
                background-color: {BG_INPUT};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
        """

    @staticmethod
    def _check_style():
        return f"""
            QCheckBox {{
                color: {TEXT_DIM};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {BORDER_LIGHT};
                border-radius: 5px;
                background-color: {BG_CARD};
            }}
            QCheckBox::indicator:checked {{
                background-color: {GREEN};
                border-color: {GREEN};
            }}
        """

    def closeEvent(self, event):
        self.voice_assistant.stop_session()
        self.stop_camera()
        event.accept()


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sentinel AI")
    app.setFont(QFont("Helvetica Neue", 10))

    window = SentinelApp()

    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel_logo.png")
    if os.path.exists(logo_path):
        app.setWindowIcon(QIcon(logo_path))
        window.setWindowIcon(QIcon(logo_path))

    window.showMaximized()
    window.log_event("system", "Sentinel AI started")
    window.log_event("system", "Press Start to begin camera feed")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
