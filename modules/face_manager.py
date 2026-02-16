# ============================================
# Sentinel AI - Face Manager Module
# ============================================

import cv2
import os
import pickle
import time
import numpy as np
import face_recognition
from datetime import datetime
import config


class FaceManager:
    """Handles face detection, recognition, enrollment, and the face database."""

    def __init__(self):
        self.known_encodings = []   # List of face encodings
        self.known_names = []       # Corresponding names
        self.recently_seen = {}     # {name: last_seen_timestamp}
        self.unknown_cooldown = {}  # {face_hash: last_prompted_timestamp}
        self.greeting_cooldown = 30  # Seconds before re-greeting same person

        # Load existing face database
        self.load_database()

    # ------------------------------------------
    # Database Management
    # ------------------------------------------

    def load_database(self):
        """Load known face encodings from disk."""
        if os.path.exists(config.ENCODINGS_FILE):
            with open(config.ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
            print(f"[FACES] Loaded {len(self.known_names)} known face(s): {set(self.known_names)}")
        else:
            print("[FACES] No existing face database found. Starting fresh.")

    def save_database(self):
        """Save known face encodings to disk."""
        data = {"encodings": self.known_encodings, "names": self.known_names}
        with open(config.ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"[FACES] Database saved. Total faces: {len(self.known_names)}")

    # ------------------------------------------
    # Face Detection & Recognition
    # ------------------------------------------

    def detect_and_recognize(self, frame):
        """
        Detect faces in a frame and try to recognize them.

        Returns:
            list of dicts: [{"name": str, "location": tuple, "is_known": bool}, ...]
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and compute encodings
        face_locations = face_recognition.face_locations(
            rgb_small, model=config.FACE_RECOGNITION_MODEL
        )
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        results = []
        for encoding, location in zip(face_encodings, face_locations):
            name = "Unknown"
            is_known = False

            if len(self.known_encodings) > 0:
                # Compare against known faces
                matches = face_recognition.compare_faces(
                    self.known_encodings, encoding,
                    tolerance=config.FACE_RECOGNITION_TOLERANCE
                )
                face_distances = face_recognition.face_distance(
                    self.known_encodings, encoding
                )

                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = self.known_names[best_match_idx]
                        is_known = True

            # Scale location back to full size
            top, right, bottom, left = location
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            results.append({
                "name": name,
                "location": (top, right, bottom, left),
                "is_known": is_known,
                "encoding": encoding
            })

        return results

    # ------------------------------------------
    # Greeting Logic
    # ------------------------------------------

    def should_greet(self, name):
        """Check if we should greet this person (cooldown-based)."""
        now = time.time()
        if name in self.recently_seen:
            elapsed = now - self.recently_seen[name]
            if elapsed < self.greeting_cooldown:
                return False
        self.recently_seen[name] = now
        return True

    def should_prompt_unknown(self, encoding):
        """Check if we should prompt this unknown face (cooldown-based)."""
        now = time.time()
        # Create a simple hash of the encoding for tracking
        face_hash = hash(encoding.tobytes())
        if face_hash in self.unknown_cooldown:
            elapsed = now - self.unknown_cooldown[face_hash]
            if elapsed < config.UNKNOWN_FACE_TIMEOUT:
                return False
        self.unknown_cooldown[face_hash] = now
        return True

    # ------------------------------------------
    # Face Enrollment
    # ------------------------------------------

    def enroll_face(self, frame, name):
        """
        Enroll a new face into the database.

        Args:
            frame: The current video frame (BGR)
            name: The person's name
        """
        # Create a directory for this person's face images
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name.lower().replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        # Save the face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(person_dir, f"{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[FACES] Saved face image: {img_path}")

        # Get encoding from the full frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=config.FACE_RECOGNITION_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) > 0:
            self.known_encodings.append(face_encodings[0])
            self.known_names.append(name)
            self.save_database()
            print(f"[FACES] {name} has been enrolled successfully!")
            return True
        else:
            print("[FACES] Could not detect a face in the frame for enrollment.")
            return False

    def enroll_face_interactive(self, frame, encoding, name):
        """
        Enroll using an already-detected encoding plus save the frame.

        Args:
            frame: The current video frame (BGR)
            encoding: The face encoding already computed
            name: The person's name
        """
        # Save face image
        person_dir = os.path.join(config.KNOWN_FACES_DIR, name.lower().replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(person_dir, f"{timestamp}.jpg")
        cv2.imwrite(img_path, frame)

        # Add encoding to database
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self.save_database()

        print(f"[FACES] ✓ {name} enrolled successfully! ({len(self.known_encodings)} total faces)")
        return True

    # ------------------------------------------
    # Drawing Utilities
    # ------------------------------------------

    @staticmethod
    def draw_face_boxes(frame, faces):
        """Draw bounding boxes and names on the frame."""
        for face in faces:
            top, right, bottom, left = face["location"]
            name = face["name"]
            is_known = face["is_known"]

            # Green for known, red for unknown
            color = (0, 200, 0) if is_known else (0, 0, 200)

            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw name label
            cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
            cv2.putText(
                frame, name, (left + 6, bottom - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

        return frame
