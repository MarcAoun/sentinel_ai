#!/usr/bin/env python3
# ============================================
#
#   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗
#   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║
#   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║
#   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║
#   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
#   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
#                        A I   S E C U R I T Y
#
# ============================================

import cv2
import sys
import time
import threading
import config
from modules.camera import Camera
from modules.face_manager import FaceManager
from modules.object_detector import ObjectDetector
from modules.notifier import Notifier


class SentinelAI:
    """Main Sentinel AI application."""

    def __init__(self, use_picamera=False):
        print("\n" + "=" * 55)
        print("  SENTINEL AI - Initializing...")
        print("=" * 55 + "\n")

        # Initialize modules
        self.camera = Camera(
            resolution=config.CAMERA_RESOLUTION,
            use_picamera=use_picamera
        )
        self.face_manager = FaceManager()
        self.object_detector = ObjectDetector()
        self.notifier = Notifier()

        # State tracking
        self.frame_count = 0
        self.running = True
        self.enrollment_mode = False
        self.pending_enrollment_encoding = None
        self.pending_enrollment_frame = None
        self.waiting_for_name = False
        self.typed_name = ""
        self.last_faces = []

        print("\n" + "=" * 55)
        print("  SENTINEL AI - Ready!")
        print("  Press 'q' to quit | 'e' to enroll | 'l' for log")
        print("=" * 55 + "\n")

    def process_faces(self, frame, faces):
        """Handle all face-related logic."""
        for face in faces:
            if face["is_known"]:
                # --- Known Person ---
                name = face["name"]
                if self.face_manager.should_greet(name):
                    self.notifier.greet_known_person(name)
                    self.notifier.door_unlock(name)

            else:
                # --- Unknown Person ---
                if self.face_manager.should_prompt_unknown(face["encoding"]):
                    self.notifier.unknown_person_alert()
                    self.notifier.prompt_unknown_person()

                    # Store encoding and frame for potential enrollment
                    self.pending_enrollment_encoding = face["encoding"]
                    self.pending_enrollment_frame = frame.copy()
                    self.enrollment_mode = True

    def process_objects(self, frame):
        """Handle object detection logic."""
        detections = self.object_detector.detect(frame)

        if detections:
            # Check for packages
            if self.object_detector.check_for_package(detections):
                self.notifier.package_detected()

            # Draw detection boxes
            self.object_detector.draw_detections(frame, detections)

        return detections

    def draw_enrollment_ui(self, frame):
        """Draw the enrollment name input on the video frame."""
        if self.waiting_for_name:
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Input box
            box_y = frame.shape[0] // 2 - 40
            cv2.rectangle(frame, (50, box_y), (frame.shape[1] - 50, box_y + 80), (40, 40, 40), -1)
            cv2.rectangle(frame, (50, box_y), (frame.shape[1] - 50, box_y + 80), (0, 200, 0), 2)

            # Prompt text
            cv2.putText(
                frame, "Enter name (type + ENTER):", (70, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )

            # Typed name with cursor
            display_name = self.typed_name + "|"
            cv2.putText(
                frame, display_name, (70, box_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        return frame

    def handle_keypress(self, key, frame):
        """Handle keyboard input."""
        if key == 255 or key == -1:
            return

        # --- Name typing mode ---
        if self.waiting_for_name:
            if key == 13 or key == 10:  # ENTER
                name = self.typed_name.strip()
                if name:
                    self.face_manager.enroll_face_interactive(
                        self.pending_enrollment_frame,
                        self.pending_enrollment_encoding,
                        name
                    )
                    self.notifier.status(f"Welcome aboard, {name}!")
                    self.enrollment_mode = False
                    self.pending_enrollment_encoding = None
                    self.pending_enrollment_frame = None
                    self.waiting_for_name = False
                    self.typed_name = ""
                else:
                    print("[ENROLL] Name can't be empty. Try again.")

            elif key == 27:  # ESC - cancel enrollment
                self.waiting_for_name = False
                self.typed_name = ""
                self.enrollment_mode = False
                print("[ENROLL] Enrollment cancelled.")

            elif key == 8 or key == 127:  # BACKSPACE
                self.typed_name = self.typed_name[:-1]

            elif 32 <= key <= 126:  # Printable characters
                self.typed_name += chr(key)

            return

        # --- Normal mode ---
        if key == ord('q'):
            self.running = False

        elif key == ord('e'):
            if self.pending_enrollment_encoding is not None:
                self.waiting_for_name = True
                self.typed_name = ""
                print("\n[ENROLL] Type the person's name on the video window and press ENTER.")
                print("[ENROLL] Press ESC to cancel.\n")
            else:
                print("[ENROLL] No unknown face detected to enroll. Show a face first!")

        elif key == ord('l'):
            self.notifier.show_log()

        elif key == ord('s'):
            print(f"\n[STATUS] Known faces: {len(set(self.face_manager.known_names))}")
            print(f"[STATUS] Names: {set(self.face_manager.known_names)}")
            print(f"[STATUS] Object detection: {'Active' if self.object_detector.available else 'Disabled'}")

    def run(self):
        """Main application loop."""
        try:
            while self.running:
                # Read frame from camera
                ret, frame = self.camera.read_frame()
                if not ret:
                    print("[ERROR] Failed to read from camera!")
                    break

                self.frame_count += 1

                # --- Process every Nth frame to save CPU ---
                if self.frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    # Face detection & recognition
                    faces = self.face_manager.detect_and_recognize(frame)
                    self.last_faces = faces

                    if faces:
                        self.process_faces(frame, faces)

                    # Object detection
                    self.process_objects(frame)

                # Draw face boxes (using last known face positions)
                if self.last_faces:
                    self.face_manager.draw_face_boxes(frame, self.last_faces)

                # --- Draw enrollment UI if active ---
                if self.waiting_for_name:
                    frame = self.draw_enrollment_ui(frame)

                # --- Status bar ---
                status_text = "SENTINEL AI | "
                status_text += f"Faces: {len(set(self.face_manager.known_names))} known | "
                status_text += f"Objects: {'ON' if self.object_detector.available else 'OFF'} | "
                if self.enrollment_mode and not self.waiting_for_name:
                    status_text += "Press 'e' to enroll unknown face"
                else:
                    status_text += "q=Quit e=Enroll l=Log s=Status"

                cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
                cv2.putText(
                    frame, status_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1
                )

                # Enrollment alert bar
                if self.enrollment_mode and not self.waiting_for_name:
                    cv2.rectangle(frame, (0, 30), (frame.shape[1], 60), (0, 0, 180), -1)
                    cv2.putText(
                        frame, "UNKNOWN FACE DETECTED - Press 'e' to enroll",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )

                cv2.imshow("Sentinel AI", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keypress(key, frame)

        except KeyboardInterrupt:
            print("\n[SENTINEL] Interrupted by user.")

        finally:
            self.shutdown()

    def shutdown(self):
        """Clean up and exit."""
        print("\n" + "=" * 55)
        print("  SENTINEL AI - Shutting down...")
        print("=" * 55)
        self.camera.release()
        cv2.destroyAllWindows()
        self.face_manager.save_database()
        print("  Goodbye!\n")


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentinel AI - Smart Security System")
    parser.add_argument(
        "--picamera", action="store_true",
        help="Use Raspberry Pi Camera instead of USB webcam"
    )
    parser.add_argument(
        "--no-objects", action="store_true",
        help="Disable object detection (faster on Pi)"
    )
    args = parser.parse_args()

    sentinel = SentinelAI(use_picamera=args.picamera)

    if args.no_objects:
        sentinel.object_detector.available = False
        print("[CONFIG] Object detection disabled via command line.")

    sentinel.run()
