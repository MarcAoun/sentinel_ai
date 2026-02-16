# ============================================
# Sentinel AI - Camera Module
# ============================================

import cv2
import sys


class Camera:
    """Handles camera input from Pi Camera or USB webcam."""

    def __init__(self, resolution=(640, 480), use_picamera=False):
        self.resolution = resolution
        self.use_picamera = use_picamera
        self.camera = None

        if use_picamera:
            try:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                print("[CAMERA] Pi Camera initialized.")
            except Exception as e:
                print(f"[CAMERA] Pi Camera failed: {e}")
                print("[CAMERA] Falling back to USB camera...")
                self.use_picamera = False

        if not self.use_picamera:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            if not self.camera.isOpened():
                print("[CAMERA] ERROR: Could not open any camera!")
                sys.exit(1)
            print("[CAMERA] USB Camera initialized.")

    def read_frame(self):
        """Returns a single frame as a BGR numpy array."""
        if self.use_picamera:
            frame = self.camera.capture_array()
            # picamera2 gives RGB, OpenCV uses BGR
            return True, frame
        else:
            return self.camera.read()

    def release(self):
        """Release camera resources."""
        if self.use_picamera:
            self.camera.stop()
        else:
            self.camera.release()
        print("[CAMERA] Camera released.")
