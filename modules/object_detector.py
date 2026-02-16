# ============================================
# Sentinel AI - Object Detection Module
# ============================================

import cv2
import numpy as np
import os
import time
import config


# MobileNet SSD classes
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Objects that could indicate a package/delivery
PACKAGE_INDICATORS = {"backpack", "suitcase", "handbag"}
# Note: MobileNet SSD doesn't have a "package" class directly,
# so we also use shape/size heuristics alongside detection


class ObjectDetector:
    """Detects objects in frames using MobileNet SSD (lightweight for Pi)."""

    def __init__(self):
        self.net = None
        self.available = False
        self.package_cooldown = 0
        self.package_cooldown_time = 30  # Seconds between package notifications

        self._load_model()

    def _load_model(self):
        """Load the MobileNet SSD model."""
        prototxt = os.path.join(config.MODELS_DIR, "MobileNetSSD_deploy.prototxt")
        model = os.path.join(config.MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

        if os.path.exists(prototxt) and os.path.exists(model):
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
            self.available = True
            print("[OBJECTS] MobileNet SSD model loaded.")
        else:
            print("[OBJECTS] Model files not found. Object detection disabled.")
            print(f"[OBJECTS] Expected files in: {config.MODELS_DIR}")
            print("[OBJECTS] Run: python3 download_model.py  to download them.")
            self.available = False

    def detect(self, frame):
        """
        Detect objects in a frame.

        Returns:
            list of dicts: [{"label": str, "confidence": float, "box": tuple}, ...]
        """
        if not self.available:
            return []

        h, w = frame.shape[:2]

        # Prepare the frame for MobileNet SSD
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > config.CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                if class_id < len(CLASSES):
                    label = CLASSES[class_id]
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")

                    results.append({
                        "label": label,
                        "confidence": float(confidence),
                        "box": (x1, y1, x2, y2)
                    })

        return results

    def check_for_package(self, detections):
        """
        Check if any detected objects look like a package.

        Returns:
            bool: True if a package-like object is detected
        """
        now = time.time()
        if now - self.package_cooldown < self.package_cooldown_time:
            return False

        for det in detections:
            label = det["label"]
            # Check for package-like objects
            if label in PACKAGE_INDICATORS:
                self.package_cooldown = now
                return True

            # Heuristic: a non-person rectangular object near the ground
            # could be a package (basic heuristic)
            if label == "bottle" or label == "tvmonitor":
                x1, y1, x2, y2 = det["box"]
                aspect_ratio = (x2 - x1) / max((y2 - y1), 1)
                if 0.5 < aspect_ratio < 2.0:  # Roughly box-shaped
                    self.package_cooldown = now
                    return True

        return False

    @staticmethod
    def draw_detections(frame, detections):
        """Draw detection boxes and labels on frame."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            confidence = det["confidence"]

            color = (255, 165, 0)  # Orange
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label}: {confidence:.1%}"
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        return frame
