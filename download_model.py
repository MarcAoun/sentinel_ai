#!/usr/bin/env python3
# ============================================
# Sentinel AI - Download Object Detection Model
# ============================================

import urllib.request
import os
import config

MODEL_FILES = {
    "MobileNetSSD_deploy.prototxt": (
        "https://raw.githubusercontent.com/chuanqi305/"
        "MobileNet-SSD/master/deploy.prototxt"
    ),
    "MobileNetSSD_deploy.caffemodel": (
        "https://github.com/chuanqi305/MobileNet-SSD/"
        "raw/master/mobilenet_iter_73000.caffemodel"
    ),
}


def download_models():
    """Download MobileNet SSD model files."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        filepath = os.path.join(config.MODELS_DIR, filename)

        if os.path.exists(filepath):
            print(f"[✓] {filename} already exists.")
            continue

        print(f"[↓] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[✓] {filename} downloaded ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[✗] Failed to download {filename}: {e}")
            print(f"    URL: {url}")
            print(f"    Try downloading manually and placing in: {config.MODELS_DIR}")

    print("\nModel download complete!")


if __name__ == "__main__":
    download_models()
