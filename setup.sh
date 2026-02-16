#!/bin/bash
# ============================================
# Sentinel AI - Raspberry Pi Setup Script
# ============================================

echo "=========================================="
echo "  Sentinel AI - Installation Starting"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    cmake \
    build-essential \
    libatlas-base-dev \
    libhdf5-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libboost-all-dev \
    libgtk-3-dev \
    libportaudio2 \
    espeak \
    2>/dev/null

# Create virtual environment
echo "[3/6] Creating Python virtual environment..."
cd ~/sentinel_ai
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install Python packages
echo "[4/6] Installing Python packages..."
pip install --upgrade pip
pip install \
    numpy \
    pillow \
    imutils \
    picamera2 \
    websockets \
    sounddevice \
    2>/dev/null

# Install dlib (this takes a while on Pi!)
echo "[5/6] Installing dlib (this may take 15-30 min on Pi)..."
pip install dlib

# Install face_recognition
echo "[6/6] Installing face_recognition..."
pip install face_recognition

echo ""
echo "=========================================="
echo "  Sentinel AI - Installation Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  cd ~/sentinel_ai && source venv/bin/activate"
echo ""
echo "To run Sentinel AI:"
echo "  python3 main.py"
echo ""
