#!/bin/bash
# Sentinel AI - Setup Script
# Run this on your Raspberry Pi to install dependencies and create a desktop shortcut

echo "=== Sentinel AI Desktop App Setup ==="
echo ""

# Install PyQt5
echo "📦 Installing PyQt5..."
sudo apt-get update
sudo apt-get install -y python3-pyqt5

# Verify installation
echo ""
echo "✅ Verifying PyQt5 installation..."
python3 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  PyQt5 apt install failed, trying pip..."
    pip install PyQt5 --break-system-packages
fi

# Create desktop shortcut
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DESKTOP_FILE="$HOME/Desktop/sentinel-ai.desktop"

echo ""
echo "🖥️  Creating desktop shortcut..."
cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=Sentinel AI
Comment=Smart Security System
Exec=python3 ${APP_DIR}/sentinel_app.py
Icon=${APP_DIR}/sentinel_logo.png
Terminal=false
Type=Application
Categories=Security;Utility;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

# Also add to applications menu
MENU_FILE="$HOME/.local/share/applications/sentinel-ai.desktop"
mkdir -p "$(dirname "$MENU_FILE")"
cp "$DESKTOP_FILE" "$MENU_FILE"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "You can now:"
echo "  1. Double-click 'Sentinel AI' on your desktop"
echo "  2. Find it in your applications menu"
echo "  3. Or run: python3 ${APP_DIR}/sentinel_app.py"
echo ""
