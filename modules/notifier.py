# ============================================
# Sentinel AI - Notification Module
# ============================================

import random
import time
from datetime import datetime
import config


class Notifier:
    """Handles all notifications, greetings, and alerts."""

    def __init__(self):
        self.log = []

    def _timestamp(self):
        """Get a formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def _log_event(self, event_type, message):
        """Log an event for history."""
        entry = {
            "time": self._timestamp(),
            "type": event_type,
            "message": message
        }
        self.log.append(entry)
        return entry

    # ------------------------------------------
    # Greetings
    # ------------------------------------------

    def greet_known_person(self, name):
        """Greet a recognized person."""
        greeting = random.choice(config.GREETINGS).format(name=name)
        print(f"\n{'='*50}")
        print(f"  🟢 [{self._timestamp()}] {greeting}")
        print(f"{'='*50}\n")
        self._log_event("greeting", greeting)
        return greeting

    def prompt_unknown_person(self):
        """Friendly prompt for an unknown person."""
        message = (
            f"\n{'='*50}\n"
            f"  👋 [{self._timestamp()}] Hey there! I don't think we've met.\n"
            f"     I'm Sentinel AI. What's your name?\n"
            f"     (Type their name in the terminal when ready)\n"
            f"{'='*50}\n"
        )
        print(message)
        self._log_event("unknown_prompt", "Prompted unknown person for name")

    # ------------------------------------------
    # Door System
    # ------------------------------------------

    def door_unlock(self, name):
        """Simulate unlocking the door."""
        message = config.DOOR_UNLOCK_MESSAGE.format(name=name)
        print(f"\n{'='*50}")
        print(f"  🔓 [{self._timestamp()}] {message}")
        print(f"{'='*50}\n")
        self._log_event("door_unlock", message)

    def door_lock(self):
        """Simulate locking the door."""
        message = config.DOOR_LOCK_MESSAGE
        print(f"  🔒 [{self._timestamp()}] {message}")
        self._log_event("door_lock", message)

    # ------------------------------------------
    # Package Detection
    # ------------------------------------------

    def package_detected(self):
        """Notify about a package detection."""
        message = config.PACKAGE_DETECTED_MESSAGE
        print(f"\n{'='*50}")
        print(f"  📦 [{self._timestamp()}] {message}")
        print(f"{'='*50}\n")
        self._log_event("package", message)

    # ------------------------------------------
    # Unknown Person Alert
    # ------------------------------------------

    def unknown_person_alert(self):
        """Alert about an unknown person."""
        message = config.UNKNOWN_PERSON_MESSAGE
        print(f"  ⚠️  [{self._timestamp()}] {message}")
        self._log_event("unknown_alert", message)

    # ------------------------------------------
    # General Status
    # ------------------------------------------

    def status(self, message):
        """Print a general status message."""
        print(f"  ℹ️  [{self._timestamp()}] {message}")
        self._log_event("status", message)

    def show_log(self, last_n=10):
        """Display recent event log."""
        print(f"\n--- Last {last_n} Events ---")
        for entry in self.log[-last_n:]:
            print(f"  [{entry['time']}] ({entry['type']}) {entry['message']}")
        print("---\n")
