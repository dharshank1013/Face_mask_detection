"""Main entry point for Face Mask Detection Desktop Application.

Run with:
    python app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    # Ensure Windows high DPI scaling is enabled
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("Face Mask Detection System")
    app.setOrganizationName("FOCV")

    # Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
