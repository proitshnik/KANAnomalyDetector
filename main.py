"""
Приложение Anomaly Detector с GUI.
Точка входа.
"""
# ===============================
# Fix for WinError 1114 with _load_dll_libraries c10.dll
# ===============================
import os
import platform
import ctypes
from importlib.util import find_spec

if platform.system() == "Windows":
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass
# ===============================

import sys
from pathlib import Path

from app_logging import configure_app_logging, get_logger
from version import PROGRAM_VERSION

configure_app_logging(log_file=Path(__file__).resolve().parent / "logs" / "app.log")

logger = get_logger("main")
logger.info("Запуск приложения (до main): Anomaly Detector %s", PROGRAM_VERSION)

from PyQt6.QtWidgets import QApplication, QMainWindow

from app_logging import attach_gui_log_handler
from app_window import MainWindow


def main():
    app = QApplication(sys.argv)
    shell = QMainWindow()
    window = MainWindow()
    shell.setCentralWidget(window)
    shell.setWindowTitle(window.windowTitle())
    shell.resize(window.minimumSize())

    attach_gui_log_handler(window.log_gui_message.emit)
    logger.info("Запуск приложения (до main): Anomaly Detector %s", PROGRAM_VERSION)

    shell.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
