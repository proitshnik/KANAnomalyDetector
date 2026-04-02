"""
Приложение Anomaly Detector с GUI.
Точка входа.
"""

import sys
from pathlib import Path

from app_logging import configure_app_logging, get_logger
from version import PROGRAM_VERSION

configure_app_logging(log_file=Path(__file__).resolve().parent / "logs" / "app.log")

logger = get_logger("main")
logger.info("Запуск приложения: Anomaly Detector %s", PROGRAM_VERSION)

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

    shell.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
