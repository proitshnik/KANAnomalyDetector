"""
Единая система логирования для приложения с GUI и без
Поддерживает уровни логирования: DEBUG, INFO, ANOMALY (26, log_anomaly), WARNING, ERROR, CRITICAL
Режим приложения: файл logs/app.log + консоль stderr + виджет в GUI
Режим standalone (core/*_standalone вне GUI): только stderr
"""

import logging
import os
import sys
from pathlib import Path

ROOT_LOGGER_NAME = "anomaly_detector"

ANOMALY = 26
logging.addLevelName(ANOMALY, "ANOMALY")

_gui_handler = None


def get_logger(name):
    """Получение логгера для модуля, причем добавляется имя модуля"""
    full_name = f"{ROOT_LOGGER_NAME}.{name}" if name else ROOT_LOGGER_NAME
    return logging.getLogger(full_name)


def log_anomaly(logger, msg):
    """Логирование аномалии с уровнем ANOMALY, между INFO и WARNING"""
    logger.log(ANOMALY, msg)


def init_default_logging_if_needed():
    """Инициализация стандартного логирования standalone, если оно не настроено"""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not root.handlers:
        configure_standalone_logging()


def configure_standalone_logging(level=logging.DEBUG):
    """Режим standalone, где только stderr, без файла, GUI не подключается"""
    os.environ["ANOMALY_LOG_STANDALONE"] = "1"
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    root.propagate = False


def configure_app_logging(log_file, console=True, console_level=logging.INFO, file_level=logging.DEBUG):
    """Режим приложения, где логирование ведется в файл и опционально в консоль, а GUI подключается отдельно через attach_gui_log_handler()"""
    
    # Если был standalone-режим, отключаем его
    os.environ.pop("ANOMALY_LOG_STANDALONE", None)
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level)
        ch.setFormatter(fmt)
        root.addHandler(ch)


class _GuiLogHandler(logging.Handler):
    """Внутренний логгер для отправки сообщений в GUI через thread-safe сигнал"""

    def __init__(self, sink):
        super().__init__()
        self._sink = sink
        self.setLevel(logging.DEBUG)
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    def emit(self, record):
        try:
            msg = self.format(record)
            if not msg.endswith("\n"):
                msg += "\n"
            self._sink(msg)
        except Exception:
            self.handleError(record)


def attach_gui_log_handler(sink):
    """Подключение GUI-логгера, который будет отправлять сообщения в виджет через переданный sink"""
    
    global _gui_handler
    root = logging.getLogger(ROOT_LOGGER_NAME)
    # Удаляем предыдущий GUI-обработчик, если он был
    if _gui_handler is not None:
        root.removeHandler(_gui_handler)
        _gui_handler = None
    _gui_handler = _GuiLogHandler(sink)
    root.addHandler(_gui_handler)
