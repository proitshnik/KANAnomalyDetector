"""
Модуль для управления путями мониторинга и источника данных встроенной имитации в режиме реального времени
"""

# Значения по умолчанию
DEFAULT_MONITOR_REALTIME_FILE_PATH = "realtime_monitoring.txt"
DEFAULT_SOURCE_REALTIME_FILE_PATH = "input/august2018_5.txt"

_monitor_realtime_file_path = DEFAULT_MONITOR_REALTIME_FILE_PATH
_source_realtime_file_path = DEFAULT_SOURCE_REALTIME_FILE_PATH
_use_imitation = True


def get_monitor_path():
    """Получить текущий путь файла мониторинга"""
    return _monitor_realtime_file_path


def get_source_path():
    """Получить текущий путь файла источника данных"""
    return _source_realtime_file_path

def get_paths():
    """Получить оба пути: путь мониторинга и путь источника данных (кортеж)"""
    return (_monitor_realtime_file_path, _source_realtime_file_path)


def get_imitation_enabled():
    """Проверить, включена ли имитация"""
    return _use_imitation


def set_paths(monitor_path, source_path):
    """Установить пути для обоих файлов"""
    global _monitor_realtime_file_path, _source_realtime_file_path
    _monitor_realtime_file_path = monitor_path
    _source_realtime_file_path = source_path


def set_monitor_path(path):
    """Установить путь файла мониторинга"""
    global _monitor_realtime_file_path
    _monitor_realtime_file_path = path


def set_source_path(path):
    """Установить путь файла источника данных"""
    global _source_realtime_file_path
    _source_realtime_file_path = path


def set_imitation_enabled(enabled):
    """Включить/отключить имитацию"""
    global _use_imitation
    _use_imitation = enabled


def reset_to_defaults():
    """Сбросить пути на значения по умолчанию"""
    global _monitor_realtime_file_path, _source_realtime_file_path, _use_imitation
    _monitor_realtime_file_path = DEFAULT_MONITOR_REALTIME_FILE_PATH
    _source_realtime_file_path = DEFAULT_SOURCE_REALTIME_FILE_PATH
    _use_imitation = True
