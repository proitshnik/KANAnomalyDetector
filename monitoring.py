import time

from app_logging import get_logger, init_default_logging_if_needed

init_default_logging_if_needed()
_log = get_logger("monitoring")


def monitor_data(monitor_file, input_length=288, output_length=10, interval_sec=5,
                callback=None, stop_check=None):
    """
    Мониторит файл в реальном времени и обрабатывает данные через callback.
    На первом шаге input_length реальных значений (и output_length предсказаний).
    На каждом следующем добавляем 1 реальное значение и 1 предсказание.
    callback работает отдельно с моделью и принимает список float значений и режим "init" - для инициализации или "step" - для обновления.
    """
    last_len = 0
    is_first_run = True

    while True:
        if stop_check and stop_check():
            break

        try:
            with open(monitor_file, "r") as f:
                lines = f.readlines()
            values = [float(line.strip()) for line in lines if line.strip()]
            current_len = len(values)

            # Первый шаг
            if is_first_run and current_len >= input_length:
                if callback:
                    callback(values, mode="init")

                last_len = current_len
                is_first_run = False

            # Новые значения
            elif (not is_first_run) and current_len > last_len:
                if callback:
                    callback(values, mode="step")

                last_len = current_len

        except Exception:
            # Ошибки чтения/парсинга файла не останавливают мониторинг, а просто пропускаются
            _log.warning("Ошибка чтения, пропуск значения")
            pass

        time.sleep(interval_sec)
