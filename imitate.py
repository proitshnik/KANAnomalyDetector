import time
import os

from app_logging import get_logger, init_default_logging_if_needed

init_default_logging_if_needed()
_log = get_logger("imitate")


def imitate_data(source_file, target_file, input_length=288, interval_sec=300, reset_file=False, stop_check=None):
    """
    Имитация поступления данных, сначала записать input_length значений, затем по одному с интервалом interval_sec. Если reset_file=True, файл мониторинга пересоздается.
    """
    
    with open(source_file, 'r') as f:
        lines = f.readlines()
        
    values = []
    for line in lines:
        values.extend([float(x) for x in line.strip().split()])
        
    total = len(values)
    
    # Создать директорию, если необходимо
    target_dir = os.path.dirname(target_file)
    if target_dir and not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    # Пересоздать файл, если требуется
    mode = 'w' if reset_file else 'a'
    with open(target_file, mode) as f:
        for i in range(input_length):
            f.write(f"{values[i]}\n")
            
    idx = input_length
    while idx < total:
        # Даем возможность корректно остановить имитацию
        if stop_check and stop_check():
            break

        # Спим с шагами для реакции на остановку
        slept = 0.0
        step = 0.5
        while slept < interval_sec:
            if stop_check and stop_check():
                return
            time.sleep(min(step, interval_sec - slept))
            slept += step

        if stop_check and stop_check():
            break
        
        with open(target_file, 'a') as f:
            f.write(f"{values[idx]}\n")
        idx += 1
