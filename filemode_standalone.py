"""
File Mode.
Работа обученной модели KAN в режиме с файлом.
Логирование standalone.
"""

# !pip install pykan==0.2.8
# !pip install -U kaleido==0.2.1
from app_logging import get_logger, configure_standalone_logging
from core import KANAnomalyDetector

import numpy as np

from datetime import datetime, timezone

import joblib
import torch
from kan import *

_log = get_logger("filemode_standalone")


if __name__ == "__main__":
    configure_standalone_logging()

    # ===============================
    # ПАРАМЕТРЫ
    # ===============================
    
    # Путь к данным и модели
    data_path = ""  # Данные для оценки
    model_path = "/content/weights/model"
    
    run_date = datetime.now(timezone.utc)
    res_files_path = f"/{run_date}/"
    
    # Загрузка скейлера (если нужен)
    scaler = None
    local_scaler = True
    if not local_scaler:
        scaler_path = "/content/scaler.bin"
        scaler = joblib.load(scaler_path)
    
    # GPU/CPU
    device = "cpu"
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    _log.info("Используется устройство: %s", device)
    
    # Параметры KANAnomalyDetector
    input_length = 1440
    output_length = 60
    test_size = 0.2
    step = 1000
    
    # Параметры split_and_normalize
    usage_volume = 1
    normalize_first = False
    
    # Параметры prepare_data
    normalize_sequence = True
    shuffle = True
    
    # Параметры build_model
    width = [input_length, 4, output_length]
    grid = 3
    k = 3
    seed = 0
    
   # ===============================
    # РЕГУЛИРОВКА ДЛЯ FILE MODE
    # ===============================
    
    usage_volume = 1.0
    # Используем всё как тестовые данные
    test_size = 1.0
    input_length = 288
    output_length = 10
    width = [input_length, 144, output_length]
    step = 1
    grid = 5
    shuffle = False
    shuffle_steps = False
    normalize_first = False
    normalize_sequence = True
    denormalize = True
    if not normalize_first and not normalize_sequence:
        denormalize = False
    
    # Параметры диапазона обработки
    # Если индексы 0, то будет обработан весь диапазон от input_length до len(data) - output_length
    # Индекс начала (включительно)
    start_index = 0
    # Индекс конца (включительно)
    end_index = 0
    # Дополнительные точки до и после
    extra_num = 0
    
    # ===============================
    # ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА МОДЕЛИ
    # ===============================
    
    _log.info("Начало загрузки данных...")
    predictor = KANAnomalyDetector(
        data_path, 
        input_length=input_length, 
        output_length=output_length, 
        test_size=test_size, 
        step=step, 
        scaler=scaler, 
        device=device
    )
    
    _log.info("Загрузка данных...")
    predictor.load_data()
    
    _log.info("Разделение и нормализация данных...")
    predictor.split_and_normalize(usage_volume=usage_volume, normalize_first=normalize_first)
    
    _log.info("Подготовка данных...")
    predictor.prepare_data(shuffle=shuffle, normalize_sequence=normalize_sequence)
    
    _log.info("Построение модели...")
    predictor.build_model(width=width, grid=grid)
    
    _log.info("Загрузка модели из: %s", model_path)
    predictor.load_model(model_path)
    
    # ===============================
    # ОБРАБОТКА ДАННЫХ И ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ
    # ===============================
    
    prediction = []
    real_output_denorm = []
    
    # Валидация диапазона
    min_index = input_length
    max_index = predictor.len_data - output_length
    
    if start_index == 0 and end_index == 0:
        start_index = min_index
        end_index = max_index
    
    if start_index < min_index or end_index > max_index or start_index > end_index:
        msg = (
            f"Некорректный диапазон! Допустимо: start_index >= {min_index}, "
            f"end_index <= {max_index}, start_index <= end_index."
        )
        _log.error(msg)
        raise ValueError(msg)
    
    # Вычисление индексов в наборе данных
    seq_start = (start_index - input_length) // step
    seq_end = (end_index - input_length) // step
    
    extra_seq_left = min(extra_num // step, seq_start)
    extra_seq_right = min(extra_num // step, predictor.len_X_test - seq_end - 1)
    
    left_range = max(0, seq_start - extra_seq_left)
    right_range = min(predictor.len_X_test, seq_end + extra_seq_right + 1)
    wanted_dot_num = right_range - left_range
    
    _log.info("Диапазон: start_index=%s, end_index=%s, extra_num=%s", start_index, end_index, extra_num)
    _log.info("Возможные индексы: %s ... %s", min_index, max_index)
    _log.info("Желаемое количество точек: %s", wanted_dot_num)
    
    # Первый шаг
    i = left_range
    input_data_for_prediction, scaler_idfp = predictor.X_test[i], predictor.X_test_scaler[i]
    pr_temp = np.round(
        predictor.predict(
            input_data_for_prediction,
            denormalize=denormalize,
            normalize_sequence=normalize_sequence,
            scaler=scaler_idfp,
        ).tolist(),
        decimals=3,
    )
    prediction.extend(pr_temp[-output_length:])
    real_output, scaler_ro = predictor.y_test[i], predictor.y_test_scaler[i]
    real_output_denorm.extend(
        np.round(predictor.denormalize_sequence_data(real_output, scaler_ro).tolist(), decimals=3)
    )
    
    prediction = prediction[:wanted_dot_num]
    real_output_denorm = real_output_denorm[:wanted_dot_num]
    
    # Остальные шаги
    for i in range(left_range + 1, right_range):
        input_data_for_prediction, scaler_idfp = predictor.X_test[i], predictor.X_test_scaler[i]
        pr_temp = np.round(
            predictor.predict(
                input_data_for_prediction,
                denormalize=denormalize,
                normalize_sequence=normalize_sequence,
                scaler=scaler_idfp,
            ).tolist(),
            decimals=3,
        )
        prediction.extend(pr_temp[-step:])
        real_output, scaler_ro = predictor.y_test[i], predictor.y_test_scaler[i]
        real_output_denorm.extend(
            np.round(
                predictor.denormalize_sequence_data(real_output, scaler_ro).tolist()[-step:],
                decimals=3,
            )
        )
        
        prediction = prediction[:wanted_dot_num]
        real_output_denorm = real_output_denorm[:wanted_dot_num]
        
        _log.debug("Шаг %s/%s: real=%.3f, pred=%.3f", i - left_range + 1, right_range - left_range, 
                  real_output_denorm[-1], prediction[-1])
        
        if len(prediction) >= wanted_dot_num and len(real_output_denorm) >= wanted_dot_num:
            break
    
    # Финальное обрезание
    prediction = prediction[:wanted_dot_num]
    real_output_denorm = real_output_denorm[:wanted_dot_num]
    
    _log.info("Первые 10 предсказаний: %s...", prediction[:10])
    _log.info("Первые 10 реальных значений: %s...", real_output_denorm[:10])
    
    # ===============================
    # ВИЗУАЛИЗАЦИЯ
    # ===============================
    
    _log.info("Построение графика сравнения...")
    if wanted_dot_num <= 100:
        predictor.plot_comparison(prediction, real_output_denorm, len_data=wanted_dot_num)
    else:
        predictor.plot_big_comparison(prediction, real_output_denorm, len_data=wanted_dot_num)
    
    end_date = datetime.now(timezone.utc)
    _log.info("Время выполнения: %s", end_date - run_date)
    _log.info("Обработка файла завершена успешно!")
