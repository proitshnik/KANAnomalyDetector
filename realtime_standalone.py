"""
Realtime Mode.
Работа обученной модели KAN в режиме реального времени с мониторингом данных.
Логирование standalone.
"""

# !pip install pykan==0.2.8
# !pip install -U kaleido==0.2.1
from app_logging import get_logger, configure_standalone_logging
from core import KANAnomalyDetector
from monitoring import monitor_data
from filter import filter_data

import numpy as np

from datetime import datetime, timezone

import joblib
import torch
from kan import *

_log = get_logger("realtime_standalone")


if __name__ == "__main__":
    configure_standalone_logging()

    # ===============================
    # ПАРАМЕТРЫ
    # ===============================
    
    # Путь к данным и модели
    data_path = ""
    monitor_file = "realtime_monitoring.txt"
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
    
    # Фильтрация
    use_filter = False
    
    # ===============================
    # РЕГУЛИРОВКА ДЛЯ РЕАЛЬНОГО ВРЕМЕНИ
    # ===============================
    
    usage_volume = 1.0
    test_size = 1.0
    input_length = 288
    output_length = 10
    width = [input_length, 144, output_length]
    step = 1
    grid = 5
    # opt="Adam"
    shuffle = False
    shuffle_steps = False
    normalize_first = False
    normalize_sequence = True
    denormalize = True
    if not normalize_first and not normalize_sequence:
        denormalize = False
    
    # Параметры мониторинга
    interval_sec = 1
    
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
    
    # остаток кода файлового режима для загрузки и обработки данных для проверки данных из файла на метрики
    if data_path:
        _log.info("Первые данные загружаются из: %s", data_path)
        _log.info("Загрузка данных...")
        predictor.load_data()
        
        _log.info("Разделение и нормализация данных...")
        predictor.split_and_normalize(usage_volume=usage_volume, normalize_first=normalize_first)
        
        # Применяем фильтрацию, если нужно (учитываем, что normalize_sequence включен и после фильтрации)
        if use_filter:
            _log.info("Фильтрация: ВКЛЮЧЕНА")
            predictor.filter_test_data()
        else:
            _log.info("Фильтрация: ОТКЛЮЧЕНА")
        
        _log.info("Подготовка данных...")
        predictor.prepare_data(shuffle=shuffle, normalize_sequence=normalize_sequence)
    
    _log.info("Построение модели...")
    predictor.build_model(width=width, grid=grid)
    
    _log.info("Загрузка модели из: %s", model_path)
    predictor.load_model(model_path)
    
    # остаток кода файлового режима для проверки данных из файла на метрики
    if data_path:
        _log.info("Оценка на тестовых данных...")
        mse, deviation_l2, deviation_l1 = predictor.evaluate()
        _log.info("Среднеквадратичная ошибка: %f", mse)
        _log.info("Отклонение по L2-норме: %f", deviation_l2)
        _log.info("Отклонение по L1-норме: %f", deviation_l1)
    
    # ===============================
    # МОНТОРИНГ В РЕАЛЬНОМ ВРЕМЕНИ
    # ===============================
    
    _log.info("Начало мониторинга файла: %s", monitor_file)
    
    # Переменные состояния
    state = {
        'realtime_real_buffer': [],
        'realtime_pred_buffer': [],
        'last_len': 0
    }
    
    def monitor_callback(values, mode):
        if mode == "init":
            # Первый шаг - берем последние input_length значений и делаем предсказание
            if len(values) >= input_length:
                real_seq = np.array(values[-input_length:])
                
                # Применяем фильтрацию, если нужно
                if use_filter:
                    _log.info("Фильтрация: ВКЛЮЧЕНА")
                    real_seq = filter_data(real_seq.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard")
                    real_seq = np.array(real_seq)
                else:
                    _log.info("Фильтрация: ОТКЛЮЧЕНА")
                
                norm_seq, seq_scaler = predictor.normalize_sequence(real_seq)
                pred = predictor.predict(
                    norm_seq,
                    denormalize=True,
                    normalize_sequence=True,
                    scaler=seq_scaler
                )
                
                state['realtime_real_buffer'] = list(values[-input_length:])
                state['realtime_pred_buffer'] = list(pred)
                state['last_len'] = len(values)
                
                _log.info("Инициализация буферов: %d реальных точек, %d предсказаний", 
                         len(state['realtime_real_buffer']), len(state['realtime_pred_buffer']))
        
        elif mode == "step":
            # Обрабатываем новые значения
            num_new_values = len(values) - state['last_len']
            
            for i in range(num_new_values):
                new_real_value = values[state['last_len'] + i]
                state['realtime_real_buffer'].append(new_real_value)
                
                if len(state['realtime_real_buffer']) >= input_length:
                    real_seq = np.array(state['realtime_real_buffer'][-input_length:])
                    
                    # Применяем фильтрацию, если нужно
                    if use_filter:
                        _log.debug(f"Фильтрация применена к последовательности из {len(real_seq)} точек")
                        real_seq = filter_data(real_seq.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard")
                        real_seq = np.array(real_seq)
                    
                    norm_seq, seq_scaler = predictor.normalize_sequence(real_seq)
                    pred = predictor.predict(
                        norm_seq,
                        denormalize=True,
                        normalize_sequence=True,
                        scaler=seq_scaler
                    )
                    state['realtime_pred_buffer'].append(pred[-1])
            
            state['last_len'] = len(values)
            
            # Вывод статистики
            num_real = len(state['realtime_real_buffer'])
            num_pred = len(state['realtime_pred_buffer'])
            _log.info("Шаг: %d реальных точек, %d предсказаний", num_real, num_pred)
            
            if num_real > 0 and num_pred > 0:
                _log.debug("Последняя реальная: %f, последнее предсказание: %f", 
                          state['realtime_real_buffer'][-1], state['realtime_pred_buffer'][-1])
    
    try:
        monitor_data(
            monitor_file,
            input_length=input_length,
            output_length=output_length,
            interval_sec=interval_sec,
            callback=monitor_callback,
            stop_check=None  # Можно добавить функцию проверки остановки
        )
    except KeyboardInterrupt:
        _log.info("Мониторинг прерван пользователем")
    except FileNotFoundError:
        _log.error("Файл мониторинга не найден: %s", monitor_file)
    except Exception as e:
        _log.error("Ошибка при мониторинге: %s", str(e))
    finally:
        # ===============================
        # ВИЗУАЛИЗАЦИЯ
        # ===============================
        
        if len(state['realtime_real_buffer']) > 0 and len(state['realtime_pred_buffer']) > 0:
            _log.info("Построение финального графика...")
            try:
                num_real = len(state['realtime_real_buffer'])
                num_pred = len(state['realtime_pred_buffer'])
                num_new_real = num_real - input_length
                total_len = input_length + max(num_new_real, num_pred)
                
                time_arr = list(range(total_len))
                
                real_full = list(state['realtime_real_buffer'][: input_length])
                if num_new_real > 0:
                    real_full.extend(state['realtime_real_buffer'][input_length :])
                if len(real_full) < total_len:
                    real_full.extend([np.nan] * (total_len - len(real_full)))
                
                pred_full = [np.nan] * input_length
                pred_full.extend(list(state['realtime_pred_buffer']))
                if len(pred_full) < total_len:
                    pred_full.extend([np.nan] * (total_len - len(pred_full)))
                
                min_len = min(len(real_full), len(pred_full), len(time_arr))
                real_full = real_full[:min_len]
                pred_full = pred_full[:min_len]
                time_arr = time_arr[:min_len]
                
                _log.info("Итого собрано: %d точек", min_len)
                _log.info("Первые 10 реальных: %s", real_full[:10])
                _log.info("Первые 10 предсказаний: %s", pred_full[:10])
                
                if min_len <= 100:
                    predictor.plot_comparison(pred_full, real_full, len_data=min_len)
                else:
                    predictor.plot_big_comparison(pred_full, real_full, len_data=min_len)
            except Exception as e:
                _log.error("Ошибка при построении графика: %s", str(e))
        
        end_date = datetime.now(timezone.utc)
        _log.info("Время работы: %s", end_date - run_date)
        _log.info("Мониторинг завершен!")
