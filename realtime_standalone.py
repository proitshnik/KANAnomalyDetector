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
from anomaly_factory import AnomalyDetectorFactory
from anomaly_formatting import AnomalyFormatter
from anomaly_reporting import AnomalyReportSession

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
    
    # Обнаружение аномалий
    anomaly_detection = True
    
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
    
    # ===============================
    # ИНИЦИАЛИЗАЦИЯ МОДУЛЯ ОБНАРУЖЕНИЯ АНОМАЛИЙ
    # ===============================
    anomaly_module = None
    if anomaly_detection:
        try:
            anomaly_module = AnomalyDetectorFactory.create(mode="Процент отклонения", threshold_percent=1)
            _log.info("Модуль обнаружения аномалий инициализирован")
        except Exception as e:
            _log.warning("Не удалось инициализировать модуль обнаружения аномалий: %s", str(e))
            anomaly_module = None
    
    # остаток кода файлового режима для проверки данных из файла на метрики
    if data_path:
        _log.info("Оценка на тестовых данных...")
        metrics = predictor.evaluate(normalize_sequence=normalize_sequence)
        _log.info("Метрики полученные:")
        _log.info("MSE: %.3f", metrics["mse"])
        _log.info("MAE: %.3f", metrics["mae"])
        _log.info("RMSE: %.3f", metrics["rmse"])
        _log.info("MAPE: %.2f%%", metrics["mape"])
        _log.info("MASE: %.3f", metrics["mase"])
        _log.info("Deviation L1: %.3f", metrics["deviation_l1"])
        _log.info("Deviation L2: %.3f", metrics["deviation_l2"])
    
    # ===============================
    # МОНИТОРИНГ В РЕАЛЬНОМ ВРЕМЕНИ
    # ===============================
    _log.info("Начало мониторинга файла: %s", monitor_file)
    
    # Переменные состояния
    state = {
        "real_output_denorm": [],
        "prediction": [],
        "report_session": AnomalyReportSession(),
        "last_len": 0
    }
    
    def monitor_callback(values, mode):
        if mode == "init":
            # Первый шаг - берем последние input_length значений и делаем предсказание
            if len(values) >= input_length:
                real_seq = np.array(values[-input_length:])
                
                # Применяем фильтрацию, если нужно
                if use_filter:
                    _log.debug("Фильтрация: ВКЛЮЧЕНА")
                    real_seq = filter_data(real_seq.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard")
                    real_seq = np.array(real_seq)
                else:
                    _log.debug("Фильтрация: ОТКЛЮЧЕНА")
                
                norm_seq, seq_scaler = predictor.normalize_sequence(real_seq)
                pred = predictor.predict(
                    norm_seq,
                    denormalize=True,
                    normalize_sequence=True,
                    scaler=seq_scaler
                )
                
                state["real_output_denorm"] = list(values[-input_length:])
                state["prediction"] = list(pred)
                state["last_len"] = len(values)
        
        elif mode == "step":
            # Обрабатываем новые значения
            num_new_values = len(values) - state["last_len"]
            for i in range(num_new_values):
                new_real_value = values[state["last_len"] + i]
                state["real_output_denorm"].append(new_real_value)

                if len(state["real_output_denorm"]) >= input_length:
                    real_seq = np.array(state["real_output_denorm"][-input_length:])
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
                    state["prediction"].append(pred[-1])
            
            state["last_len"] = len(values)

        # Построение полных массивов для анализа
        num_real = len(state["real_output_denorm"])
        num_pred = len(state["prediction"])
        num_new_real = num_real - input_length
        total_len = input_length + max(num_new_real, num_pred)

        time_arr = list(range(total_len))
        real_output_denorm_full = list(state["real_output_denorm"][: input_length])
        if num_new_real > 0:
            real_output_denorm_full.extend(state["real_output_denorm"][input_length:])
        if len(real_output_denorm_full) < total_len:
            real_output_denorm_full.extend([np.nan] * (total_len - len(real_output_denorm_full)))

        prediction_full = [np.nan] * input_length
        prediction_full.extend(list(state["prediction"]))
        if len(prediction_full) < total_len:
            prediction_full.extend([np.nan] * (total_len - len(prediction_full)))

        min_len = min(len(real_output_denorm_full), len(prediction_full), len(time_arr))
        real_output_denorm_full = real_output_denorm_full[:min_len]
        prediction_full = prediction_full[:min_len]
        time_arr = time_arr[:min_len]

        # Обнаружение аномалий
        check_start = input_length
        check_end = len(real_output_denorm_full)
        valid_pairs = [
            (i, real_output_denorm_full[i], prediction_full[i])
            for i in range(check_start, check_end)
            if not np.isnan(real_output_denorm_full[i]) and not np.isnan(prediction_full[i])
        ]

        anomaly_indices = []
        if anomaly_module and valid_pairs:
            indices, real_vals, pred_vals = zip(*valid_pairs)
            try:
                detected_indices = anomaly_module.detect(list(real_vals), list(pred_vals))
                anomaly_indices = [indices[idx] for idx in detected_indices if idx < len(indices)]
            except Exception as e:
                _log.debug("Ошибка при обнаружении аномалий: %s", str(e))

        new_anomalies = state["report_session"].new_indices(anomaly_indices)
        if new_anomalies:
            for idx in new_anomalies:
                try:
                    msg = AnomalyFormatter.format_anomaly_message(
                        [idx], [real_output_denorm_full[idx]], [prediction_full[idx]],
                        title=f"Аномалия в точке {idx}",
                    )
                    if msg:
                        _log.warning("АНОМАЛИЯ ОБНАРУЖЕНА:\n%s", msg)
                except Exception as e:
                    _log.debug("Ошибка при форматировании сообщения об аномалии: %s", str(e))
    
    try:
        monitor_data(
            monitor_file,
            input_length=input_length,
            output_length=output_length,
            interval_sec=interval_sec,
            callback=monitor_callback,
            stop_check=None
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
        
        if len(state["real_output_denorm"]) > 0 and len(state["prediction"]) > 0:
            _log.info("Построение финального графика...")
            try:
                num_real = len(state["real_output_denorm"])
                num_pred = len(state["prediction"])
                num_new_real = num_real - input_length
                total_len = input_length + max(num_new_real, num_pred)
                
                time_arr = list(range(total_len))
                
                real_output_denorm = list(state["real_output_denorm"][: input_length])
                if num_new_real > 0:
                    real_output_denorm.extend(state["real_output_denorm"][input_length:])
                if len(real_output_denorm) < total_len:
                    real_output_denorm.extend([np.nan] * (total_len - len(real_output_denorm)))
                
                prediction = [np.nan] * input_length
                prediction.extend(list(state["prediction"]))
                if len(prediction) < total_len:
                    prediction.extend([np.nan] * (total_len - len(prediction)))
                
                min_len = min(len(real_output_denorm), len(prediction), len(time_arr))
                real_output_denorm = real_output_denorm[:min_len]
                prediction = prediction[:min_len]
                time_arr = time_arr[:min_len]
                
                _log.info("Итого собрано: %d точек", min_len)
                _log.info("Первые 10 реальных: %s", real_output_denorm[:10])
                _log.info("Первые 10 предсказаний: %s", prediction[:10])
                
                if min_len <= 100:
                    predictor.plot_comparison(prediction, real_output_denorm, len_data=min_len)
                else:
                    predictor.plot_big_comparison(prediction, real_output_denorm, len_data=min_len)
            except Exception as e:
                _log.error("Ошибка при построении графика: %s", str(e))
        
        end_date = datetime.now(timezone.utc)
        _log.info("Время работы: %s", end_date - run_date)
        _log.info("Мониторинг завершен!")
