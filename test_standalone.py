"""
Test Mode.
Тестирование модели KAN.
Логирование standalone.
"""

# !pip install pykan==0.2.8
# !pip install -U kaleido==0.2.1
from app_logging import get_logger, configure_standalone_logging
from anomaly_detector import AnomalyDetectorModule
from core import KANAnomalyDetector
import numpy as np
import math

from datetime import datetime, timezone

import torch
from kan import *

_log = get_logger("test_standalone")


if __name__ == "__main__":
    configure_standalone_logging()

    # ===============================
    # ПАРАМЕТРЫ
    # ===============================
    
    # Путь к данным и модели
    data_path = "kan_general_Filt.txt"
    model_path = "/content/weights/model"
    
    run_date = datetime.now(timezone.utc)
    res_files_path = f"/{run_date}/"
    
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
    
    # Визуализация
    symbolic_view_flag = False
    plot_view_flag = False
    comparison_flag = True
    
    # Обнаружение аномалий
    anomaly_detection = False
    
    # ===============================
    # РЕГУЛИРОВКА ДЛЯ ТЕСТИРОВАНИЯ
    # ===============================
    
    usage_volume = 0.1
    normalize_first = False
    normalize_sequence = True
    denormalize = True
    if not normalize_first and not normalize_sequence:
        denormalize = False
    
    test_size = 0.2
    input_length = 288
    output_length = 10
    width = [input_length, 144, output_length]
    step = 1
    grid = 5
    # opt="Adam"
    
    # ===============================
    # ТЕСТИРОВАНИЕ
    # ===============================
    
    _log.info("Начало загрузки данных...")
    predictor = KANAnomalyDetector(
        data_path, 
        input_length=input_length, 
        output_length=output_length, 
        test_size=test_size, 
        step=step, 
        device=device
    )
    
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
    
    if symbolic_view_flag:
        _log.info("Построение символической модели...")
        predictor.symbolic_model_view(train_mode=False)
    
    if plot_view_flag:
        _log.info("Построение графика модели...")
        predictor.plot_model_view(train_mode=False)
    
    # ===============================
    # ОЦЕНКА
    # ===============================
    
    _log.info("Оценка на тестовых данных...")
    metrics = predictor.evaluate(normalize_sequence=normalize_sequence)
    
    _log.info("Метрики:")
    _log.info("MSE: %.3f", metrics["mse"])
    _log.info("MAE: %.3f", metrics["mae"])
    _log.info("RMSE: %.3f", metrics["rmse"])
    _log.info("MAPE: %.2f%%", metrics["mape"])
    _log.info("MASE: %.3f", metrics["mase"])
    _log.info("Deviation L1: %.3f", metrics["deviation_l1"])
    _log.info("Deviation L2: %.3f", metrics["deviation_l2"])
    
    # ===============================
    # СРАВНЕНИЕ ПРЕДСКАЗАНИЙ И РЕАЛЬНЫХ ЗНАЧЕНИЙ
    # ===============================
    
    if comparison_flag:
        wanted_dot_num = 60
        prediction = []
        real_output_denorm = []
        
        input_data_for_prediction, scaler_idfp = predictor.X_test[0], predictor.X_test_scaler[0]
        prediction.extend(
            np.round(
                predictor.predict(
                    input_data_for_prediction, 
                    denormalize=denormalize, 
                    normalize_sequence=normalize_sequence, 
                    scaler=scaler_idfp
                ).tolist(), 
                decimals=3
            )
        )
        
        real_output, scaler_ro = predictor.y_test[0], predictor.y_test_scaler[0]
        if denormalize:
            if not normalize_sequence:
                real_output_denorm.extend(
                    np.round(predictor.denormalize_data(real_output).tolist(), decimals=3)
                )
            else:
                real_output_denorm.extend(
                    np.round(
                        predictor.denormalize_sequence_data(real_output, scaler_ro).tolist(), 
                        decimals=3
                    )
                )
        else:
            real_output_denorm.extend(real_output)
        
       if output_length < wanted_dot_num:
            range_num = math.ceil((wanted_dot_num - output_length) / step)
            if range_num > predictor.len_X_test:
                range_num = predictor.len_X_test
            
            for i in range(1, range_num + 1):
                input_data_for_prediction, scaler_idfp = predictor.X_test[i], predictor.X_test_scaler[i]
                pr_temp = np.round(
                    predictor.predict(
                        input_data_for_prediction, 
                        denormalize=denormalize, 
                        normalize_sequence=normalize_sequence, 
                        scaler=scaler_idfp
                    ).tolist(), 
                    decimals=3
                )
                prediction.extend(pr_temp[output_length - step:])
                
                real_output, scaler_ro = predictor.y_test[i], predictor.y_test_scaler[i]
                if denormalize:
                    if not normalize_sequence:
                        real_output_denorm.extend(
                            np.round(
                                predictor.denormalize_data(real_output).tolist()[output_length - step:], 
                                decimals=3
                            )
                        )
                    else:
                        real_output_denorm.extend(
                            np.round(
                                predictor.denormalize_sequence_data(real_output, scaler_ro).tolist()[output_length - step:], 
                                decimals=3
                            )
                        )
                else:
                    real_output_denorm.extend(real_output[output_length - step:])
        
        prediction = prediction[:wanted_dot_num]
        real_output_denorm = real_output_denorm[:wanted_dot_num]
        _log.info("Первые 10 предсказаний: %s", prediction[:10])
        _log.info("Первые 10 реальных значений: %s", real_output_denorm[:10])
        
        # Обнаружение аномалий (опционально)
        anomaly_indices = []
        if anomaly_detection:
            try:
                anomaly_module = AnomalyDetectorModule()
                anomaly_indices = anomaly_module.detect(real_output_denorm, prediction)
                if anomaly_indices:
                    _log.info("Обнаружено %d аномалий в индексах: %s", len(anomaly_indices), anomaly_indices)
                else:
                    _log.info("Аномалий не обнаружено")
            except Exception as e:
                _log.warning("Не удалось обнаружить аномалии: %s", str(e))
        
        if wanted_dot_num <= 100:
            predictor.plot_comparison(prediction, real_output_denorm, len_data=wanted_dot_num)
        else:
            predictor.plot_big_comparison(prediction, real_output_denorm, len_data=wanted_dot_num)
    
    end_date = datetime.now(timezone.utc)
    _log.info("Время выполнения: %s", end_date - run_date)
    _log.info("Тестирование завершено успешно!")
