"""
Модуль для оценки модели обнаружения аномалий
"""

import numpy as np
from app_logging import get_logger, init_default_logging_if_needed
from anomaly_factory import AnomalyDetectorFactory

init_default_logging_if_needed()
_log = get_logger("evaluation")


# ===============================
# ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЯ МЕТРИК КАЧЕСТВА ПРЕДСКАЗАНИЯ (РЕГРЕССИЯ)
# ===============================
def calculate_mse(real, predicted):
    """Среднеквадратичная ошибка (MSE, Mean Squared Error)"""
    return np.mean((predicted - real) ** 2)


def calculate_deviation_l1(real, predicted):
    """Отклонение по L1-норме (сумма абсолютных ошибок)"""
    return np.sum(np.abs(predicted - real))


def calculate_deviation_l2(real, predicted):
    """Отклонение по L2-норме (евклидова норма)"""
    return np.sqrt(np.sum((predicted - real) ** 2))


def calculate_mae(real, predicted):
    """Средняя абсолютная ошибка (MAE, Mean Absolute Error)"""
    return np.mean(np.abs(predicted - real))


def calculate_rmse(real, predicted):
    """Корневая среднеквадратичная ошибка (RMSE, Root Mean Squared Error)"""
    return np.sqrt(calculate_mse(real, predicted))


def calculate_mape(real, predicted):
    """Средняя абсолютная процентная ошибка (MAPE, Mean Absolute Percentage Error)"""
    mask = real != 0
    if np.sum(mask) == 0:
        return 0.0 if np.allclose(predicted, 0) else float("inf")
    
    ape = np.abs((real[mask] - predicted[mask]) / real[mask])
    return np.mean(ape) * 100


def calculate_mase(real, predicted):
    """
    Средняя абсолютная масштабированная ошибка (MASE, Mean Absolute Scaled Error)
    Масштабируется по сравнению с наивным прогнозом (предыдущее значение)
    """
    n = len(real)
    if n < 2:
        return 0.0
    
    naive_forecast = np.concatenate(([real[0]], real[:-1]))
    denominator = np.mean(np.abs(real - naive_forecast))
    
    if denominator == 0:
        return 0.0 if np.allclose(real, predicted) else float("inf")
    
    return np.mean(np.abs(real - predicted)) / denominator


# ===============================
# КЛАСС ДЛЯ ОЦЕНКИ КАЧЕСТВА ПРЕДСКАЗАНИЯ (РЕГРЕССИЯ)
# ===============================
class RegressionEvaluator:
    """Оценка качества предсказания на обычных данных для регрессии"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, real, predicted):
        """Оценка предсказаний"""
        if len(real) != len(predicted):
            _log.error(f"Длины не совпадают: {len(real)} and {len(predicted)}")
            raise ValueError()
        
        self.metrics = {
            "mse": calculate_mse(real, predicted),
            "mae": calculate_mae(real, predicted),
            "rmse": calculate_rmse(real, predicted),
            "mape": calculate_mape(real, predicted),
            "mase": calculate_mase(real, predicted),
            "deviation_l1": calculate_deviation_l1(real, predicted),
            "deviation_l2": calculate_deviation_l2(real, predicted),
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Вывести метрики"""
        if not self.metrics:
            _log.warning("Метрики не вычислены")
            return
        
        _log.info("Метрики качества предсказания:")
        _log.info(f"MSE: {self.metrics["mse"]:.3f}")
        _log.info(f"MAE: {self.metrics["mae"]:.3f}")
        _log.info(f"RMSE: {self.metrics["rmse"]:.3f}")
        _log.info(f"MAPE: {self.metrics["mape"]:.2f}%")
        _log.info(f"MASE: {self.metrics["mase"]:.3f}")
        _log.info(f"Deviation L1: {self.metrics["deviation_l1"]:.3f}")
        _log.info(f"Deviation L2: {self.metrics["deviation_l2"]:.3f}")


# ===============================
# КЛАСС ДЛЯ ОЦЕНКИ ЭФФЕКТИВНОСТИ ОБНАРУЖЕНИЯ АНОМАЛИЙ
# ===============================
class AnomalyDetectionEvaluator:
    """Оценка эффективности обнаружения аномалий"""
    
    def __init__(self, real_anomaly_ranges):
        """
        Принимает список кортежей, где есть аномалии в реальных данных.
        """
        self.real_anomaly_ranges = real_anomaly_ranges
        self.detected_anomaly_indices = None
        self.metrics = {}
    
    def set_detected_anomalies(self, detected_indices):
        """Установить индексы, где модель обнаружила аномалии"""
        self.detected_anomaly_indices = np.array(detected_indices, dtype=int)
    
    def _get_real_anomaly_indices(self, total_length):
        """Получить все индексы реальных аномалий"""
        indices = set()
        for start, end in self.real_anomaly_ranges:
            indices.update(range(start, min(end + 1, total_length)))
        return np.array(sorted(list(indices)))
    
    def _get_segments_from_indices(self, indices):
        """Преобразовать индексы в сегменты"""
        if len(indices) == 0:
            return []
        
        segments = []
        start = indices[0]
        prev = indices[0]
        
        for idx in indices[1:]:
            if idx > prev + 1:
                segments.append((start, prev))
                start = idx
            prev = idx
        
        segments.append((start, prev))
        return segments
    
    def _is_segment_detected(self, real_segment, detected_indices, min_coverage=0.01):
        """Проверить, обнаружен ли сегмент аномалии"""
        segment_start, segment_end = real_segment
        segment_length = segment_end - segment_start + 1
        
        if 0 < min_coverage < 1:
            threshold = max(1, int(segment_length * min_coverage))
        else:
            threshold = max(1, int(min_coverage))
        
        intersection = len([idx for idx in detected_indices if segment_start <= idx <= segment_end])
        
        return intersection >= threshold
    
    def evaluate(self, total_length, min_coverage_for_segment=0.01, min_index_coverage=None):
        """
        Оценка эффективности обнаружения.
            total_length - общая длина данных.
            min_coverage_for_segment - минимальное покрытие сегмента для его засчитывания как обнаруженного (процент от длины сегмента).
            min_index_coverage - явное количество индексов для покрытия (если указано, переопределяет процент).
        """
        if self.detected_anomaly_indices is None:
            _log.error("Не установлены обнаруженные аномалии для оценки")
            raise ValueError()
        
        real_anomaly_indices = self._get_real_anomaly_indices(total_length)
        real_segments = self._get_segments_from_indices(real_anomaly_indices)
        detected_segments = self._get_segments_from_indices(self.detected_anomaly_indices)
        
        if min_index_coverage is not None:
            min_coverage = min_index_coverage
        else:
            min_coverage = min_coverage_for_segment
        
        # 1 Покрытие участков
        detected_segments_count = sum(
            1 for seg in real_segments 
            if self._is_segment_detected(seg, self.detected_anomaly_indices, min_coverage)
        )
        total_real_segments = len(real_segments)
        segments_coverage_percent = (detected_segments_count / total_real_segments * 100) if total_real_segments > 0 else 0.0
        
        # 2 Покрытие индексов
        detected_real_indices = len(np.intersect1d(real_anomaly_indices, self.detected_anomaly_indices))
        total_real_indices = len(real_anomaly_indices)
        indices_coverage_percent = (detected_real_indices / total_real_indices * 100) if total_real_indices > 0 else 0.0
        
        # 3 Ложные срабатывания
        non_anomaly_indices = np.array([i for i in range(total_length) if i not in real_anomaly_indices])
        false_positives = len(np.intersect1d(non_anomaly_indices, self.detected_anomaly_indices))
        total_non_anomaly_indices = len(non_anomaly_indices)
        false_positives_percent = (false_positives / total_non_anomaly_indices * 100) if total_non_anomaly_indices > 0 else 0.0
        
        # 4 Сдвиги начала/конца
        shift_metrics = self._calculate_shift_metrics(real_segments, detected_segments)
        
        # 5 Дополнительные метрики: Precision, Recall, F1-score, IoU
        # TP - истинные положительные (правильно обнаруженные аномалии)
        tp = detected_real_indices
        # FP - ложные положительные (неправильно обнаруженные аномалии)
        fp = false_positives
        # FN - ложные отрицательные (пропущенные аномалии)
        fn = total_real_indices - detected_real_indices
        # TN - истинные отрицательные (правильно не обнаруженные неаномалии)
        tn = total_non_anomaly_indices - false_positives
        
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        intersection = len(np.intersect1d(real_anomaly_indices, self.detected_anomaly_indices))
        union = len(np.union1d(real_anomaly_indices, self.detected_anomaly_indices))
        iou = (intersection / union * 100) if union > 0 else 0.0
        
        self.metrics = {
            "segments_coverage": {
                "percent": segments_coverage_percent,
                "detected": detected_segments_count,
                "total": total_real_segments,
                "debug": f"{detected_segments_count}/{total_real_segments}"
            },
            "indices_coverage": {
                "percent": indices_coverage_percent,
                "detected": detected_real_indices,
                "total": total_real_indices,
                "debug": f"{detected_real_indices}/{total_real_indices}"
            },
            "false_positives": {
                "percent": false_positives_percent,
                "count": false_positives,
                "total_non_anomaly": total_non_anomaly_indices,
                "debug": f"{false_positives}/{total_non_anomaly_indices}"
            },
            "shift_metrics": shift_metrics,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "iou": iou,
            "confusion_matrix": {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        }
        
        return self.metrics
    
    def _calculate_shift_metrics(self, real_segments, detected_segments):
        """Вычислить метрики сдвига начала и конца аномалий"""
        if len(real_segments) == 0:
            return {
                "start_earlier": {"percent": 0.0, "count": 0, "total": 0, "avg_shift": 0, "shifts": []},
                "start_later": {"percent": 0.0, "count": 0, "total": 0, "avg_shift": 0, "shifts": []},
                "end_earlier": {"percent": 0.0, "count": 0, "total": 0, "avg_shift": 0, "shifts": []},
                "end_later": {"percent": 0.0, "count": 0, "total": 0, "avg_shift": 0, "shifts": []},
            }
        
        start_earlier = []
        start_later = []
        end_earlier = []
        end_later = []
        
        for real_start, real_end in real_segments:
            closest_detected = None
            min_distance = float("inf")
            
            for det_start, det_end in detected_segments:
                distance = min(abs(real_start - det_start), abs(real_end - det_end))
                if distance < min_distance:
                    min_distance = distance
                    closest_detected = (det_start, det_end)
            
            if closest_detected is None:
                continue
            
            det_start, det_end = closest_detected
            
            start_shift = det_start - real_start
            end_shift = det_end - real_end
            
            if start_shift < 0:
                start_earlier.append(int(abs(start_shift)))
            elif start_shift > 0:
                start_later.append(int(start_shift))
            
            if end_shift < 0:
                end_earlier.append(int(abs(end_shift)))
            elif end_shift > 0:
                end_later.append(int(end_shift))
        
        return {
            "start_earlier": {
                "percent": (len(start_earlier) / len(real_segments) * 100) if real_segments else 0.0,
                "count": len(start_earlier),
                "total": len(real_segments),
                "avg_shift": np.mean(start_earlier) if start_earlier else 0,
                "shifts": start_earlier,
            },
            "start_later": {
                "percent": (len(start_later) / len(real_segments) * 100) if real_segments else 0.0,
                "count": len(start_later),
                "total": len(real_segments),
                "avg_shift": np.mean(start_later) if start_later else 0,
                "shifts": start_later,
            },
            "end_earlier": {
                "percent": (len(end_earlier) / len(real_segments) * 100) if real_segments else 0.0,
                "count": len(end_earlier),
                "total": len(real_segments),
                "avg_shift": np.mean(end_earlier) if end_earlier else 0,
                "shifts": end_earlier,
            },
            "end_later": {
                "percent": (len(end_later) / len(real_segments) * 100) if real_segments else 0.0,
                "count": len(end_later),
                "total": len(real_segments),
                "avg_shift": np.mean(end_later) if end_later else 0,
                "shifts": end_later,
            }
        }
    
    def print_metrics(self, ultra_debug=False):
        """Вывести метрики"""
        if not self.metrics:
            _log.warning("Метрики не вычислены")
            return
        
        _log.info("Метрики эффективности обнаружения аномалий:")
        
        m1 = self.metrics["segments_coverage"]
        _log.info(f"1. Покрытие участков аномалий: {m1["percent"]:.2f}% ({m1["debug"]})")
        
        m2 = self.metrics["indices_coverage"]
        _log.info(f"2. Покрытие индексов аномалий: {m2["percent"]:.2f}% ({m2["debug"]})")
        
        m3 = self.metrics["false_positives"]
        _log.info(f"3. Ложные срабатывания: {m3["percent"]:.2f}% ({m3["debug"]})")
        
        _log.info("4. Анализ сдвигов начала и конца аномалий в предсказании:")
        sm = self.metrics["shift_metrics"]
        
        for key, label in [("start_earlier", "Начало раньше"),
                            ("start_later", "Начало позже"),
                            ("end_earlier", "Конец раньше"),
                            ("end_later", "Конец позже")]:
            _log.info(f"{label}: {sm[key]['percent']:.2f}% ({sm[key]['count']}/{sm[key]['total']}, в среднем {sm[key]['avg_shift']:.1f} шагов)")
            
            if ultra_debug and sm[key]["shifts"]:
                _log.debug(f"   Детали: {sm[key]['shifts']}")
        
        _log.info("5. Дополнительные метрики:")
        
        cm = self.metrics["confusion_matrix"]
        _log.info(f"TP: {cm['tp']}, FP: {cm['fp']}, FN: {cm['fn']}, TN: {cm['tn']}")

        _log.info(f"Precision: {self.metrics['precision']:.2f}%")
        _log.info(f"Recall: {self.metrics['recall']:.2f}%")
        _log.info(f"F1 Score: {self.metrics['f1_score']:.2f}%")
        _log.info(f"IoU: {self.metrics['iou']:.2f}%")


# ===============================
# ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ДАННЫХ И ВЫПОЛНЕНИЯ СЦЕНАРИЯ ОЦЕНКИ
# ===============================
def load_data_from_file(filepath):
    """Загрузить данные из текстового файла"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    values = []
    for line in lines:
        values.extend([float(x) for x in line.strip().split()])
    
    return np.array(values)


def evaluation_workflow():
    """
    Сценарий для оценки модели.
        real_data_path - путь к файлу с реальными данными.
        predicted_data_path - путь к файлу с предсказанными данными.
        real_anomaly_ranges - индексы реальных аномалий (кортежи начало-конец).
        detector_mode - режим работы детектора.
        detector_threshold - порог обнаружения в % отклонения.
        test_mode - "regression" для оценки качества предсказания, "anomaly_detection" для оценки обнаружения аномалий.
        ultra_debug - если True, выводит дополнительные детали для отладки.
    """
    
    # ===============================
    # Параметры для оценивания модели
    # ===============================
    real_data_path = "input/real_data.txt"
    predicted_data_path = "input/predicted_data.txt"
    real_anomaly_ranges = [
        (0, 0),
    ]
    detector_mode = "Процент отклонения"
    detector_threshold = 2.0
    # "regression" или "anomaly_detection"
    test_mode = "regression"
    ultra_debug = False
    
    # ===============================
    # Загрузка данных
    # ===============================
    _log.info("Загрузка данных...")
    try:
        real_data = load_data_from_file(real_data_path)
        predicted_data = load_data_from_file(predicted_data_path)
    except Exception as e:
        _log.error(f"Ошибка загрузки файлов: {e}")
        return
    
    _log.info(f"Реальные: {len(real_data)}, Предсказанные: {len(predicted_data)}")
    
    if len(real_data) != len(predicted_data):
        _log.error("Длины реальных и предсказанных данных не совпадают")
        return
    
    total_length = len(real_data)
    
    # ===============================
    # Оценка качества предсказания и обнаружения аномалий
    # ===============================
    if test_mode == "regression":
        _log.info("Оценка качества предсказания (регрессии)...")
        reg_evaluator = RegressionEvaluator()
        reg_evaluator.evaluate(real_data, predicted_data)
        reg_evaluator.print_metrics()
    elif test_mode == "anomaly_detection":
        _log.info(f"Обнаружение аномалий (режим {detector_mode}, порог {detector_threshold}%)...")
        detector = AnomalyDetectorFactory.create(mode=detector_mode, threshold_percent=detector_threshold)
        detected_anomaly_indices = detector.detect(real_data, predicted_data)
        _log.info(f"Обнаружено {len(detected_anomaly_indices)} точек аномалий")
        if ultra_debug:
            _log.debug(f"Индексы обнаруженных аномалий: {detected_anomaly_indices}")
        
        _log.info("Оценка эффективности обнаружения аномалий...")
        anomaly_evaluator = AnomalyDetectionEvaluator(real_anomaly_ranges)
        anomaly_evaluator.set_detected_anomalies(detected_anomaly_indices)
        anomaly_evaluator.evaluate(total_length, min_coverage_for_segment=0.01)
        anomaly_evaluator.print_metrics(ultra_debug=ultra_debug)
    else:
        _log.error(f"Неизвестный режим тестирования: {test_mode}")
        return
    
    _log.info("Готово")


if __name__ == "__main__":
    evaluation_workflow()
