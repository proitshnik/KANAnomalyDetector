import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from anomaly_formatting import AnomalyFormatter
from anomaly_reporting import AnomalyReportSession, safe_pick
from imitate import imitate_data
from monitoring import monitor_data
from filter import filter_data
from app_logging import get_logger

_worker_log = get_logger("workers")


class FileModeWorker(QThread):
    """Worker для обработки данных из файла - выполняет предсказания, обнаруживает аномалии и отправляет обновления в UI через сигналы graph_update_signal и finished_signal. Реализует логику обработки данных в заданном диапазоне с учетом дополнительных точек, а также буферизацию для отображения на графике и предотвращения повторного обнаружения одних и тех же аномалий"""
    graph_update_signal = pyqtSignal(list, list, list, list)
    output_anomaly_signal = pyqtSignal(str)
    start_anomaly_signal = pyqtSignal()
    finished_signal = pyqtSignal()

    def __init__(self, model, anomaly_module, config, start_index, end_index, extra_num, use_filter=False):
        super().__init__()
        self.model = model
        self.anomaly_module = anomaly_module
        self.config = config
        self.start_index = start_index
        self.end_index = end_index
        self.extra_num = extra_num
        # Используется в предобработке данных, здесь не используется
        self.use_filter = use_filter
        self.stop_requested = None

    def run(self):
        try:
            input_length = self.config["input_length"]
            output_length = self.config["output_length"]
            step = self.config["step"]

            predictor = self.model
            start_index = self.start_index
            end_index = self.end_index
            extra_num = self.extra_num

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
                _worker_log.info(msg)
                self.finished_signal.emit()
                return

            seq_start = (start_index - input_length) // step
            seq_end = (end_index - input_length) // step

            extra_seq_left = min(extra_num // step, seq_start)
            extra_seq_right = min(extra_num // step, predictor.len_X_test - seq_end - 1)

            left_range = max(0, seq_start - extra_seq_left)
            right_range = min(predictor.len_X_test, seq_end + extra_seq_right + 1)
            wanted_dot_num = right_range - left_range

            _worker_log.info(f"Диапазон: start_index={start_index}, end_index={end_index}, extra_num={extra_num}")
            _worker_log.info(f"Возможные индексы: {min_index} ... {max_index}")

            prediction = []
            real_output_denorm = []
            report_session = AnomalyReportSession()

            # Первый шаг
            i = left_range
            input_data_for_prediction, scaler_idfp = predictor.X_test[i], predictor.X_test_scaler[i]
            pr_temp = np.round(
                predictor.predict(
                    input_data_for_prediction,
                    denormalize=True,
                    normalize_sequence=True,
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

            time_arr = list(range(len(prediction)))
            anomaly_indices = self.anomaly_module.detect(real_output_denorm, prediction)

            self.graph_update_signal.emit(
                time_arr,
                real_output_denorm,
                prediction,
                anomaly_indices,
            )

            new_anomalies = report_session.new_indices(anomaly_indices)
            if new_anomalies:
                msg = AnomalyFormatter.format_anomaly_message(
                    new_anomalies,
                    safe_pick(real_output_denorm, new_anomalies),
                    safe_pick(prediction, new_anomalies),
                    title="Первые обнаруженные аномалии",
                )
                if msg:
                    self.output_anomaly_signal.emit(msg)
                    self.start_anomaly_signal.emit()

            _worker_log.info(f"Step {1}/{right_range-left_range}: {prediction[-1]}, {real_output_denorm[-1]}")

            # Остальные шаги
            for i in range(left_range + 1, right_range):
                if self.stop_requested and self.stop_requested():
                    _worker_log.info("Работа остановлена пользователем")
                    break

                input_data_for_prediction, scaler_idfp = predictor.X_test[i], predictor.X_test_scaler[i]
                pr_temp = np.round(
                    predictor.predict(
                        input_data_for_prediction,
                        denormalize=True,
                        normalize_sequence=True,
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

                time_arr = list(range(len(prediction)))
                anomaly_indices = self.anomaly_module.detect(real_output_denorm, prediction)

                new_indices = report_session.new_indices(anomaly_indices)
                if new_indices:
                    for idx in new_indices:
                        msg = AnomalyFormatter.format_anomaly_message(
                            [idx],
                            [real_output_denorm[idx]],
                            [prediction[idx]],
                            title=f"Аномалия в точке {idx}",
                        )
                        if msg:
                            self.output_anomaly_signal.emit(msg)
                            self.start_anomaly_signal.emit()

                self.graph_update_signal.emit(
                    time_arr,
                    real_output_denorm,
                    prediction,
                    anomaly_indices,
                )

                _worker_log.info(f"Step {i-left_range+1}/{right_range-left_range}: {prediction[-1]}, {real_output_denorm[-1]}")

                if len(prediction) >= wanted_dot_num and len(real_output_denorm) >= wanted_dot_num:
                    break

            prediction = prediction[:wanted_dot_num]
            real_output_denorm = real_output_denorm[:wanted_dot_num]
            time_arr = list(range(len(prediction)))
            anomaly_indices = self.anomaly_module.detect(real_output_denorm, prediction)

            self.graph_update_signal.emit(
                time_arr,
                real_output_denorm,
                prediction,
                anomaly_indices,
            )

            if anomaly_indices:
                msg = AnomalyFormatter.format_anomaly_message(
                    anomaly_indices,
                    [real_output_denorm[idx] for idx in anomaly_indices],
                    [prediction[idx] for idx in anomaly_indices],
                    title="Итоговый отчет по аномалиям",
                )
                if msg:
                    self.output_anomaly_signal.emit(msg)

            _worker_log.debug(
                f"Готово! Первые десять предсказаний: {safe_pick(prediction, [*range(10)])}\nРеальные: {safe_pick(real_output_denorm, [*range(10)])}"
            )
        except Exception as e:
            _worker_log.error(f"Ошибка: {str(e)}")
        finally:
            self.finished_signal.emit()


class RealTimeImitateWorker(QThread):
    """Worker для имитации данных в реальном времени из файла - читает данные из source_file и записывает их в monitor_file с заданным интервалом, отправляет сигнал при завершении"""
    finished_signal = pyqtSignal()

    def __init__(self, source_file, monitor_file, input_length, interval_sec):
        super().__init__()
        self.source_file = source_file
        self.monitor_file = monitor_file
        self.input_length = input_length
        self.interval_sec = interval_sec
        self.stop_requested = None

    def run(self):
        try:
            imitate_data(
                self.source_file,
                self.monitor_file,
                input_length=self.input_length,
                interval_sec=self.interval_sec,
                reset_file=True,
                stop_check=lambda: self.stop_requested() if self.stop_requested else False,
            )
        finally:
            self.finished_signal.emit()


class SetupWorker(QThread):
    """Worker для выполнения настройки в реальном времени - выполняет переданную функцию setup_fn, которая должна вернуть anomaly_module, model и config, и отправляет их через сигнал ready_signal. Если возникает ошибка, отправляет сообщение об ошибке через error_signal"""
    # anomaly_module, model, config
    ready_signal = pyqtSignal(object, object, object)
    error_signal = pyqtSignal(str)

    def __init__(self, setup_fn):
        super().__init__()
        self._setup_fn = setup_fn

    def run(self):
        try:
            anomaly_module, model, config = self._setup_fn()
            self.ready_signal.emit(anomaly_module, model, config)
        except Exception as e:
            self.error_signal.emit(str(e))


class RealTimeMonitorWorker(QThread):
    """Worker для мониторинга данных в реальном времени - читает данные из monitor_file, выполняет предсказания с помощью модели, обнаруживает аномалии с помощью anomaly_module и отправляет обновления в UI через сигналы graph_update_signal, output_anomaly_signal и start_anomaly_signal. Реализует буферизацию данных для отображения на графике и предотвращения повторного обнаружения одних и тех же аномалий"""
    graph_update_signal = pyqtSignal(list, list, list, list)
    output_anomaly_signal = pyqtSignal(str)
    start_anomaly_signal = pyqtSignal()
    finished_signal = pyqtSignal()

    def __init__(self, model, anomaly_module, monitor_file, input_length, output_length, use_filter=False):
        super().__init__()
        self.model = model
        self.anomaly_module = anomaly_module
        self.monitor_file = monitor_file
        self.input_length = input_length
        self.output_length = output_length
        self.use_filter = use_filter
        # callback для получения значения чекбокса фильтрации из UI
        self.use_filter_checkbox = None
        self.stop_requested = None

        self.real_output_denorm = []
        self.prediction = []
        self.report_session = AnomalyReportSession()

    def _is_filter_enabled(self):
        """Проверяет текущее состояние фильтрации"""
        if self.use_filter_checkbox is not None:
            return self.use_filter_checkbox.isChecked()
        return self.use_filter

    def run(self):
        try:
            last_len = 0

            def monitor_callback(values, mode):
                nonlocal last_len
                
                if mode == "init":
                    # Первый шаг - берем последние input_length значений и делаем предсказание
                    if len(values) >= self.input_length:
                        real_seq = np.array(values[-self.input_length:])
                        
                        # Применяем фильтрацию, если нужно
                        if self._is_filter_enabled():
                            _worker_log.debug("Фильтрация: ВКЛЮЧЕНА")
                            real_seq = filter_data(real_seq.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard")
                            real_seq = np.array(real_seq)
                        else:
                            _worker_log.debug("Фильтрация: ОТКЛЮЧЕНА")
                        
                        norm_seq, seq_scaler = self.model.normalize_sequence(real_seq)
                        pred = self.model.predict(
                            norm_seq,
                            denormalize=True,
                            normalize_sequence=True,
                            scaler=seq_scaler
                        )
                        
                        self.real_output_denorm = list(values[-self.input_length:])
                        self.prediction = list(pred)
                        last_len = len(values)
                
                elif mode == "step":
                    # Обрабатываем новые значения
                    num_new_values = len(values) - last_len
                    for i in range(num_new_values):
                        new_real_value = values[last_len + i]
                        self.real_output_denorm.append(new_real_value)

                        if len(self.real_output_denorm) >= self.input_length:
                            real_seq = np.array(self.real_output_denorm[-self.input_length:])
                            
                            # Применяем фильтрацию, если нужно
                            if self._is_filter_enabled():
                                _worker_log.debug(f"Фильтрация применена к последовательности из {len(real_seq)} точек")
                                real_seq = filter_data(real_seq.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard")
                                real_seq = np.array(real_seq)
                            
                            norm_seq, seq_scaler = self.model.normalize_sequence(real_seq)
                            pred = self.model.predict(
                                norm_seq,
                                denormalize=True,
                                normalize_sequence=True,
                                scaler=seq_scaler
                            )
                            self.prediction.append(pred[-1])
                    
                    last_len = len(values)

                # Отправляем обновления в UI
                num_real = len(self.real_output_denorm)
                num_pred = len(self.prediction)
                num_new_real = num_real - self.input_length
                total_len = self.input_length + max(num_new_real, num_pred)

                time_arr = list(range(total_len))

                real_output_denorm = list(self.real_output_denorm[: self.input_length])
                if num_new_real > 0:
                    real_output_denorm.extend(self.real_output_denorm[self.input_length :])
                if len(real_output_denorm) < total_len:
                    real_output_denorm.extend([np.nan] * (total_len - len(real_output_denorm)))

                prediction = [np.nan] * self.input_length
                prediction.extend(list(self.prediction))
                if len(prediction) < total_len:
                    prediction.extend([np.nan] * (total_len - len(prediction)))

                min_len = min(len(real_output_denorm), len(prediction), len(time_arr))
                real_output_denorm = real_output_denorm[:min_len]
                prediction = prediction[:min_len]
                time_arr = time_arr[:min_len]

                check_start = self.input_length
                check_end = len(real_output_denorm)

                # Сбор валидных пар (индекс, реальное значение, предсказанное значение)
                valid_pairs = [
                    (i, real_output_denorm[i], prediction[i])
                    for i in range(check_start, check_end)
                    if not np.isnan(real_output_denorm[i]) and not np.isnan(prediction[i])
                ]

                anomaly_indices = []
                if valid_pairs:
                    indices, real_vals, pred_vals = zip(*valid_pairs)
                    detected_indices = self.anomaly_module.detect(list(real_vals), list(pred_vals))
                    anomaly_indices = [indices[idx] for idx in detected_indices if idx < len(indices)]

                new_anomalies = self.report_session.new_indices(anomaly_indices)
                if new_anomalies:
                    for idx in new_anomalies:
                        msg = AnomalyFormatter.format_anomaly_message(
                            [idx],
                            [real_output_denorm[idx]],
                            [prediction[idx]],
                            title=f"Аномалия в точке {idx}",
                        )
                        if msg:
                            self.output_anomaly_signal.emit(msg)
                            self.start_anomaly_signal.emit()

                self.graph_update_signal.emit(time_arr, real_output_denorm, prediction, anomaly_indices)

            monitor_data(
                self.monitor_file,
                input_length=self.input_length,
                output_length=self.output_length,
                interval_sec=1,
                callback=monitor_callback,
                stop_check=lambda: self.stop_requested() if self.stop_requested else False,
            )
        finally:
            self.finished_signal.emit()
