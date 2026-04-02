import datetime

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication, QBoxLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from anomaly_factory import AnomalyDetectorFactory
from app_logging import get_logger, log_anomaly
from core import KANAnomalyDetector
import imit_mon_path
from layouts import ControlLayout, GraphLayout, OutputLayout, ServiceLayout, ImitMonPathsLayout
from workers import FileModeWorker, RealTimeImitateWorker, RealTimeMonitorWorker, SetupWorker

from version import PROGRAM_VERSION

_ui_log = get_logger("ui")


class MainWindow(QWidget):
    # Сигналы для взаимодействия между потоками и UI
    # Сигнал для запуска мигания при обнаружении аномалии
    blink_start_signal = pyqtSignal()
    # Сигнал для остановки мигания при фиксации аномалии пользователем
    blink_stop_signal = pyqtSignal()
    # Сигнал для обновления графика с новыми данными (time_arr, real, pred, anomalies)
    graph_update_signal = pyqtSignal(list, list, list, list)
    # Сигнал прихода аномального события для последующего логирования
    output_signal = pyqtSignal(str)
    # Сигнал для добавления сообщения в лог UI
    log_gui_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.mainLayout = QVBoxLayout()

        # Зона графика и управления
        self.graphZoneLayout = QHBoxLayout()
        # Контейнер для GraphLayout для потенциального управления
        self.graphWidget = QWidget()
        self.graphLayout = GraphLayout(QBoxLayout.Direction.TopToBottom)
        self.graphWidget.setLayout(self.graphLayout)
        # Контейнер для ControlLayout для потенциального управления
        self.controlWidget = QWidget()
        self.controlLayout = ControlLayout(QBoxLayout.Direction.TopToBottom)
        self.controlWidget.setLayout(self.controlLayout)
        # Добавляем виджеты графика и управления в зону графика
        self.graphZoneLayout.addWidget(self.graphWidget)
        self.graphZoneLayout.addWidget(self.controlWidget)

        # Зона сервисных элементов (загрузка файлов, настройка путей мониторинга и имитации, вывод логов)
        self.serviceZoneLayout = QHBoxLayout()
        # Контейнер для ServiceLayout, чтобы его можно было ограничить в размере
        self.serviceWidget = QWidget()
        self.serviceWidget.setMaximumWidth(220)
        self.serviceLayout = ServiceLayout(QBoxLayout.Direction.TopToBottom)
        self.serviceWidget.setLayout(self.serviceLayout)
        # Контейнер для ImitMonPathsLayout, чтобы его можно было скрыть/показать целиком при переключении режимов
        self.imitMonPathsWidget = QWidget()
        self.imitMonPathsWidget.setMaximumWidth(220)
        self.imitMonPathsLayout = ImitMonPathsLayout(QBoxLayout.Direction.TopToBottom)
        self.imitMonPathsWidget.setLayout(self.imitMonPathsLayout)
        # Контейнер для OutputLayout для потенциального управления
        self.outputWidget = QWidget()
        self.outputLayout = OutputLayout(QBoxLayout.Direction.TopToBottom)
        self.outputWidget.setLayout(self.outputLayout)
        # Добавляем виджеты сервисной зоны
        self.serviceZoneLayout.addWidget(self.serviceWidget)
        self.serviceZoneLayout.addWidget(self.imitMonPathsWidget)
        self.serviceZoneLayout.addWidget(self.outputWidget)

        # Сборка главного layout
        self.mainLayout.addLayout(self.graphZoneLayout)
        self.mainLayout.addLayout(self.serviceZoneLayout)

        self.setLayout(self.mainLayout)
        self.setWindowTitle(f"Anomaly Detector {PROGRAM_VERSION}")
        self.setMinimumSize(1000, 800)

        self.anomaly_active = False

        # callback для запуска проверки аномалий
        self.controlLayout.on_evaluate_requested = self.run_evaluate
        # callback для остановки работы
        self.controlLayout.on_stop_requested = self.stop_work
        # callback для фиксации аномалии пользователем
        self.controlLayout.fix_anomaly_callback = self.fix_anomaly

        # callbacks для переключения режимов
        self.controlLayout.file_mode.stateChanged.connect(self.on_mode_changed)
        self.controlLayout.realtime_mode.stateChanged.connect(self.on_mode_changed)

        # callbacks для обновления путей и переключения имитации
        self.imitMonPathsLayout.paths_changed.connect(self.on_monitoring_paths_changed)
        self.imitMonPathsLayout.imitation_toggled.connect(self.on_imitation_toggled)
        
        # Изначально скрываем layout мониторинга, так как по умолчанию выбран файловый режим
        self._hide_monitoring_layout()

        # callback для кнопки следования за графиком
        self.controlLayout.follow_button.clicked.connect(self.on_follow_button_clicked)
        self.auto_follow_graph = True

        self.last_time_arr = []
        self.last_real = []
        self.last_pred = []
        self.last_anomalies = []

        self.realtime_mode = False
        self.file_setup_worker = None
        self.file_mode_worker = None
        self.realtime_imitate_worker = None
        self.realtime_monitor_worker = None
        self.realtime_setup_worker = None
        self.is_running = False
        self.stop_requested = False
        
        # callbacks для сигналов от рабочих потоков
        self.blink_start_signal.connect(self._on_blink_start)
        self.blink_stop_signal.connect(self._on_blink_stop)
        self.graph_update_signal.connect(self._on_graph_update_signal)
        self.output_signal.connect(self._on_output_signal)
        self.log_gui_message.connect(self.append_log_to_output)

    def append_log_to_output(self, text):
        w = self.outputLayout.outputWidget
        cursor = w.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        w.setTextCursor(cursor)
        w.insertPlainText(text)

    def on_follow_button_clicked(self):
        self.auto_follow_graph = self.controlLayout.follow_button.isChecked()
        _ui_log.info(f"Авто-следование за графиком {'ВКЛЮЧЕНО' if self.auto_follow_graph else 'ОТКЛЮЧЕНО'}")
        self.update_graph_view()

    def on_mode_changed(self):
        """Callback для переключения между файловым режимом и режимом реального времени"""
        # Показываем/скрываем поля для загрузки файлов в зависимости от выбранного режима
        show_file_fields = self.controlLayout.file_mode.isChecked()
        self.serviceLayout.data_file_label.setVisible(show_file_fields)
        self.serviceLayout.load_data_button.setVisible(show_file_fields)
        
        # Показываем/скрываем layout мониторинга в зависимости от режима
        if not show_file_fields:
            self._show_monitoring_layout()
        else:
            self._hide_monitoring_layout()

    def _show_monitoring_layout(self):
        """Показать виджет ImitMonPathsLayout для путей файлов мониторинга и источника имитации полностью"""
        self.imitMonPathsWidget.show()

    def _hide_monitoring_layout(self):
        """Скрыть виджет ImitMonPathsLayout для путей файлов мониторинга и источника имитации полностью"""
        self.imitMonPathsWidget.hide()

    def on_monitoring_paths_changed(self, monitor_path, source_path):
        """Callback для обновления путей при их изменении в интерфейсе"""
        imit_mon_path.set_paths(monitor_path, source_path)
        _ui_log.info(f"Пути мониторинга обновлены - Monitor: {monitor_path}, Source: {source_path}")

    def on_imitation_toggled(self, enabled):
        """Callback для переключения имитации"""
        imit_mon_path.set_imitation_enabled(enabled)
        _ui_log.info(f"Имитация: {'ВКЛЮЧЕНА' if enabled else 'ОТКЛЮЧЕНА'}")

    def update_graph_view(self):
        if self.last_time_arr and self.last_real and self.last_pred:
            self.graphLayout.update_plot(
                self.last_time_arr,
                self.last_real,
                self.last_pred,
                anomalies=self.last_anomalies,
                show_values=False,
            )
            if self.auto_follow_graph:
                vb = self.graphLayout.plot_comparison.getViewBox()
                vb.enableAutoRange(axis=vb.XAxis, enable=False)
                right = len(self.last_time_arr)
                left = max(0, right - 16)
                vb.setXRange(left, right, padding=0)
            QApplication.processEvents()

    def _has_active_threads(self):
        """Проверяет, есть ли активные потоки"""
        return (
            (self.file_setup_worker and self.file_setup_worker.isRunning()) or
            (self.file_mode_worker and self.file_mode_worker.isRunning()) or
            (self.realtime_setup_worker and self.realtime_setup_worker.isRunning()) or
            (self.realtime_imitate_worker and self.realtime_imitate_worker.isRunning()) or
            (self.realtime_monitor_worker and self.realtime_monitor_worker.isRunning())
        )

    def _start_mode(self):
        """Устанавливает флаг работы и сбрасывает флаг запроса остановки при запуске режима"""
        self.is_running = True
        self.stop_requested = False

    def _restore_buttons(self):
        """Восстанавливает состояние кнопок до начального: включает кнопку 'Запустить' и отключает кнопку 'Остановить'"""
        self.controlLayout.start_button.setEnabled(True)
        self.controlLayout.stop_button.setEnabled(False)

    def _start_buttons(self):
        """Устанавливает состояние кнопок при запуске режима: отключает кнопку 'Запустить' и включает кнопку 'Остановить'"""
        self.controlLayout.start_button.setEnabled(False)
        self.controlLayout.stop_button.setEnabled(True)

    def _get_model_config(self):
        """Возвращает конфигурацию модели"""
        return {
            "input_length": 288,
            "output_length": 10,
            "test_size": 1.0,
            "step": 1,
            "device": "cpu",
            "width": [288, 144, 10],
            "grid": 5,
        }

    def _get_processing_config(self):
        """Возвращает конфигурацию обработки данных для файлового режима"""
        return {
            "usage_volume": 1.0,
            "shuffle": False,
            "normalize_first": False,
            "normalize_sequence": True,
        }

    def _create_anomaly_and_model(self, data_path, anomaly_mode, percent, model_path, prepare_data=False):
        """
        Инициализирует модуль аномалий и модель для обнаружения аномалий.
        Если prepare_data True, выполняет полную подготовку данных для файлового режима.
        """
        config = self._get_model_config()
        anomaly_module = AnomalyDetectorFactory.create(anomaly_mode, threshold_percent=percent)
        model = KANAnomalyDetector(
            data_path,
            input_length=config["input_length"],
            output_length=config["output_length"],
            test_size=config["test_size"],
            step=config["step"],
            device=config["device"],
        )
        
        if prepare_data:
            # Для файлового режима полная подготовка данных
            processing_config = self._get_processing_config()
            model.load_data()
            model.split_and_normalize(
                usage_volume=processing_config["usage_volume"], 
                normalize_first=processing_config["normalize_first"]
            )
            model.prepare_data(
                shuffle=processing_config["shuffle"], 
                normalize_sequence=processing_config["normalize_sequence"]
            )
        
        model.build_model(width=config["width"], grid=config["grid"])
        model.load_model(model_path)
        return anomaly_module, model, config

    def _init_graph_and_buffers(self):
        """Очищает график и сбрасывает буферы данных перед запуском нового режима"""
        self.graphLayout.clear_plot()
        self.last_time_arr = []
        self.last_real = []
        self.last_pred = []
        self.last_anomalies = []

    def run_evaluate(self, start_index, end_index, extra_num, anomaly_mode, percent):
        """Запуск проверки аномалий с заданными параметрами"""
        
        # Проверяем, что все потоки завершились
        if self._has_active_threads():
            _ui_log.error("Ожидание завершения предыдущей операции...")
            return

        # Проверяем выбранный режим и необходимые файлы
        if self.controlLayout.realtime_mode.isChecked():
            self.realtime_mode = True
            if not self.serviceLayout.model_path:
                _ui_log.error("Режим реального времени: не выбран файл модели")
                return
        else:
            self.realtime_mode = False
            if not self.serviceLayout.data_path or not self.serviceLayout.model_path:
                _ui_log.error("Режим файла: не выбраны файл данных или модели")
                return

        self._start_buttons()
        _ui_log.info("Проверка аномалий запущена")

        if self.anomaly_active:
            self.controlLayout.show_anomaly_controls(True)
            self.controlLayout.attention_text.start()

        self.stop_requested = False

        if self.controlLayout.realtime_mode.isChecked():
            if self.file_mode_worker and self.file_mode_worker.isRunning():
                self.stop_requested = True

            self.serviceLayout.data_file_label.setVisible(False)
            self.serviceLayout.load_data_button.setVisible(False)
            self.start_realtime_monitoring(anomaly_mode, percent)
        else:
            self.stop_requested = True

            self.serviceLayout.data_file_label.setVisible(True)
            self.serviceLayout.load_data_button.setVisible(True)
            self.start_file_mode(start_index, end_index, extra_num, anomaly_mode, percent)

    def start_file_mode(self, start_index, end_index, extra_num, anomaly_mode, percent):
        """Запуск проверки аномалий в режиме файла с заданными параметрами"""
        self._start_mode()
        self._init_graph_and_buffers()

        _ui_log.info(f"Запуск файлового режима: {datetime.datetime.now().isoformat()}")
        _ui_log.info(f"Способ проверки: {anomaly_mode}, порог={percent}%")
        _ui_log.info(f"Модель: {self.serviceLayout.model_path}")

        # Setup worker для инициализации модели и модуля аномалий в режиме файла с подготовкой данных
        self.file_setup_worker = SetupWorker(
            realtime_setup_fn=lambda: self._create_anomaly_and_model(
                self.serviceLayout.data_path, anomaly_mode, percent, self.serviceLayout.model_path, prepare_data=True
            )
        )

        def on_file_ready(anomaly_module, model, config):
            """Callback при успешной инициализации модели в режиме файла"""
            if self.stop_requested:
                _ui_log.warning("Инициализация отменена по запросу пользователя")
                self._restore_buttons()
                return

            self.file_mode_worker = FileModeWorker(
                model, anomaly_module, config, start_index, end_index, extra_num
            )
            self.file_mode_worker.stop_requested = lambda: self.stop_requested
            self.file_mode_worker.progress_signal.connect(self.on_progress_update)
            self.file_mode_worker.finished_signal.connect(self.on_file_mode_finished)
            self.file_mode_worker.start()

        def on_file_error(msg):
            """Callback при ошибке инициализации модели в режиме файла"""
            _ui_log.error(f"Ошибка инициализации модели: {msg}")
            self._restore_buttons()

        self.file_setup_worker.ready_signal.connect(on_file_ready)
        self.file_setup_worker.error_signal.connect(on_file_error)
        self.file_setup_worker.start()

    def start_realtime_monitoring(self, anomaly_mode, percent):
        """Запуск проверки аномалий в режиме реального времени с заданными параметрами"""
        source_file = imit_mon_path.get_source_path()
        monitor_file = imit_mon_path.get_monitor_path()
        interval_sec = 20

        self._start_mode()
        self._init_graph_and_buffers()

        _ui_log.info(f"Запуск реального времени: {datetime.datetime.now().isoformat()}")
        _ui_log.info(f"Способ проверки: {anomaly_mode}, порог={percent}%")
        _ui_log.info(f"Модель: {self.serviceLayout.model_path}")

        # Setup worker для инициализации модели и модуля аномалий в режиме файла без подготовки данных
        self.realtime_setup_worker = SetupWorker(
            realtime_setup_fn=lambda: self._create_anomaly_and_model(None, anomaly_mode, percent, self.serviceLayout.model_path)
        )

        def on_ready(anomaly_module, model, config):
            """Callback при успешной инициализации модели в режиме реального времени"""
            if self.stop_requested:
                _ui_log.warning("Инициализация отменена по запросу пользователя")
                self._restore_buttons()
                return

            # Запускаем имитацию только если она включена
            if imit_mon_path.get_imitation_enabled():
                self.realtime_imitate_worker = RealTimeImitateWorker(
                    source_file, monitor_file, config["input_length"], interval_sec
                )
                self.realtime_imitate_worker.stop_requested = lambda: self.stop_requested
                self.realtime_imitate_worker.finished_signal.connect(self.on_imitate_finished)
                self.realtime_imitate_worker.start()
                _ui_log.info("Встроенная имитация поступления данных запущена")
            else:
                _ui_log.info("Встроенная имитация отключена, используется работа с другим поставщиком данных")

            self.realtime_monitor_worker = RealTimeMonitorWorker(
                model, anomaly_module, monitor_file, config["input_length"], config["output_length"]
            )
            self.realtime_monitor_worker.stop_requested = lambda: self.stop_requested
            self.realtime_monitor_worker.graph_update_signal.connect(self._on_graph_update_signal)
            self.realtime_monitor_worker.output_signal.connect(self._on_output_signal)
            self.realtime_monitor_worker.blink_start_signal.connect(self._on_blink_start)
            self.realtime_monitor_worker.finished_signal.connect(self.on_monitor_finished)
            self.realtime_monitor_worker.start()

        def on_error(msg):
            """Callback при ошибке инициализации модели в режиме реального времени"""
            _ui_log.error(f"Ошибка запуска realtime: {msg}")
            self._restore_buttons()

        self.realtime_setup_worker.ready_signal.connect(on_ready)
        self.realtime_setup_worker.error_signal.connect(on_error)
        self.realtime_setup_worker.start()

    def _on_blink_start(self):
        """Callback для запуска мигания при обнаружении аномалии"""
        self.controlLayout.attention_text.start()
        self.controlLayout.show_anomaly_controls(True)
        self.anomaly_active = True

    def _on_blink_stop(self):
        """Callback для остановки мигания при фиксации аномалии пользователем"""
        self.controlLayout.attention_text.stop()
        self.controlLayout.show_anomaly_controls(False)
        self.anomaly_active = False
        
    def fix_anomaly(self):
        """Callback для фиксации аномалии пользователем"""
        self.controlLayout.attention_text.stop()
        self.controlLayout.show_anomaly_controls(False)
        self.anomaly_active = False
        _ui_log.info("Аномалия зафиксирована пользователем.")

    def _on_graph_update_signal(self, time_arr, real, pred, anomalies):
        """Callback для обновления графика с новыми данными (time_arr, real, pred, anomalies)"""
        self.last_time_arr = time_arr
        self.last_real = real
        self.last_pred = pred
        self.last_anomalies = anomalies
        self.update_graph_view()

    def _on_output_signal(self, text):
        """Callback для логирования аномального события"""
        log_anomaly(_ui_log, text.rstrip())

    def on_progress_update(self, data):
        """Callback для обновления прогресса в режиме файла с данными из data, который содержит ключи graph_update, anomaly_msg, step_msg, error_msg, debug_msg, anomaly_detected"""
        if "graph_update" in data:
            self.last_time_arr = data["time_arr"]
            self.last_real = data["real"]
            self.last_pred = data["pred"]
            self.last_anomalies = data["anomalies"]
            self.update_graph_view()

        if "anomaly_msg" in data:
            log_anomaly(_ui_log, data["anomaly_msg"].rstrip())

        if "step_msg" in data:
            _ui_log.info(data["step_msg"])
            
        if "error_msg" in data:
            _ui_log.error(data["error_msg"])
            
        if "debug_msg" in data:
            _ui_log.debug(data["debug_msg"])

        if "anomaly_detected" in data and data["anomaly_detected"]:
            self.controlLayout.attention_text.start()
            self.controlLayout.show_anomaly_controls(True)
            self.anomaly_active = True
            
    def on_imitate_finished(self):
        """Callback при завершении работы встроенной имитации"""
        if self.stop_requested:
            _ui_log.warning("Имитация остановлена пользователем по запросу остановки режима реального времени")
        else:
            _ui_log.info("Имитация завершена")

    def on_monitor_finished(self):
        """Callback при завершении работы мониторинга в режиме реального времени"""
        self.is_running = False
        self._restore_buttons()
        if self.stop_requested:
            _ui_log.warning("Режим реального времени остановлен пользователем")
        else:
            _ui_log.info("Режим реального времени завершен успешно")

    def on_file_mode_finished(self):
        """Callback при завершении работы в режиме файла"""
        self.is_running = False
        self._restore_buttons()
        if self.stop_requested:
            _ui_log.warning("Режим файла остановлен пользователем")
        else:
            _ui_log.info("Режим файла завершен успешно")

    def stop_work(self):
        """Callback для остановки работы по запросу пользователя - устанавливает флаг запроса остановки и отключает кнопку 'Остановить'"""
        # Отключаем кнопку "Остановить" чтобы нельзя было нажать ее несколько раз
        self.controlLayout.stop_button.setEnabled(False)
        
        self.stop_requested = True
        _ui_log.warning("Остановка работы по запросу пользователя...")

        if self.realtime_mode:
            _ui_log.warning("Режим реального времени: отправлен сигнал на остановку")
        else:
            _ui_log.warning("Режим файла: отправлен сигнал на остановку")
