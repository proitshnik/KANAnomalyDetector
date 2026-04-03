import numpy as np

import pyqtgraph as pg

from app_logging import get_logger
import imit_mon_path

_layout_log = get_logger("layouts")

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QBoxLayout, QCheckBox, QButtonGroup, QPushButton, QTextEdit, QFileDialog, QLabel, QLineEdit, QMessageBox, QComboBox
from PyQt6.QtGui import QCursor
from widgets import BlinkLabel


class GraphLayout(QBoxLayout):
    def __init__(self, direction: 'QBoxLayout.Direction'):
        super().__init__(direction)

        self.plot_comparison = pg.PlotWidget()
        self.real_pen = None
        self.predicted_pen = None
        
        # Кэш элементов графика
        self._real_item = None
        self._predicted_item = None

        self.config_view()

        self.addWidget(self.plot_comparison)

    def config_view(self):
        self.plot_comparison.setBackground("w")
        self.plot_comparison.showGrid(x=True, y=True)
        self.plot_comparison.setLabel("left", "Monitor")
        self.plot_comparison.setLabel("bottom", "Time")
        self.plot_comparison.addLegend()
        self.real_pen = pg.mkPen(color=(0, 0, 255), width=2)
        self.predicted_pen = pg.mkPen(color=(255, 0, 0), width=2)

    def plot_real(self, time=None, monitor=None):
        if time is None:
            # Example graph data for testing
            time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if monitor is None:
            # Example graph data for testing
            monitor = [100.519, 101.018, 99.087, 99.345, 100.814, 99.915, 99.608, 100.202, 100.029, 98.572]

        self.plot_comparison.plot(time, monitor, name="Real monitor data", pen=self.real_pen)

    def plot_predicted(self, time=None, monitor=None):
        if time is None:
            # Example graph data for testing
            time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if monitor is None:
            # Example graph data for testing
            monitor = [101.519, 102.018, 100.087, 100.345, 101.814, 100.915, 100.608, 101.202, 101.029, 99.572]

        self.plot_comparison.plot(time, monitor, name="Predicted KAN data", pen=self.predicted_pen)

    def clear_plot(self):
        """Полная очистка графика и сброс кэша элементов."""
        self.plot_comparison.clear()
        self._real_item = None
        self._predicted_item = None
        self.plot_comparison.addLegend()

    def update_plot(self, time, real, predicted, anomalies=None, show_values=False):
        # Проверяем, есть ли уже линии на графике
        items = self.plot_comparison.listDataItems()
        
        # Удаляем старые отметки аномалий (все элементы кроме основных линий графиков)
        for item in items:
            if item != self._real_item and item != self._predicted_item:
                self.plot_comparison.removeItem(item)
        
        if self._real_item is None or self._predicted_item is None:
            # Первый раз
            self._real_item = self.plot_comparison.plot(time, real, name="Real monitor data", pen=self.real_pen)
            self._predicted_item = self.plot_comparison.plot(time, predicted, name="Predicted KAN data", pen=self.predicted_pen)
        else:
            # Обновляем
            self._real_item.setData(time, real)
            self._predicted_item.setData(time, predicted)
        
        # Добавляем визуальные отметки аномалий
        if anomalies:
            for idx in anomalies:
                if idx < len(time):
                    # Вертикальная линия на месте аномалии
                    self.plot_comparison.addLine(x=time[idx], pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
                    # Отметка на реальном значении
                    if idx < len(real) and not np.isnan(real[idx]):
                        self.plot_comparison.plot([time[idx]], [real[idx]], pen=None, symbol='o', symbolBrush='r', symbolSize=15)
                    # Отметка на предсказанном значении
                    if idx < len(predicted) and not np.isnan(predicted[idx]):
                        self.plot_comparison.plot([time[idx]], [predicted[idx]], pen=None, symbol='x', symbolBrush='r', symbolSize=15)

class ControlLayout(QBoxLayout):
    def __init__(self, direction: 'QBoxLayout.Direction'):
        super().__init__(direction)

        # Выпадающий список режимов обнаружения аномалий
        self.anomaly_mode_combo = QComboBox()
        self.anomaly_mode_combo.addItem("Процент отклонения")
        self.addWidget(QLabel("Режим обнаружения аномалий:"))
        self.addWidget(self.anomaly_mode_combo)

        # Поле для процента отклонения
        self.percent_label = QLabel("Порог отклонения (%)")
        self.percent_edit = QLineEdit("1")
        self.addWidget(self.percent_label)
        self.addWidget(self.percent_edit)

        # Параметры file mode
        self.start_index_label = QLabel("Индекс начала данных:")
        self.end_index_label = QLabel("Индекс конца данных:")
        self.extra_num_label = QLabel("Доп. количество точек:")
        self.start_index_edit = QLineEdit("0")
        self.end_index_edit = QLineEdit("0")
        self.extra_num_edit = QLineEdit("0")
        
        self.addWidget(self.start_index_label)
        self.addWidget(self.start_index_edit)
        self.addWidget(self.end_index_label)
        self.addWidget(self.end_index_edit)
        self.addWidget(self.extra_num_label)
        self.addWidget(self.extra_num_edit)
        
        # Фильтрация данных (с помощью вейвлет-пакетов и пороговой функции)
        self.use_filter = QCheckBox("Применить вейвлет-фильтрацию")
        self.use_filter.setChecked(False)
        self.addWidget(self.use_filter)
        
        # Выбор режима работы
        self.file_mode = QCheckBox("Режим файла")
        self.file_mode.setChecked(True)
        self.realtime_mode = QCheckBox("Режим реального времени")
        modeGroup = QButtonGroup(self)
        modeGroup.setExclusive(True)
        modeGroup.addButton(self.file_mode)
        modeGroup.addButton(self.realtime_mode)

        # Начать работу
        self.start_button = QPushButton("Начать работу!")
        self.start_button.setToolTip("Запуск проверки аномалий")
        self.start_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.start_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; border: none; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #5CBF60; }
            QPushButton:pressed { background-color: #3D8B40; }
            QPushButton:disabled { background-color: #CCCCCC; color: #888888; }
        """)
        self.start_button.clicked.connect(self.on_start_button_clicked)
        
        # Остановить работу
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setToolTip("Остановить текущую работу")
        self.stop_button.setEnabled(False)
        self.stop_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.stop_button.setStyleSheet("""
            QPushButton { background-color: #DB3C30; color: white; font-weight: bold; border: none; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #E55C50; }
            QPushButton:pressed { background-color: #AB2C20; }
            QPushButton:disabled { background-color: #CCCCCC; color: #888888; }
        """)
        self.stop_button.clicked.connect(self.on_stop_button_clicked)

        # Кнопка для переключения авто-следования за графиком
        self.follow_button = QPushButton("Следовать за графиком")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.follow_button.setStyleSheet("""
            QPushButton { border: none; padding: 5px; border-radius: 3px; }
            QPushButton:checked { background-color: #2E7D8E; color: white; font-weight: bold; }
            QPushButton:checked:hover { background-color: #3A99B0; }
            QPushButton:checked:pressed { background-color: #1F5560; }
            QPushButton:!checked { background-color: #555555; color: white; font-weight: bold; }
            QPushButton:!checked:hover { background-color: #666666; }
            QPushButton:!checked:pressed { background-color: #444444; }
        """)
        self.follow_button.clicked.connect(self.on_follow_button_clicked)

        # Предупреждение об аномалии
        self.attention_text = BlinkLabel("ANOMALY ATTENTION!")
        self.attention_text.setVisible(False)
        
        # Фиксация аномалии
        self.fix_anomaly_button = QPushButton("Зафиксировать аномалию")
        self.fix_anomaly_button.setToolTip("Остановить мигание аномалии (требуется подтверждение)")
        self.fix_anomaly_button.clicked.connect(self.confirm_fix_anomaly)
        self.fix_anomaly_button.setVisible(False)

        self.addWidget(self.follow_button, stretch=1)
        self.addWidget(self.file_mode, stretch=1)
        self.addWidget(self.realtime_mode, stretch=1)
        self.addWidget(self.start_button, stretch=1)
        self.addWidget(self.stop_button, stretch=1)
        self.addWidget(self.attention_text, stretch=1)
        self.addWidget(self.fix_anomaly_button, stretch=1)

        # callback для запуска проверки аномалий, который будет установлен из app_window и вызывать соответствующий метод для запуска проверки аномалий в основном окне
        self.on_evaluate_requested = None
        # callback для остановки работы, который будет установлен из app_window и вызывать соответствующий метод для остановки работы в основном окне
        self.on_stop_requested = None
        # callback для фиксации аномалии пользователем, который будет установлен из app_window и вызывать соответствующий метод для фиксации аномалии в основном окне
        self.fix_anomaly_callback = None

        self._init_mode_signals()
        self._init_anomaly_signals()
        
    def _init_mode_signals(self):
        # Инициализация сигналов для режимов работы программы
        self.file_mode.stateChanged.connect(self.on_mode_changed)
        self.realtime_mode.stateChanged.connect(self.on_mode_changed)
        self.on_mode_changed()

    def _init_anomaly_signals(self):
        # Инициализация сигналов для режима аномалий
        self.anomaly_mode_combo.currentTextChanged.connect(self.on_anomaly_mode_changed)
        self.on_anomaly_mode_changed(self.anomaly_mode_combo.currentText())

    def on_mode_changed(self):
        show = self.file_mode.isChecked()
        self.toggle_bounds_fields(show)

    def on_anomaly_mode_changed(self, mode):
        # Показывать поле процента только для режима "Процент отклонения"
        show_percent = (mode == "Процент отклонения")
        self.percent_label.setVisible(show_percent)
        self.percent_edit.setVisible(show_percent)

    def toggle_bounds_fields(self, show):
        self.start_index_edit.setVisible(show)
        self.end_index_edit.setVisible(show)
        self.extra_num_edit.setVisible(show)
        self.start_index_label.setVisible(show)
        self.end_index_label.setVisible(show)
        self.extra_num_label.setVisible(show)

    def show_anomaly_controls(self, show):
        self.attention_text.setVisible(show)
        self.fix_anomaly_button.setVisible(show)
        if not show:
            self.attention_text.stop()

    def confirm_fix_anomaly(self):
        reply = QMessageBox.question(None, 
                                    "Подтвердите", "Вы уверены, что хотите зафиксировать аномалию и остановить мигание?", 
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes and self.fix_anomaly_callback:
            self.fix_anomaly_callback()

    def on_start_button_clicked(self):
        if self.on_evaluate_requested:
            anomaly_mode = self.anomaly_mode_combo.currentText()
            percent = float(self.percent_edit.text()) if anomaly_mode == "Процент отклонения" else None
            use_filter = self.use_filter.isChecked()
            self.on_evaluate_requested(
                self._get_safe_index(self.start_index_edit.text()),
                self._get_safe_index(self.end_index_edit.text()),
                self._get_safe_extra(self.extra_num_edit.text()),
                anomaly_mode,
                percent,
                use_filter
            )
    
    def on_stop_button_clicked(self):
        _layout_log.warning("Остановка работы по запросу пользователя...")
        if self.on_stop_requested:
            self.on_stop_requested()

    def on_follow_button_clicked(self):
        # callback для переключения авто-следования за графиком
        # переопределен в app_window
        pass

    def _get_safe_index(self, value):
        try:
            idx = int(value)
            if idx < 0:
                return 0
            return idx
        except Exception:
            return 0

    def _get_safe_extra(self, value):
        try:
            extra = int(value)
            if extra < 0:
                return 0
            return extra
        except Exception:
            return 0

class ServiceLayout(QBoxLayout):
    def __init__(self, direction: 'QBoxLayout.Direction'):
        super().__init__(direction)

        # Загрузка нейронной сети
        self.kan_file_label = QLabel("Файл модели\n не выбран")
        self.kan_file_label.setWordWrap(True)
        self.kan_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_kan_button = QPushButton("Загрузить модель KAN")
        self.load_kan_button.clicked.connect(self.load_kan_file)

        # Загрузка данных нейтронного монитора
        self.data_file_label = QLabel("Файл данных\n не выбран")
        self.data_file_label.setWordWrap(True)
        self.data_file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_data_button = QPushButton("Загрузить данные")
        self.load_data_button.clicked.connect(self.load_data_file)

        self.addWidget(self.kan_file_label)
        self.addWidget(self.load_kan_button)
        self.addWidget(self.data_file_label)
        self.addWidget(self.load_data_button)

        self.model_path = None
        self.data_path = None

    def load_kan_file(self):
        fname, _filter = QFileDialog.getOpenFileName(
            None,
            "Выберите файл модели",
            "",
            "Файлы модели (*.pt *.bin);;Все файлы (*.*)"
        )
        if fname:
            self.kan_file_label.setText(f"Выбран файл:\n{fname}")
            self.model_path = fname
            _layout_log.info("Выбран файл модели: %s", fname)

    def load_data_file(self):
        fname, _filter = QFileDialog.getOpenFileName(
            None,
            "Выберите файл данных",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if fname:
            self.data_file_label.setText(f"Выбран файл:\n{fname}")
            self.data_path = fname
            _layout_log.info("Выбран файл данных: %s", fname)

class OutputLayout(QBoxLayout):
    def __init__(self, direction: 'QBoxLayout.Direction'):
        super().__init__(direction)

        # Код временно закомментирован из-за логирования в файл, взятое за основу, возможно позже можно будет регулировать дублирование в консоль через чекбокс
        # self.duplicate = True

        self.outputWidget = QTextEdit()
        self.outputWidget.setReadOnly(True)

        # self.duplicate_chk = QCheckBox("Дублировать в консоль")
        # self.duplicate_chk.setChecked(True)
        # self.duplicate_chk.stateChanged.connect(self.on_duplicate_changed)

        self.addWidget(self.outputWidget)
        # self.addWidget(self.duplicate_chk)

    # def on_duplicate_changed(self, state):
    #     self.duplicate = (state == Qt.CheckState.Checked)

    def print(self, text):
        """Дополнительный метод для логирования текста в виджет и файл, который также может дублировать текст в консоль, если включена соответствующая опция, а сам виджет заполняется через attach_gui_log_handler в приложении"""
        get_logger("ui").info(text.rstrip("\n"))


class ImitMonPathsLayout(QBoxLayout):
    """Слой для управления путями мониторинга и имитации в реальном времени и режимом имитации"""
    
    # Сигналы для уведомления об изменении
    # monitor_path, source_path
    paths_changed = pyqtSignal(str, str)
    # use_imitation
    imitation_toggled = pyqtSignal(bool)
    
    def __init__(self, direction: 'QBoxLayout.Direction', 
                default_monitor_path=None,
                default_source_path=None):
        super().__init__(direction)
        
        if default_monitor_path is None:
            default_monitor_path = imit_mon_path.DEFAULT_MONITOR_REALTIME_FILE_PATH
        if default_source_path is None:
            default_source_path = imit_mon_path.DEFAULT_SOURCE_REALTIME_FILE_PATH
        
        self.default_monitor_path = default_monitor_path
        self.default_source_path = default_source_path
        
        title_label = QLabel("Источники режима реального времени")
        title_label.setWordWrap(True)
        title_font = title_label.font()
        title_font.setBold(True)
        title_label.setFont(title_font)
        self.addWidget(title_label)
        
        # Чекбокс для включения/отключения встроенной имитации
        self.imitation_check = QCheckBox("Использовать имитацию")
        self.imitation_check.setChecked(imit_mon_path.get_imitation_enabled())
        self.imitation_check.stateChanged.connect(self._on_imitation_changed)
        self.addWidget(self.imitation_check)
        
        # Путь для файла мониторинга
        monitor_label = QLabel("Мониторинг файла:")
        self.addWidget(monitor_label)
        
        self.monitor_browse_btn = QPushButton("...")
        self.monitor_browse_btn.setMaximumWidth(40)
        self.monitor_browse_btn.clicked.connect(self._select_monitor_file)
        self.monitor_input = QLineEdit()
        self.monitor_input.setPlaceholderText("Путь мониторинга")
        self.monitor_input.setText(default_monitor_path)
        self.monitor_input.setMaximumWidth(220)
        self.monitor_input.textChanged.connect(self._on_monitor_edited)
        
        monitor_row = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        monitor_row.addWidget(self.monitor_browse_btn)
        monitor_row.addWidget(self.monitor_input)
        self.addLayout(monitor_row)
        
        #Путь для файла источника данных (активно только при включенной встроенной имитации)
        source_label = QLabel("Источник встроенной имитации:")
        self.source_label = source_label
        self.addWidget(source_label)
        
        self.source_browse_btn = QPushButton("...")
        self.source_browse_btn.setMaximumWidth(40)
        self.source_browse_btn.clicked.connect(self._select_source_file)
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Путь источника")
        self.source_input.setText(default_source_path)
        self.source_input.setMaximumWidth(220)
        self.source_input.textChanged.connect(self._on_source_edited)
        
        source_row = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        source_row.addWidget(self.source_browse_btn)
        source_row.addWidget(self.source_input)
        self.source_layout = source_row
        self.addLayout(source_row)
        
        # Сброс до значений по умолчанию
        reset_row = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        self.reset_button = QPushButton("Сброс")
        self.reset_button.setMaximumWidth(60)
        self.reset_button.clicked.connect(self._reset_to_defaults)
        reset_row.addStretch()
        reset_row.addWidget(self.reset_button)
        self.addLayout(reset_row)
        
        # Обновляем активность полей в зависимости от состояния имитации
        self._update_source_field_enabled()
        
        self.addStretch()
    
    def _select_monitor_file(self):
        """Диалог выбора файла мониторинга"""
        fname, _filter = QFileDialog.getOpenFileName(
            None,
            "Выберите файл мониторинга",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if fname:
            self.monitor_input.setText(fname)
            _layout_log.info("Выбран файл мониторинга: %s", fname)
    
    def _select_source_file(self):
        """Диалог выбора файла источника"""
        fname, _filter = QFileDialog.getOpenFileName(
            None,
            "Выберите файл источника данных для имитации",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if fname:
            self.source_input.setText(fname)
            _layout_log.info("Выбран файл источника: %s", fname)
    
    def _on_monitor_edited(self, text):
        """При изменении пути мониторинга"""
        if text:
            imit_mon_path.set_monitor_path(text)
            self.paths_changed.emit(text, imit_mon_path.get_source_path())
    
    def _on_source_edited(self, text):
        """При изменении пути источника"""
        if text:
            imit_mon_path.set_source_path(text)
            self.paths_changed.emit(imit_mon_path.get_monitor_path(), text)
    
    def _on_imitation_changed(self, state):
        """Переключение имитации"""
        # state == 2 означает, что чекбокс включен
        use_imitation = (state == 2)
        imit_mon_path.set_imitation_enabled(use_imitation)
        self.imitation_toggled.emit(use_imitation)
        self._update_source_field_enabled()
        _layout_log.info(f"Имитация: {'ВКЛЮЧЕНА' if use_imitation else 'ОТКЛЮЧЕНА'}")
    
    def _reset_to_defaults(self):
        """Сбросить пути на значения по умолчанию"""
        imit_mon_path.reset_to_defaults()
        self.monitor_input.setText(imit_mon_path.DEFAULT_MONITOR_REALTIME_FILE_PATH)
        self.source_input.setText(imit_mon_path.DEFAULT_SOURCE_REALTIME_FILE_PATH)
        self.imitation_check.setChecked(True)
        _layout_log.info("Пути сброшены на значения по умолчанию")
    
    def _update_source_field_enabled(self):
        """Обновить активность поля источника имитации в зависимости от состояния имитации"""
        is_imitation_enabled = self.imitation_check.isChecked()
        self.source_label.setEnabled(is_imitation_enabled)
        self.source_input.setEnabled(is_imitation_enabled)
        self.source_browse_btn.setEnabled(is_imitation_enabled)
