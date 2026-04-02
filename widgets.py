from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QLabel


class BlinkLabel(QLabel):
    """Виджет QLabel, который интервально мигает, переключая цвет текста между заданным и прозрачным"""
    def __init__(self, text, color = "red", interval = 500, parent=None):
        super().__init__(text, parent)

        # текущий цвет текста
        self._color = color
        # цвет при видимом состоянии
        self._visible_color = color
        # цвет при скрытом состоянии
        self._hidden_color = "transparent"
        # интервал мигания
        self._interval = interval         

        # таймер для управления миганием
        self._timer = QTimer(self)
        self._timer.setInterval(self._interval)
        self._timer.timeout.connect(self._on_timeout)

        # флаг показа/скрытия текста
        self._showing = True

        # Инициализация UI
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._apply_color(self._visible_color)

    def _apply_color(self, color):
        # установка цвета текста через стиль
        self.setStyleSheet(f"color: {color};")

    def _on_timeout(self):
        # По таймеру переключаем видимость
        if self._showing:
            self._apply_color(self._hidden_color)
        else:
            self._apply_color(self._visible_color)
        self._showing = not self._showing

    def start(self):
        """Запустить мигание виджета"""
        if not self._timer.isActive():
            self._showing = True
            self._apply_color(self._visible_color)
            self._timer.start()

    def stop(self):
        """Остановить мигание виджета и сделать его видимым"""
        if self._timer.isActive():
            self._timer.stop()
        self._showing = True
        self._apply_color(self._visible_color)

    def setInterval(self, msec):
        """Изменить интервал мигания в миллисекундах"""
        self._interval = msec
        self._timer.setInterval(self._interval)

    def setColor(self, color):
        """Изменить цвет текста в видимом состоянии"""
        self._color = color
        self._visible_color = color
        if self._showing:
            self._apply_color(self._visible_color)
