from anomaly_detector import AnomalyDetectorModule


class AnomalyDetectorFactory:
    """Фабрика детекторов аномалий"""
    @staticmethod
    def create(mode, threshold_percent=None):
        """Создание детектора аномалий по заданному режиму и параметрам"""
        if mode == "Процент отклонения":
            if threshold_percent is None:
                threshold_percent = 1.0
            return AnomalyDetectorModule(threshold_percent=threshold_percent)

        # Заглушка на будущие режимы детектора обнаружения
        raise ValueError(f"Неизвестный режим обнаружения аномалий: {mode}")
    
    @staticmethod
    def create_default():
        """Создание детектора аномалий с настройками по умолчанию"""
        return AnomalyDetectorFactory.create("Процент отклонения", threshold_percent=1.0)

