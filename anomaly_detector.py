class AnomalyDetectorModule:
    """Класс, содержащий логику обнаружения аномалий"""
    def __init__(self, threshold_percent=1):
        self.threshold_percent = threshold_percent

    def detect(self, real, predicted):
        """
        Обнаружение аномалий на основе процента отклонения
        """
        anomalies = []
        for i, (r, p) in enumerate(zip(real, predicted)):
            if r == 0:
                continue
            deviation = abs(r - p) / abs(r) * 100
            if deviation > self.threshold_percent:
                anomalies.append(i)
        return anomalies