import numpy as np


class AnomalyFormatter:
    """Единое форматирование сообщений об обнаружении аномалий"""
    @staticmethod
    def format_anomaly_message(anomaly_indices, real_values, pred_values, title="Аномалия обнаружена"):
        """Форматирование сообщения об обнаружении аномалии с информацией: индексов, предсказанных и реальных значений, а также процентного отклонения"""
        if not anomaly_indices or len(real_values) != len(pred_values) or len(anomaly_indices) != len(real_values):
            return None

        msg = f"{title}:\n"
        details = "idx | predicted | real | отклонение\n"

        for idx, real_val, pred_val in zip(anomaly_indices, real_values, pred_values):
            if real_val != 0 and not np.isnan(real_val):
                percent_diff = round(abs(pred_val - real_val) / abs(real_val) * 100, 2)
            else:
                percent_diff = "N/A"

            details += f"{idx} | {pred_val:.3f} | {real_val:.3f} | {percent_diff}%\n"

        return msg + details
