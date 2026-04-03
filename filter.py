import numpy as np
import pywt
from scipy import stats

from app_logging import get_logger, init_default_logging_if_needed

init_default_logging_if_needed()
_filter_log = get_logger("filter")


def apply_thresholds_wp(wp, alpha, threshold_type = "hard"):
    """
    4–5 Вычисление порогов и определение информационных компонент 
    для каждого узла дерева вейвлет-пакета (на максимальном уровне разложения).
    Пороги рассчитываются по выборочной дисперсии данных в конкретном узле.
    """
    # Берем все узлы максимального уровня, то есть полное дерево вейвлет-пакета
    nodes = wp.get_level(wp.maxlevel, order="natural")

    # Проходим по каждому узлу и применяем пороговую функцию
    for node in nodes:
        c = node.data
        K = len(c)

        if K < 2:
            _filter_log.warning("Узел пропущен из-за недостаточного количества данных (K < 2) для расчета t-распределения")
            continue

        # Выборочное стандартное отклонение
        sigma = np.std(c, ddof=1)
        if sigma == 0:
            node.data = np.zeros_like(c)
            continue

        # t-критическое значение
        df = K - 1
        t_crit = stats.t.ppf(1 - alpha / 2, df)

        # Порог t (по алгоритму: t = σ · t_{1-α/2, K-1})
        t = sigma * t_crit

        # Пороговая функция (информационные компоненты)
        if threshold_type == "soft":
            # Мягкая пороговая функция
            new_c = np.sign(c) * np.maximum(np.abs(c) - t, 0)
        elif threshold_type == "hard":
            # Жесткая пороговая функция
            new_c = np.where(np.abs(c) > t, c, 0)
        else:
            _filter_log.warning(f"Неизвестный тип пороговой функции '{threshold_type}'. Используется жесткая (hard) по умолчанию.")
            # default hard
            new_c = np.where(np.abs(c) > t, c, 0)

        node.data = new_c


def filter_data(data, wavelet = "db3", alpha = 0.05, J = 5, threshold_type = "hard"):
    """
    Алгоритм фильтрации данных с помощью вейвлет-пакетов и пороговой функции.
    Шаги алгоритма:
    1 Ввод данных, вейвлета, alpha, J и типа пороговой функции (soft или hard)
    2–3 Разложение в дерево вейвлет-пакетов до уровня J
    4 Расчет порогов по дисперсии в каждом узле
    5 Определение информационных компонент (пороговая функция)
    6 Вейвлет-восстановление данных
    """
    
    # 1 Ввод исходных данных

    if len(data) == 0:
        _filter_log.error("Пустые входные данные. Невозможно выполнить фильтрацию.")
        return []

    # Преобразуем входные данные в numpy для удобства обработки
    np_data = np.array(data, dtype=float)
    _filter_log.info(f"Начало фильтрации данных. Длина входных данных: {len(data)}, wavelet: '{wavelet}', alpha: {alpha}, maxlevel: {J}, threshold_type: '{threshold_type}'.")

    # 2–3 Разложение в вейвлет-пакеты (рекурсивно до уровня J)
    _filter_log.info(f"Начало разложения данных в дерево вейвлет-пакетов с wavelet='{wavelet}', maxlevel={J} и mode='symmetric'.")
    wp = pywt.WaveletPacket(
        data=np_data,
        wavelet=wavelet,
        maxlevel=J,
        mode="symmetric"
    )

    # 4–5 Пороги и информационные компоненты в каждом узле
    _filter_log.info(f"Рассчет порогов и определение информационных компонент для каждого узла дерева вейвлет-пакета, alpha={alpha}, threshold_type={threshold_type}.")
    apply_thresholds_wp(wp, alpha, threshold_type)

    # 6 Вейвлет-восстановление сигнала
    _filter_log.info(f"Начало вейвлет-восстановления данных после применения порогов.")
    filtered = wp.reconstruct(update=False)

    # Возвращаем данные той же длины, что и исходные
    _filter_log.info(f"Фильтрация завершена. Исходная длина данных: {len(data)}, длина отфильтрованных данных: {len(filtered)} (возвращаем длину исходных данных).")
    return filtered[:len(data)].tolist()
