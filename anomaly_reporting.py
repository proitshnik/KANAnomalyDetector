class AnomalyReportSession:
    """
    Дедупликация и формирование данных новых аномалий для логирования.
    Хранит индексы в координатах текущего массива.
    """

    def __init__(self):
        self.logged_indices = set()

    def new_indices(self, detected_indices):
        new_idx = []
        for idx in detected_indices:
            if idx not in self.logged_indices:
                new_idx.append(idx)
        self.logged_indices.update(new_idx)
        return new_idx


def safe_pick(values, indices):
    """
    Возвращает список float значений по индексам с проверками безопасности, сохраняя индексацию с добавлением None для невалидных позиций.
    """
    result = []
    for idx in indices:
        try:
            # Проверка типа индекса
            if not isinstance(idx, int):
                result.append(None)
                continue
            
            # Проверка границ
            if idx < 0 or idx >= len(values):
                result.append(None)
                continue
            
            val = values[idx]
            
            # Пропускаем None
            if val is None:
                result.append(None)
                continue
            
            # Преобразуем в float
            result.append(float(val))
        except (ValueError, TypeError):
            # Для ошибок преобразования или других исключений возвращаем None
            result.append(None)
    
    return result
