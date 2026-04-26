import os

import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
from kan import *

from app_logging import get_logger, init_default_logging_if_needed
from filter import filter_data

# from sympy import sympify, latex
# from IPython.display import display, Math

init_default_logging_if_needed()
_log = get_logger("core")


class KANAnomalyDetector:
  def __init__(self, file_path, input_length=1440, output_length=60, test_size=0.2, step=1000, scaler=None, device=torch.device("cpu")):
    self.file_path = file_path
    self.input_length = input_length
    self.output_length = output_length
    self.test_size = test_size
    self.step = step
    
    if scaler is None:
      self.scaler = MinMaxScaler()
    else:
      self.scaler = scaler
      
    self.device = device
      
    self.raw_data = None
    self.len_data = None
    self.train_data = None
    self.test_data = None
    
    self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    self.X_train_scaler, self.X_test_scaler, self.y_train_scaler, self.y_test_scaler = None, None, None, None
    self.len_X_train, self.len_X_test = None, None
    self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor = None, None, None, None
    
    self.dataset = None
    self.model = None

  def load_data(self):
    # загрузка данных с нейтронного монитора в виде последовательности значений
    with open(self.file_path, 'r') as f:
      lines = f.readlines()
    data = []
    for line in lines:
      data.extend([float(x) for x in line.strip().split()])
    self.raw_data = np.array(data)
    self.len_data = self.raw_data.size
    _log.info("Len of load data: %s", self.len_data)
    return self

  def split_and_normalize(self, usage_volume=1, normalize_first=True):
    # разделение на тренировочные и тестовые данные
    split_train_idx = int(self.len_data * (1 - self.test_size) * usage_volume)
    split_test_idx = int(self.len_data * self.test_size * usage_volume)
    self.train_data = self.raw_data[:split_train_idx]
    self.test_data = self.raw_data[split_train_idx : split_train_idx + split_test_idx]
    _log.info("Размер тренировочных и тестовых данных: %s", (self.train_data.size, self.test_data.size))

    if normalize_first:
      # нормализация
      if self.train_data.size > 0:
        # _log.debug("DEBUG 0.1: ", self.train_data)
        self.train_data = self.normalize_train(self.train_data)
        # _log.debug("DEBUG 0.2: ", self.train_data)
        joblib.dump(self.scaler, "scaler.bin", compress=True)
      # _log.debug("DEBUG 0.3: ", self.test_data)
      self.test_data = self.normalize_test(self.test_data)
      # _log.debug("DEBUG 0.4: ", self.test_data)
    return self

  def normalize_train(self, data):
    data = np.log1p(data)
    return self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

  def normalize_test(self, data):
    data = np.log1p(data)
    return self.scaler.transform(data.reshape(-1, 1)).flatten()

  def denormalize_data(self, data):
    # денормализация
    # _log.debug("DEBUG 7: ", data, data.reshape(-1, 1))
    data = self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    return np.expm1(data)

  def normalize_sequence(self, data):
    data = np.log1p(data).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data).flatten(), scaler

  def denormalize_sequence_data(self, data, scaler):
    # денормализация
    # _log.debug("DEBUG 7: ", data, data.reshape(-1, 1))
    data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    return np.expm1(data)

  def _create_sequences(self, data, normalize_sequence=False):
    # создание последовательностей срезов для предсказания и срезов образца предсказания из данных
    X, y = [], []
    X_scaler, y_scaler = [], []
    total_length = self.input_length + self.output_length
    
    for i in range(0, len(data) - total_length + 1, self.step):
      if not normalize_sequence:
        X.append(data[i : i + self.input_length])
        y.append(data[i + self.input_length : i + self.input_length + self.output_length])
      else:
        norm_sequence_data = self.normalize_sequence(data[i : i + self.input_length])
        X.append(norm_sequence_data[0])
        X_scaler.append(norm_sequence_data[1])
        norm_sequence_data = self.normalize_sequence(data[i + self.input_length : i + self.input_length + self.output_length])
        y.append(norm_sequence_data[0])
        y_scaler.append(norm_sequence_data[1])
    return np.array(X), np.array(y), np.array(X_scaler), np.array(y_scaler)

  def prepare_data(self, shuffle=True, normalize_sequence=False):
    # создание последовательностей для данных для обучения и тестовых данных
    self.X_train, self.y_train, self.X_train_scaler, self.y_train_scaler = self._create_sequences(self.train_data, normalize_sequence=normalize_sequence)
    _log.info("Последовательности данных для обучения: %s", (self.X_train.shape, self.y_train.shape))
    self.X_test, self.y_test, self.X_test_scaler, self.y_test_scaler = self._create_sequences(self.test_data, normalize_sequence=normalize_sequence)
    _log.info("Последовательности данных для тестирования: %s", (self.X_test.shape, self.y_test.shape))
    self.len_X_train, self.len_X_test = self.X_train.shape[0], self.X_test.shape[0]

    # преобразование к тензору
    self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
    self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
    self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
    self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)

    # создание датасета
    self.dataset = {}
    if not shuffle:
      self.dataset['train_input'] = self.X_train_tensor
      self.dataset['train_label'] = self.y_train_tensor
    else:
      indices = torch.randperm(self.X_train_tensor.size(0))
      self.dataset['train_input'] = self.X_train_tensor[indices]
      self.dataset['train_label'] = self.y_train_tensor[indices]
    self.dataset['test_input'] = self.X_test_tensor
    self.dataset['test_label'] = self.y_test_tensor

    # self.dataset['train_input'] = self.X_train_tensor
    # self.dataset['train_label'] = torch.from_numpy(self.y_train[:,None]).to(self.device)
    # self.dataset['test_input'] = self.X_test_tensor
    # self.dataset['test_label'] = torch.from_numpy(self.y_test[:,None]).to(self.device)
    return self

  def build_model(self, width=None, grid=3, k=3, seed=0):
    if width == None:
      width=[self.input_length, 4, self.output_length]

    # создание модели
    self.model = KAN(width=width, grid=grid, k=k, seed=seed)
    self.model= self.model.to(self.device)
    _log.info("Модель находится на устройстве: %s", next(self.model.parameters()).device)
    return self

  def load_model(self, model_path):
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Файл модели {model_path} не найден")

    # загрузка данных модели
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()
    _log.info("Модель загружена из %s", model_path)

  def train(self, shuffle_steps=True, opt="LBFGS", steps=1, lamb=0.01, lamb_entropy=2., save_folder_path="./weights/", filename = "model"):
    os.makedirs(save_folder_path, exist_ok=True)
    # обучение
    for i in range(steps):
      results = self.model.fit(self.dataset, opt=opt, steps=1, lamb_entropy=lamb_entropy)
      if not shuffle_steps:
        pass
      else:
        indices = torch.randperm(self.dataset['train_input'].size(0))
        self.dataset['train_input'] = self.dataset['train_input'][indices]
        self.dataset['train_label'] = self.dataset['train_label'][indices]
    # сохранение модели
    torch.save(self.model.state_dict(), save_folder_path + filename)
    _log.info("Модель обучена и сохранена в %s", save_folder_path + filename)
    return results

  def evaluate(self, normalize_sequence=False):
    if self.model is None:
      raise ValueError("Модель не инициализирована или не загружена")
    if self.X_test_tensor is None or self.y_test_tensor is None:
      raise ValueError("Тестовые данные не подготовлены")

    # оценка работы модели по множеству метрик
    test_input = self.X_test_tensor
    test_label = self.y_test_tensor

    predictions = self.model(test_input)

    # Нормализованные метрики
    mse = torch.mean((predictions - test_label) ** 2).item()
    mae = torch.mean(torch.abs(predictions - test_label)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mape = self.calculate_mape(test_label.detach().numpy(), predictions.detach().numpy())
    mase = self.calculate_mase(test_label.detach().numpy(), predictions.detach().numpy())
    deviation_l1 = torch.norm(predictions - test_label, p=1).item()
    deviation_l2 = torch.norm(predictions - test_label, p=2).item()
    
    _log.debug("MSE (Mean Squared Error) (Нормализованные): %.3f", mse)
    _log.debug("MAE (Mean Absolute Error) (Нормализованные): %.3f", mae)
    _log.debug("RMSE (Root Mean Squared Error) (Нормализованные): %.3f", rmse)
    _log.debug("MAPE (Mean Absolute Percentage Error) (Нормализованные): %.2f%%", mape)
    _log.debug("MASE (Mean Absolute Scaled Error) (Нормализованные): %.3f", mase)
    _log.debug("Deviation L1 (Sum of Absolute Errors) (Нормализованные): %.3f", deviation_l1)
    _log.debug("Deviation L2 (Euclidean Norm) (Нормализованные): %.3f", deviation_l2)

    if not normalize_sequence:
      preds_original = self.denormalize_data(predictions.detach().numpy())
      labels_original = self.denormalize_data(test_label.detach().numpy())
      _log.debug("NaNs in preds_original: %s", np.isnan(preds_original).sum())
      _log.debug("Infs in preds_original: %s", np.isinf(preds_original).sum())
      _log.debug("NaNs in labels_original: %s", np.isnan(labels_original).sum())
      _log.debug("Infs in labels_original: %s", np.isinf(labels_original).sum())
      
      # Преобразуем обратно в тензоры
      preds_tensor = torch.tensor(preds_original, dtype=torch.float32)
      labels_tensor = torch.tensor(labels_original, dtype=torch.float32)
      
      # Считаем все метрики на денормализованных данных
      mseo = torch.mean((preds_tensor - labels_tensor) ** 2).item()
      maeo = torch.mean(torch.abs(preds_tensor - labels_tensor)).item()
      rmseo = torch.sqrt(torch.tensor(mseo)).item()
      mapeo = self.calculate_mape(labels_original, preds_original)
      maseo = self.calculate_mase(labels_original, preds_original)
      deviation_l1o = torch.norm(preds_tensor - labels_tensor, p=1).item()
      deviation_l2o = torch.norm(preds_tensor - labels_tensor, p=2).item()
      
      _log.info("MSE (Mean Squared Error) (Денормализованные, не срезовая): %.3f", mseo)
      _log.info("MAE (Mean Absolute Error) (Денормализованные, не срезовая): %.3f", maeo)
      _log.info("RMSE (Root Mean Squared Error) (Денормализованные, не срезовая): %.3f", rmseo)
      _log.info("MAPE (Mean Absolute Percentage Error) (Денормализованные, не срезовая): %.2f%%", mapeo)
      _log.info("MASE (Mean Absolute Scaled Error) (Денормализованные, не срезовая): %.3f", maseo)
      _log.info("Deviation L1 (Sum of Absolute Errors) (Денормализованные, не срезовая): %.3f", deviation_l1o)
      _log.info("Deviation L2 (Euclidean Norm) (Денормализованные, не срезовая): %.3f", deviation_l2o)
      
      return {
        "mse": mseo, "mae": maeo, "rmse": rmseo, "mape": mapeo, "mase": maseo,
        "deviation_l1": deviation_l1o, "deviation_l2": deviation_l2o
      }
    else:
      # Денормализация для normalize_sequence=True - каждая последовательность имеет свой scaler
      preds_np = predictions.detach().numpy()
      labels_np = test_label.detach().numpy()
      preds_original = []
      labels_original = []
      
      for i in range(len(preds_np)):
        pred_denorm = self.denormalize_sequence_data(preds_np[i], self.y_test_scaler[i])
        label_denorm = self.denormalize_sequence_data(labels_np[i], self.y_test_scaler[i])
        preds_original.append(pred_denorm)
        labels_original.append(label_denorm)
      
      preds_original = np.array(preds_original).flatten()
      labels_original = np.array(labels_original).flatten()
      
      # Преобразуем в тензоры и считаем все метрики
      preds_tensor = torch.tensor(preds_original, dtype=torch.float32)
      labels_tensor = torch.tensor(labels_original, dtype=torch.float32)
      
      mseo = torch.mean((preds_tensor - labels_tensor) ** 2).item()
      maeo = torch.mean(torch.abs(preds_tensor - labels_tensor)).item()
      rmseo = torch.sqrt(torch.tensor(mseo)).item()
      mapeo = self.calculate_mape(labels_original, preds_original)
      maseo = self.calculate_mase(labels_original, preds_original)
      deviation_l1o = torch.norm(preds_tensor - labels_tensor, p=1).item()
      deviation_l2o = torch.norm(preds_tensor - labels_tensor, p=2).item()
      
      _log.info("MSE (Mean Squared Error) (Денормализованные, срезовая): %.3f", mseo)
      _log.info("MAE (Mean Absolute Error) (Денормализованные, срезовая): %.3f", maeo)
      _log.info("RMSE (Root Mean Squared Error) (Денормализованные, срезовая): %.3f", rmseo)
      _log.info("MAPE (Mean Absolute Percentage Error) (Денормализованные, срезовая): %.2f%%", mapeo)
      _log.info("MASE (Mean Absolute Scaled Error) (Денормализованные, срезовая): %.3f", maseo)
      _log.info("Deviation L1 (Sum of Absolute Errors) (Денормализованные, срезовая): %.3f", deviation_l1o)
      _log.info("Deviation L2 (Euclidean Norm) (Денормализованные, срезовая): %.3f", deviation_l2o)
      
      return {
        "mse": mseo, "mae": maeo, "rmse": rmseo, "mape": mapeo, "mase": maseo,
        "deviation_l1": deviation_l1o, "deviation_l2": deviation_l2o
      }

  def predict(self, input_data, denormalize=True, normalize_sequence=True, scaler=None):
    if self.model is None:
      raise ValueError("Модель не инициализирована")

    # получение данных предсказания по вводным данным
    input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(1, -1)
    # Обработка batch_size=1 для избежания UserWarning: std(): degrees of freedom is <= 0
    batch_size = max(2, input_tensor.size(0))  # минимум batch_size=2
    if input_tensor.size(0) < batch_size:
      input_tensor = input_tensor.expand(batch_size, -1)
    input_tensor = input_tensor.to(self.device)
    prediction = self.model(input_tensor)
    prediction = prediction[0].detach().cpu().numpy().flatten()

    # денормализация из-за нормализации
    if denormalize:
      if not normalize_sequence:
        # _log.debug("DEBUG 5: ", prediction)
        prediction = self.denormalize_data(prediction)
        # _log.debug("DEBUG 6: ", prediction)
      else:
        prediction = self.denormalize_sequence_data(prediction, scaler)

    return prediction

  def plot_model_view(self, folder="./view", title="plot_kan_model_view.png", train_mode=False):
    if self.model is None:
      raise ValueError("Модель не инициализирована или не загружена")

    # for file in glob.glob(f"{folder}/*.png"):
    #   if file != f".{folder}/{title}.png":
    #     os.remove(file)

    if not train_mode:
      # 2 for fix for UserWarning: std(): degrees of freedom is <= 0 and alpha error.
      _ = self.model(self.X_test_tensor[:2])

    # визуальная структура модели
    self.model.plot(title=title)

    # _log.info(f"Визуальная структура модели сохранена в {folder}/{title}")

  def symbolic_model_view(self, train_mode=False):
    if self.model is None:
      raise ValueError("Модель не инициализирована или не загружена")

    if not train_mode:
      # 2 for fix for UserWarning: std(): degrees of freedom is <= 0 and alpha error.
      _ = self.model(self.X_test_tensor[:2])

    _log.info("Формульная структура модели: создание...")
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    self.model.auto_symbolic(lib=lib)
    formula1 = self.model.symbolic_formula()[0]
    with open('formula1.txt', 'w') as f:
      for value in formula1:
          f.write(f"{value}\n")
    _log.info("Формула модели: %s", formula1)
    # formula1 = sympify(formula1)
    # formula1 = latex(formula1)
    # _log.info(f"LaTeX формула модели: {formula1}")
    # display(Math(formula1))
    return self

  def plot_comparison(self, predicted_data, real_data, len_data=None, save_path="comparison_real_kan_plot.png", save_data=True):
    # сравнение предсказанных данных по вводным данным с реальными данными
    if len_data == None:
      len_data = self.output_length
    plt.figure(figsize=(10, 6))
    plt.plot(range(len_data), real_data, label="Реальные значения", color="blue", marker="o")

    if save_data:
      with open('real_data.txt', 'w') as f:
        for value in real_data:
            f.write(f"{value}\n")
      _log.debug("Реальные данные сохранены в real_data.txt")

    plt.plot(range(len_data), predicted_data, label="Предсказанные значения", color="red", marker="x")

    if save_data:
      with open('predicted_data.txt', 'w') as f:
        for value in predicted_data:
            f.write(f"{value}\n")
      _log.debug("Предсказанные данные сохранены в predicted_data.txt")

    plt.title("Сравнение реальных и предсказанных значений")
    plt.xlabel("Индекс значения")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.savefig(save_path)
    _log.info("График сохранён в %s", save_path)
    plt.close()

  def plot_big_comparison(self, predicted_data, real_data, len_data=None, save_path="comparison_real_kan_plot.png", save_path2="comparison_real_kan_plot.html", save_data=True):
    # сравнение предсказанных данных по вводным данным с реальными данными
    if len_data == None:
      len_data = self.output_length

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(range(len_data)),
        y=real_data,
        marker=dict(color='blue', symbol='circle', size=7),
        mode='markers+lines',
        name='Реальные значения'
    ))

    if save_data:
      with open('real_data.txt', 'w') as f:
        for value in real_data:
            f.write(f"{value}\n")
      _log.debug("Реальные данные сохранены в real_data.txt")

    fig.add_trace(go.Scatter(
        x=list(range(len_data)),
        y=predicted_data,
        marker=dict(color='red', symbol='x', size=7),
        mode='markers+lines',
        name='Предсказанные значения'
    ))

    if save_data:
      with open('predicted_data.txt', 'w') as f:
        for value in predicted_data:
            f.write(f"{value}\n")
      _log.debug("Предсказанные данные сохранены в predicted_data.txt")

    fig.update_layout(
        title="Сравнение реальных и предсказанных значений",
        xaxis_title="Индекс значения",
        yaxis_title="Значение",
        legend=dict(x=1, y=1),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        template="plotly_white"
    )

    try:
      fig.show()
    except Exception as e:
      _log.warning("Не удалось отобразить graph интерактивно: %s", e)

    try:
      fig.write_image(save_path)
      _log.info("График сохранён в %s", save_path)
    except Exception as e:
      _log.warning("Не удалось сохранить PNG: %s", e)
      
    try:
      fig.write_html(save_path2)
      _log.info("График сохранён в %s", save_path2)
    except Exception as e:
      _log.warning("Не удалось сохранить HTML: %s", e)

  def filter_train_data(self):
    """Применить фильтрацию к тренировочным данным"""
    if self.train_data is not None and len(self.train_data) > 0:
      _log.info(f"Фильтрация тренировочных данных ({len(self.train_data)} точек)...")
      self.train_data = np.array(filter_data(self.train_data.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard"))
      _log.debug(f"Фильтрация применена к тренировочным данным")

  def filter_test_data(self):
    """Применить фильтрацию к тестовым данным"""
    if self.test_data is not None and len(self.test_data) > 0:
      _log.info(f"Фильтрация тестовых данных ({len(self.test_data)} точек)...")
      self.test_data = np.array(filter_data(self.test_data.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard"))
      _log.debug(f"Фильтрация применена к тестовым данным")

  def filter_raw_data(self):
    """Применить фильтрацию к изначально загруженным данным (после load_data)"""
    if self.raw_data is not None and len(self.raw_data) > 0:
      _log.info(f"Фильтрация исходных данных ({len(self.raw_data)} точек)...")
      self.raw_data = np.array(filter_data(self.raw_data.tolist(), wavelet="db3", alpha=0.05, J=5, threshold_type="hard"))
      _log.debug(f"Фильтрация применена к исходным данным")

  # Метрики регрессии для оценки качества предсказания модели
  def calculate_mse(self, real, predicted):
    """Среднеквадратичная ошибка (MSE, Mean Squared Error)"""
    return np.mean((predicted - real) ** 2)

  def calculate_deviation_l1(self, real, predicted):
    """Отклонение по L1-норме (сумма абсолютных ошибок)"""
    return np.sum(np.abs(predicted - real))

  def calculate_deviation_l2(self, real, predicted):
    """Отклонение по L2-норме (евклидова норма)"""
    return np.sqrt(np.sum((predicted - real) ** 2))

  def calculate_mae(self, real, predicted):
    """Средняя абсолютная ошибка (MAE, Mean Absolute Error)"""
    return np.mean(np.abs(predicted - real))

  def calculate_rmse(self, real, predicted):
    """Корневая среднеквадратичная ошибка (RMSE, Root Mean Squared Error)"""
    return np.sqrt(self.calculate_mse(real, predicted))

  def calculate_mape(self, real, predicted):
    """Средняя абсолютная процентная ошибка (MAPE, Mean Absolute Percentage Error)"""
    mask = real != 0
    if np.sum(mask) == 0:
      return 0.0 if np.allclose(predicted, 0) else float("inf")
    
    ape = np.abs((real[mask] - predicted[mask]) / real[mask])
    return np.mean(ape) * 100

  def calculate_mase(self, real, predicted):
    """
    Средняя абсолютная масштабированная ошибка (MASE, Mean Absolute Scaled Error)
    Масштабируется по сравнению с наивным прогнозом (предыдущее значение)
    """
    n = len(real)
    if n < 2:
      return 0.0
    
    # Используем предыдущее значение
    naive_forecast = np.concatenate(([real[0]], real[:-1]))
    denominator = np.mean(np.abs(real - naive_forecast))
    
    if denominator == 0:
      return 0.0 if np.allclose(real, predicted) else float("inf")
    
    return np.mean(np.abs(real - predicted)) / denominator
