"""Загрузка готовой модели без повторного обучения."""
from pathlib import Path
import pickle
import math
import pandas as pd

BASE = Path(__file__).resolve().parent


def load_model():
    path = BASE / 'models' / 'iris_model.pkl'
    if not path.exists():
        raise FileNotFoundError('Сначала запустите python train.py в папке ИКМ.')
    # Загружаю только локальный файл своей модели.
    with path.open('rb') as stream:
        return pickle.load(stream)


def prepare_input(values, features):
    if len(values) != 4:
        raise ValueError('Нужны четыре измерения.')
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError('Каждое поле должно содержать число.') from None
    if any(not math.isfinite(value) or value <= 0 for value in numbers):
        raise ValueError('Размеры должны быть положительными конечными числами в сантиметрах.')
    return pd.DataFrame([numbers], columns=features)


def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    payload = load_model()
    X = prepare_input([sepal_length, sepal_width, petal_length, petal_width], payload['features'])
    code = int(payload['model'].predict(X)[0])
    return payload['species'][code]


def prediction_text(sepal_length, sepal_width, petal_length, petal_width):
    try:
        payload = load_model()
        X = prepare_input([sepal_length, sepal_width, petal_length, petal_width], payload['features'])
        code = int(payload['model'].predict(X)[0])
        outside = [name for name in payload['features']
                   if not payload['train_min'][name] <= X.iloc[0][name] <= payload['train_max'][name]]
        text = f'Предполагаемый вид: {payload["species"][code]}.\nМодель: {payload["model_name"]}.'
        if outside:
            text += '\nЧасть размеров выходит за диапазон обучающих примеров. Такой результат нужно проверять отдельно.'
        return text
    except (ValueError, FileNotFoundError) as error:
        return str(error)
