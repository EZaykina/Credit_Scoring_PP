"""
Программа: Пайплайн для тренировки модели
"""
import json

import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
import pandas as pd


def create_dict_metrics(y_test: pd.Series,
                        y_predict: pd.Series,
                        y_proba: pd.Series
) -> dict:
    """
    Получение словаря с метриками для задачи классификации и запись в словарь
    :param y_test: реальные данные
    :param y_predict: предсказанные значения
    :param y_proba: предсказанные вероятности
    :return: словарь с метриками
    """
    dict_metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba[:, 1]), 3),
        "precision": round(precision_score(y_test, y_predict), 3),
        "recall": round(recall_score(y_test, y_predict), 3),
        "f1": round(f1_score(y_test, y_predict), 3),
        "logloss": round(log_loss(y_test, y_proba), 3),
    }
    return dict_metrics


def save_metrics(x_test: pd.DataFrame,
                 y_test: pd.Series,
                 model: object,
                 metric_path: str
) -> None:
    """
    Получение и сохранение метрик
    :param x_test: объект-признаки
    :param y_test: целевая переменная
    :param model: модель
    :param metric_path: путь для сохранения метрик
    """
    metrics = create_dict_metrics(y_test=y_test,
                                  y_predict=model.predict(x_test),
                                  y_proba=model.predict_proba(x_test),
    )
    with open(metric_path, "w") as file:
        json.dump(metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получение метрик из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """

    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics

