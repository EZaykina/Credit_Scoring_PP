"""
Программа: Тренировка модели на backend, отображение метрик и
графиков обучения на экране
"""

import os
import json
import joblib
import requests
import streamlit as st


def start_training(config: dict, endpoint: object) -> None:
    """
    Тренировка модели и вывод результатов
    :param config: конфигурационный файл
    :param endpoint: endpoint
    """
    # загружаем метрики
    if os.path.exists(config["train"]["metrics_path"]):
        with open(config["train"]["metrics_path"]) as json_file:
            old_metrics = json.load(json_file)
    else:
        # если до этого модель не обучалась и значений метрик нет
        old_metrics = {"roc_auc": 0, "precision": 0, "recall": 0, "f1": 0, "logloss": 0}

    # Тренируем модель
    with st.spinner("Модель подбирает оптимальные параметры..."):
        output = requests.post(endpoint, timeout=8000)
    st.success("Success!")

    new_metrics = output.json()["metrics"]

    # diff metrics
    roc_auc, precision, recall, f1_metric, logloss = st.columns(5)
    roc_auc.metric(
        "ROC-AUC",
        new_metrics["roc_auc"],
        f"{new_metrics['roc_auc']-old_metrics['roc_auc']:.3f}",
    )
    precision.metric(
        "Precision",
        new_metrics["precision"],
        f"{new_metrics['precision']-old_metrics['precision']:.3f}",
    )
    recall.metric(
        "Recall",
        new_metrics["recall"],
        f"{new_metrics['recall']-old_metrics['recall']:.3f}",
    )
    f1_metric.metric(
        "F1 score", new_metrics["f1"], f"{new_metrics['f1']-old_metrics['f1']:.3f}"
    )
    logloss.metric(
        "Logloss",
        new_metrics["logloss"],
        f"{new_metrics['logloss']-old_metrics['logloss']:.3f}",
    )


