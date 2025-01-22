"""
Программа: Тренировка данных
"""

from catboost import CatBoostClassifier

import pandas as pd
import numpy as np

from ..train.metrics import save_metrics


def find_optimal_params(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
) -> np.array:
    """
    Подбор гиперпараметров CatBoostClassifier
    :param x_train: объект-признаки
    :param y_train: целевая переменная
    :return: словарь с лучшими параметрами
    """
    # Формируем список из колонок с категориальными данными для подачи в Catboost
    cat_features = x_train.select_dtypes('category').columns.tolist()

    grid = {
        "n_estimators": [300],
        "learning_rate": np.logspace(-3, -1, 3),
        "max_depth": list(range(4, 15)),
        "l2_leaf_reg": np.logspace(-5, 2, 5),
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS", "No"],
        "border_count": [128, 254],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "random_state": [kwargs['random_state']]}

    model = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               cat_features=cat_features,
                               silent=True)

    grid_search_result = model.randomized_search(grid,
                                                 X=x_train,
                                                 y=y_train)

    best_params_cat = grid_search_result['params']
    return best_params_cat


def train_model(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_test: pd.DataFrame,
        y_test: pd.Series,
        best_params: dict,
        metric_path: str,
) -> CatBoostClassifier:
    """
    Обучение модели на лучших параметрах
    :param x_train: тренировочный набор объект-признаки
    :param y_train: тренировочный набор таргет
    :param x_test: тестовый набор объект-признаки
    :param y_test: тестовый набор таргет
    :param best_params: словарь с лучшими метриками
    :param metric_path: путь до папки с метриками
    :return: CatBoostClassifier
    """
    # Формируем список из колонок с категориальными данными для подачи в Catboost
    cat_features = x_train.select_dtypes('category').columns.tolist()

    # обучение на лучших параметрах
    clf = CatBoostClassifier(**best_params,
                              cat_features=cat_features,
                              loss_function='Logloss',
                              eval_metric='AUC')
    clf.fit(x_train, y_train, verbose=False)

    # сохранение метрик
    save_metrics(x_test=x_test, y_test=y_test, model=clf, metric_path=metric_path)
    return clf
