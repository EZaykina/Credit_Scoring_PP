"""
Программа: Пайплайн для тренировки модели
"""

import os
import joblib
import yaml

from ..train.train import find_optimal_params, train_model
from ..data.get_data import get_dataset
from ..data.split_dataset import split_train_test
from ..transform.transform import pipeline_preprocess


def pipeline_training(config_path: str) -> None:
    """
    Полный цикл получения данных, их предобработка и тренировка модели
    :param config_path: путь до конфигурационного файла
    :return: None
    """
    # извлекаем параметры из конфигурационного файла
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # загружаем датасет
    train_data = get_dataset(dataset_path=preprocessing_config["train_path"])

    # preprocessing тренировочных данных
    train_data = pipeline_preprocess(data=train_data, flg_evaluate=False, **preprocessing_config)

    # разбиение данных на train/test
    x_train, x_test, y_train, y_test = split_train_test(dataset=train_data, **preprocessing_config)

    # поиск оптимальных гиперпараметров модели
    params = find_optimal_params(x_train=x_train, y_train=y_train, **train_config)

    # тренировка модели на лучших параметрах
    clf = train_model(x_train=x_train,
                      x_test=x_test,
                      y_train=y_train,
                      y_test=y_test,
                      best_params=params,
                      metric_path= train_config["metrics_path"])

    # сохраняем модель
    joblib.dump(clf, os.path.join(train_config["model_path"]))
