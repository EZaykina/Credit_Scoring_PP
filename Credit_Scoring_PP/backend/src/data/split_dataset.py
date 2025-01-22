"""
Программа: разделение данных на train и test
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_test(dataset: pd.DataFrame, **kwargs):
    """
    Разбиение датасета на тренировочные и тестовые данные (объект-признаки и таргет)
    :param dataset: датасет
    :return: набор тренировочных и тестовых данных
    """
    # Разбиение на объект-признаки и таргет
    x = dataset.drop(columns= kwargs['target_column'],axis=1)
    y = dataset[kwargs['target_column']]

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        stratify=y,
                                                        test_size=kwargs['test_size'],
                                                        random_state=kwargs['random_state'])

    return x_train, x_test, y_train, y_test