"""
Программа: Отрисовка графиков
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def barplot_group(
        data: pd.DataFrame,
        col_main: str,
        col_group: str,
        title: str
) -> matplotlib.figure.Figure:
    """
     Функция для отрисовки сгруппированных барплотов
     :param data: датасет
     :param col_main: колонка для анализа
     :param col_group: колонка для группировки
     :param title: название графика
     :return: график
    """
    data_group = (
        data.groupby([col_group])[col_main]
        .value_counts(normalize=True)
        .rename('percentage')
        .mul(100)
        .reset_index()
        .sort_values(col_group)
    )

    fig = plt.figure(figsize=(15, 7))

    ax = sns.barplot(x=col_main,
                     y="percentage",
                     hue=col_group,
                     data=data_group,
                     palette='Paired',
                     saturation=.5
                    )

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(
            percentage,  # текст
            (p.get_x() + p.get_width() / 2., p.get_height()),  # координата xy
            ha='center',  # центрирование
            va='center',
            xytext=(0, 10),
            textcoords='offset points',  # точка смещения относительно координаты
            fontsize=14)

    plt.title(title, fontsize=20)
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

def boxplot_group(
        data: pd.DataFrame,
        col_x: str,
        col_y: str,
        hue: str,
        title: str
) -> matplotlib.figure.Figure:
    """
    Функция для отрисовки сгруппированных боксплотов
    :param data: датасет
    :param col_x: колонка для анализа (категориальные данные)
    :param col_y: колонка для анализа (числовые данные)
    :param hue: колонка для группировки
    :param title: название графика
    :return: график
    """
    fig = plt.figure(figsize=(20, 10))

    sns.boxplot(data=data, x=col_x, y=col_y, hue=hue, palette='Paired', saturation=.5)

    plt.title(title, fontsize=30)
    plt.ylabel(col_y, fontsize=20)
    plt.xlabel(col_x, fontsize=20)
    plt.legend(loc='upper right', fontsize=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig

def boxplot(
        data: pd.DataFrame,
        col_main: str,
        col_group: str,
        title: str
) -> matplotlib.figure.Figure:
    """
    Функция для отрисовки одиночных боксплотов
    :param data: датасет
    :param col_main: колонка для анализа
    :param col_group: колонка для группировки
    :param title: название графика
    :return: график
    """
    fig = plt.figure(figsize=(20, 10))

    sns.boxplot(data=data, y=col_main, hue=col_group, palette='Paired', saturation=.5)

    plt.title(title, fontsize=30)
    plt.ylabel(col_main, fontsize=20)
    plt.legend(loc='upper right', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    return fig