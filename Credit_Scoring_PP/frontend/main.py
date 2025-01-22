"""
Программа: Frontend часть
"""

import os

import streamlit as st
import yaml

import os

from src.data.get_dataset import load_data, get_dataset
from src.plotting.charts import barplot_group, boxplot_group, boxplot
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_from_file

CONFIG_PATH = "../config/params.yml"



def main_page():
    """
    Страница с описанием проекта
    """
    st.image(
        "https://avatars.dzeninfra.ru/get-zen_doc/3822405/pub_6194456a04dce76333561873_619446f221deaa297707443b/scale_1200",
        width=600,
    )

    st.markdown("# Описание проекта")
    st.title("Модель кредитного скоринга")
    st.write(
        """
        Банк предоставил информацию для прогнозирования результатов одобрения кредита на основе данных 
        о каждом заявителе, финансовых показателей и факторов, связанных с конкретным кредитом. 
        Банку требуется модель, которая на основании предоставленных данных сможет спрогнозировать 
        **стоит ли выдавать конкретному клиенту запрашиваемый кредит** или нет.
        Чтобы предсказать, будет ли одобрен клиенту кредит у нас есть информация о различных характеристиках,
        влияющих на решение об одобрении кредита: личные данные клиентов (возраст, доход, трудовой стаж, история неисполнения кредитных обязательств и продолжительность кредитной истории), данные о запрашиваемом кредите (сумма и цель кредита, процентная ставка и степень риска).
        
        Target - loan_status
        """
    )

    # наименование колонок
    st.markdown(
        """
        ### Описание полей 
            - person_age - возраст претендента в годах.
            - person_income - Годовой доход претендента в долларах США.
            - person_home_ownership: Статус владельца жилья (например, арендатор, собственник, ипотечник).
            - person_emp_length - трудовой стаж в годах.
            - loan_intent - Цель кредита (например, образование, лечение, личные нужды).
            - loan_grade - степень риска, присвоенная кредиту, для оценки кредитоспособности заявителя.
            - loan_amnt - Общая сумма кредита, запрашиваемая заявителем.
            - loan_int_rate - Процентная ставка, связанная с кредитом.
            - loan_percent_income - процент дохода заявителя, выделяемый на погашение кредита.
            - cb_person_default_on_file - указывает, есть ли у заявителя история неисполнения обязательств («Y» — да, «N» — нет).
            - cb_person_cred_hist_length - продолжительность кредитной истории заявителя в годах.
            - loan_status - статус одобрения кредита (1 - одобрен или 0 - не одобрен).
    """
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory Data Analysis️")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # выгружаем датасет
    data = get_dataset(dataset_path=config["preprocessing"]["train_path"])
    st.write(data.head(10))

    # отрисовка графиков
    ## описываем чек-боксы
    loan_status_home_ownership = st.sidebar.checkbox("Статус домовладения - Статус выдачи кредита")
    loan_status_loan_intent = st.sidebar.checkbox("Цель кредита - Статус выдачи кредита")
    loan_status_loan_grade = st.sidebar.checkbox("Степень риска кредита - Статус выдачи кредита")
    loan_status_pers_default = st.sidebar.checkbox("История неисполнения кредитных обязательств - Статус выдачи кредита")
    loan_status_perc_income = st.sidebar.checkbox("Процент дохода, выделяемого на оплату кредита - Статус выдачи кредита")
    loan_status_loan_amount = st.sidebar.checkbox("Сумма кредита - Статус выдачи кредита")
    loan_status_home_ownership_age = st.sidebar.checkbox("Возраст - Статус домовладения - Статус выдачи кредита")
    loan_status_risk_rate = st.sidebar.checkbox("Процентная ставка - Степень риска кредита - Статус выдачи кредита")

    if loan_status_home_ownership:
        st.pyplot(
            barplot_group(
                data=data,
                col_main='loan_status',
                col_group='person_home_ownership',
                title='Статус домовладения - Статус выдачи кредита'
            )
        )
        st.write('Людям, арендующим жилье, одобряют кредиты чаще других')

    if loan_status_loan_intent:
        st.pyplot(
            barplot_group(
                data=data,
                col_main='loan_status',
                col_group='loan_intent',
                title='Цель кредита - Статус выдачи кредита'
            )
        )
        st.write('Чаще всего отказывают в кредитах на венчурные услуги и образование, чаще всего одобряют кредиты на рефинансирование')

    if loan_status_loan_grade:
        st.pyplot(
            barplot_group(
                data=data,
                col_main='loan_status',
                col_group='loan_grade',
                title='Степень риска кредита - Статус выдачи кредита'
            )
        )
        st.write('Чем выше степень риска кредита, тем чаще отказывают в его выдаче')

    if loan_status_pers_default:
        st.pyplot(
            barplot_group(
                data=data,
                col_main='loan_status',
                col_group='cb_person_default_on_file',
                title='История неисполнения кредитных обязательств - Статус выдачи кредита'
            )
        )
        st.write('Заемщикам, имеющим историю неисполнения обязательств, кредиты одобряют чаще')

    if loan_status_perc_income:
        st.pyplot(
            boxplot(
                data=data,
                col_main='loan_percent_income',
                col_group='loan_status',
                title='Процент дохода, выделяемого на оплату кредита - Статус выдачи кредита'
            )
        )
        st.write('Прямая зависимость между одобрением кредита и процентом дохода, выделяемого на его погашение, отсутствует. Кредиты часто выдает и при высоком проценте.')

    if loan_status_loan_amount:
        st.pyplot(
            boxplot(
                data=data,
                col_main='loan_amnt',
                col_group='loan_status',
                title='Сумма кредита - Статус выдачи кредита'
            )
        )
        st.write('Прямой зависимости нет.')

    if loan_status_home_ownership_age:
        st.pyplot(
            boxplot_group(
                data=data,
                col_x='person_home_ownership',
                col_y='person_age',
                hue='loan_status',
                title='Возраст - Статус домовладения - Статус выдачи кредита'
            )
        )
        st.write('Связь возраст-статус домовладельца-статус выдачи кредита отсутствует, распределение по возрасту во всех группах домовладения примерно одинаковое')

    if loan_status_risk_rate:
        st.pyplot(
            boxplot_group(
                data=data,
                col_x='loan_grade',
                col_y='loan_int_rate',
                hue='loan_status',
                title='Процентная ставка - Степень риска кредита - Статус выдачи кредита'
            )
        )
        st.write('Чем выше степень риска кредита, тем выше процент под который выдают кредит')


def training():
    """
    Тренировка модели
    """
    st.markdown("# Тренировка модели СatBoost")
    # get params
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    # endpoint
    endpoint = config["endpoints"]["train"]

    if st.button("Начать тренировку"):
        start_training(config=config, endpoint=endpoint)


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Предсказание модели")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]

    upload_file = st.file_uploader(
        "", type=["csv", "xlsx"], accept_multiple_files=False
    )
    # проверка загружен ли файл
    if upload_file:
        dataset_csv_df, files = load_data(data=upload_file, type_data="Test")
        # проверка на наличие сохраненной модели
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_csv_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Exploratory Data Analysis": exploratory,
        "Тренировка модели": training,
        "Предсказание из файла": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
