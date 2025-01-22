"""
Программа: Модель для прогнозирования того, стоит ли выдавать заемщику кредит
"""

import warnings

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

from src.pipelines.pipeline import pipeline_training
from src.evaluate.evaluate import pipeline_evaluate
from src.train.metrics import load_metrics

warnings.filterwarnings("ignore")

# Создаем экземпляр класса FastAPI
app = FastAPI()
CONFIG_PATH = "../config/params.yml"


class LoanCustomer(BaseModel):
    """
    Признаки для получения результатов модели
    """
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

@app.get("/hello")
def welcome():
    """
    Hello
    :return: None
    """
    return {"message": "Hello"}


@app.post("/train")
def training():
    """
    Обучение модели, логирование метрик
    """
    pipeline_training(config_path=CONFIG_PATH)
    metrics = load_metrics(config_path=CONFIG_PATH)

    return {"metrics": metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    result = pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"

    return {"prediction": result[:5]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
