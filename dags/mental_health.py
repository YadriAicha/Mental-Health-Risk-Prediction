from datetime import datetime
from airflow.decorators import dag, task

from src.training import run_training
from src.evaluation import evaluate_model


@dag(
    dag_id="health_dag",
    schedule=None,
    start_date=datetime(2026, 9, 15),
    catchup=False
)
def handle_dataset():

    @task
    def training():
        run_training()

    @task
    def evaluation():
        evaluate_model()

    train_task = training()
    evaluate_task = evaluation()

    train_task >> evaluate_task


handle_dataset()