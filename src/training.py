import pandas as pd

from sklearn.model_selection import train_test_split

from src.data import clean_data
from src.model import train_model


def run_training():

    df = clean_data()

    X = df.drop(columns=["Mental_Health_Risk"])
    y = df["Mental_Health_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_test.to_csv("/opt/airflow/data/X_test.csv", index=False)
    y_test.to_csv("/opt/airflow/data/y_test.csv", index=False)

    model = train_model(X_train, y_train)

    print("Training completed!")