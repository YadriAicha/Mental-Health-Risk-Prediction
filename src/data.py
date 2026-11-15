import pandas as pd
from pathlib import Path

def load_data():
    df = pd.read_csv("/opt/airflow/dataset/raw/student_mental_health.csv")
    return df


def clean_data():

    df = load_data()

    df = df.drop(columns=["Student_ID"])
    df = df.drop_duplicates()

    output_path = Path("/opt/airflow/data/cleaned_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return df

