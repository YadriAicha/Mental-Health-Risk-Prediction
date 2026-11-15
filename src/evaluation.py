import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model():

    model = joblib.load("/opt/airflow/models/logistic.joblib")

    X_test = pd.read_csv("/opt/airflow/data/X_test.csv")

    y_true = pd.read_csv(
        "/opt/airflow/data/y_test.csv"
    ).squeeze("columns")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="macro"
    )

    recall = recall_score(
        y_true, y_pred, average="macro"
    )

    f1 = f1_score(
        y_true, y_pred, average="macro"
    )

    auc_score = roc_auc_score(
        y_true,
        y_proba,
        average="macro",
        multi_class="ovr"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=model.classes_
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc_score)
    print("Confusion Matrix:\n", cm)