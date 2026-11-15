import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.preprocessing import get_preprocessor, get_pipeline


def train_model(X_train, y_train):

    preprocessor = get_preprocessor()

    logistic_model = LogisticRegression(max_iter=1000)

    pipeline = get_pipeline(logistic_model, preprocessor)

    param = {
        "classifier__C": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)


    Path("/opt/airflow/models").mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        grid_search.best_estimator_,
        "/opt/airflow/models/logistic.joblib"
    )

    print("Best parameters:", grid_search.best_params_)
    print("Best F1 macro:", grid_search.best_score_)

    return grid_search