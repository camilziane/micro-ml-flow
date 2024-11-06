import ml
import os


from mlflow.models import infer_signature

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

host = os.getenv("MLFLOW_HOST", "localhost")
port = os.getenv("MLFLOW_PORT", "8080")


def iris_model(X_train, X_test, y_train, y_test):
    params = {
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 8888,
    }
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    return params, lr


if __name__ == "__main__":
    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    iris_params, iris_lr = iris_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )

    uri = f"http://{host}:{port}"
    ml.set_tracking_uri(uri=uri)
    ml.set_experiment("MLflow Iris")
    with ml.start_run():
        # Log the hyperparameters
        ml.log_params(iris_params)

        # Log the loss metric
        ml.log_metric(
            "accuracy", accuracy_score(y_true=y_test, y_pred=iris_lr.predict(X_test))
        )

        ml.set_tag("Training Info", "Basic LR model for iris data")

        signature = infer_signature(X_train, iris_lr.predict(X_train))
        model_info = ml.sklearn.log_model(
            sk_model=iris_lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-iris",
        )
