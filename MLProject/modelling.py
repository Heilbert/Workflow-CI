import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "diabetes_preprocessing")

mlflow.set_experiment("Diabetes Classification - Autolog")


def load_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Outcome", axis=1)
    y_train = train_df["Outcome"]

    X_test = test_df.drop("Outcome", axis=1)
    y_test = test_df["Outcome"]

    return X_train, X_test, y_train, y_test


def main():

    # print("MLflow version:")
    # print(mlflow.__version__)

    X_train, X_test, y_train, y_test = load_data()

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    main()