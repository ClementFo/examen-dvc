import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import os


def main():
    X_test_scaled = pd.read_csv("data/processed_data/normalized/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/split_data/y_test.csv")

    model = joblib.load("models/models_trained/model_trained.pkl")

    pred_test = model.predict(X_test_scaled)

    preds = y_test.copy()
    preds["prediction"] = pred_test
    os.makedirs("data/processed_data/prediction", exist_ok=True)
    preds.to_csv("data/processed_data/prediction/predictions.csv", index=False)

    metrics = {
        "score test": model.score(X_test_scaled, y_test),
        "mse": mean_squared_error(y_test, pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, pred_test)),
        "mae": mean_absolute_error(y_test, pred_test)
    }

    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
