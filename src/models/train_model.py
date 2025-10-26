import pandas as pd
from sklearn.linear_model import ElasticNetCV
import joblib
import os


def main():
    X_train_scaled = pd.read_csv("data/processed_data/normalized/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/split_data/y_train.csv")

    best_params = joblib.load("models/best_param/best_params.pkl")

    best_alpha = [best_params['alpha']]
    best_l1_ratio = best_params['l1_ratio']

    best_model = ElasticNetCV(cv=3, l1_ratio=best_l1_ratio, alphas=best_alpha)

    best_model.fit(X_train_scaled, y_train)
    os.makedirs("models/models_trained", exist_ok=True)
    joblib.dump(best_model, "models/models_trained/model_trained.pkl")


if __name__ == "__main__":
    main()
