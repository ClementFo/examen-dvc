import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import joblib
import os


def main():
    X_train_scaled = pd.read_csv("data/processed_data/normalized/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/split_data/y_train.csv")
    model = ElasticNet()

    parametre = {
        'alpha': [0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'l1_ratio': np.arange(0.0, 1.01, 0.05)
    }

    grid_search = GridSearchCV(estimator=model, param_grid=parametre, cv=3)

    grid_search.fit(X_train_scaled, y_train)

    os.makedirs("models/best_param", exist_ok=True)
    joblib.dump(grid_search.best_params_, "models/best_param/best_params.pkl")


if __name__ == "__main__":
    main()
