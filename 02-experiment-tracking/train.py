import os
import pickle
import click
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
def run_train(data_path: str):
    # Configure tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("nyc-green-taxi-experiment")

    # Dynamically generate run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rf_{timestamp}"
    with mlflow.start_run(run_name=run_name):
        print("Starting run:", run_name)
        mlflow.autolog()

        # Load datasets
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Start training process
        start = time.time()

        print("Training Random Forest Regressor...")
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        end = time.time()
        print(f"Training completed in {end - start:.2f} seconds")

        # Make predictions
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"RMSE: {rmse:.2f}")

        mlflow.log_metric("validation_rmse", rmse)


if __name__ == "__main__":
    run_train()
