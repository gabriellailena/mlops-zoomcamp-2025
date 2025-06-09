import pandas as pd
import numpy as np
import mlflow
import pickle
from sklearn.linear_model import LinearRegression 
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from typing import List

mlflow.set_tracking_uri("http://mlflow-local:5000")
mlflow.set_experiment("nyc-yellow-taxi-experiment")


if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def train_linear_regression_model(*args, **kwargs):
    def transform_dataframe_to_dict(df: pd.DataFrame, columns: List[str], cast: str = str) -> List[dict]:
        """
        Transform the DataFrame into a list of dictionaries for the specified columns. If cast is str, cast the columns to string.
        :param df: DataFrame to transform
        :param columns: List of columns to include in the transformation
        :param cast: Type to cast the columns to (default is str)
        :return: List of dictionaries representing the DataFrame
        """
        # Keep only the relevant columns and cast to desired type
        df_features = df[columns].astype(cast)

        # Convert the DataFrame to a list of dictionaries
        df_dict = df_features.to_dict(orient='records')
        print("Transformed data (first 3 rows):", df_dict[:3])
        return df_dict

    def train_categorical_one_hot_encode(list_dict: List[dict]) -> np.ndarray:
        """
        Train a one-hot encoder for the specified categorical columns in the DataFrame, and return the feature matrix.

        :param df: DataFrame to encode
        :param column: Column to one-hot encode
        :return: DataFrame with one-hot encoded column
        """
        # Vectorize the data
        dv = DictVectorizer(sparse=True)  # Using sparse=True for memory efficiency
        dv = dv.fit(list_dict)

        return dv

    features = ['PULocationID', 'DOLocationID']
    target = 'duration'

    with mlflow.start_run():
        mlflow.log_param("features", features)
        mlflow.log_param("target", target)

        # Prepare data
        print("Preparing data...")
        df_preprocessed = args[0]
        df_dict = transform_dataframe_to_dict(df_preprocessed, features)

        # One-hot encode the categorical features
        dv = train_categorical_one_hot_encode(df_dict)
        X_train = dv.transform(df_dict)

        # Define the target variable
        y_train = df_preprocessed[target].to_numpy()

        # Train the model
        print("Training LinearRegressionModel...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        print(f"Successfully trained LinearRegression model with intercept = {model.intercept_}")
        
        # Calculate RMSE on the training data
        y_train_preds = model.predict(X_train)
        rmse_train = root_mean_squared_error(y_train, y_train_preds)
        mlflow.log_metric("rmse_train", rmse_train)

        print("Registering model...")
        with open("dict_vectorizer.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(
            "dict_vectorizer.pkl", 
            artifact_path="preprocessor"
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )
        print("Model registered to MLFlow model registry.")

    return dv, model


if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
    
@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert len(output) == 2, 'Output tuple should contain two elements'
    assert isinstance(output[0], DictVectorizer), 'First element of output tuple should be DictVectorizer'
    assert isinstance(output[1], LinearRegression), 'Second element of output tuple should be LinearRegression'