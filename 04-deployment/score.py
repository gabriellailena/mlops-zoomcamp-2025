#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd


def load_model(filepath):
    with open(filepath, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(model_file: str, dicts: list):
    # Load the model and vectorizer
    dv, model = load_model(model_file)

    # Transform the input data
    X_val = dv.transform(dicts)

    # Make predictions
    y_pred = model.predict(X_val)
    
    return y_pred


def save_output(df, y_pred, output_file):
    df_result = pd.DataFrame(
        {'ride_id': df['ride_id'], 'prediction': y_pred}
    )
    
    # Save the results to a parquet file
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(f'Results saved to {output_file}')
    return

def run():
    # Read data
    year = 2023
    month = 3

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')

    # Make predictions
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')

    y_pred = predict('model.bin', dicts)

    # Checking the standard deviation of predictions
    print(f'Standard deviation of predictions: {y_pred.std():.2f}')

    # Save the ride_id and predictions to a parquet file
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    output_file = f'yellow_tripdata_{year:04d}-{month:02d}_predictions.parquet'
    save_output(df, y_pred, output_file)






