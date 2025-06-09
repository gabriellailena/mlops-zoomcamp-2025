import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_yellow_taxi_data(*args, **kwargs):
    """
    Loads the Yellow Taxi trip data for March 2023 from a public S3 bucket.
    """
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(url)

    print(f"Read {len(df)} rows from {url}")

    return df


@test
def test_output(output, *args) -> None:
    """
    Test block output.
    """
    assert output is not None, 'The output is undefined'