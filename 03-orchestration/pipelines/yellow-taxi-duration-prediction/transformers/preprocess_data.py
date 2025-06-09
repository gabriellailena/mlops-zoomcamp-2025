if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def preprocess_nyc_yellow_taxi_data(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Dataframe
    """
    # Get output of previous block
    df = args[0]

    # Get trip duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Remove outliers
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    print(f"Retrieved {len(df)} rows after preprocessing.")

    return df



@test
def test_output(output, *args) -> None:
    """
    Test block output.
    """
    assert output is not None, 'The output is undefined'