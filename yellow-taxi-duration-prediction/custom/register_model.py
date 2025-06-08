import mlflow
import pickle
from pathlib import Path

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@custom
def register_sklearn_model(*args, **kwargs):
    """
    Registers a Scikit-Learn model to MLFLow Model Registry.
    """
    dv = args[0][0]
    model = args[0][1]

    # Log the DictVectorizer object
    with open("models/vectorizer.pkl", "wb") as f_out:
        pickle.dump(dv, f_out)

    mlflow.log_artifact(
        "models/vectorizer.pkl", 
        artifact_path="vectorizer"
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="linear-regression-model",
        registered_model_name="linear-reg-model",
    )

    return None


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is None, 'Unexpected output'
