import itertools
from typing import Iterable, List, Union, Optional
import numpy as np
import pandas as pd

from autora.utils.deprecation import deprecated_alias
from sklearn.preprocessing import StandardScaler


def score_sample(conditions: Union[pd.DataFrame, np.ndarray],
                 models: List,
                 num_samples: Optional[int] = None):
    """
    A experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        X: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool
    """

    if isinstance(conditions, Iterable) and not isinstance(conditions, pd.DataFrame):
        conditions = np.array(list(conditions))

    condition_pool_copy = conditions.copy()
    conditions = np.array(conditions)

    X_predict = np.array(conditions)
    if len(X_predict.shape) == 1:
        X_predict = X_predict.reshape(-1, 1)

    model_disagreement = list()

    # collect diagreements for each model pair
    for model_a, model_b in itertools.combinations(models, 2):

        # determine the prediction method
        if hasattr(model_a, "predict_proba") and hasattr(model_b, "predict_proba"):
            model_a_predict = model_a.predict_proba
            model_b_predict = model_b.predict_proba
        elif hasattr(model_a, "predict") and hasattr(model_b, "predict"):
            model_a_predict = model_a.predict
            model_b_predict = model_b.predict
        else:
            raise AttributeError(
                "Models must both have `predict_proba` or `predict` method."
            )

        # get predictions from both models
        y_a = model_a_predict(X_predict)
        y_b = model_b_predict(X_predict)

        assert y_a.shape == y_b.shape, "Models must have same output shape."

        # determine the disagreement between the two models in terms of mean-squared error
        if len(y_a.shape) == 1:
            disagreement = (y_a - y_b) ** 2
        else:
            disagreement = np.mean((y_a - y_b) ** 2, axis=1)

        model_disagreement.append(disagreement)

    assert len(model_disagreement) >= 1, "No disagreements to compare."

    # sum up all model disagreements
    summed_disagreement = np.sum(model_disagreement, axis=0)

    if isinstance(condition_pool_copy, pd.DataFrame):
        conditions = pd.DataFrame(conditions, columns=condition_pool_copy.columns)
    else:
        conditions = pd.DataFrame(conditions)

    # normalize the distances
    scaler = StandardScaler()
    score = scaler.fit_transform(summed_disagreement.reshape(-1, 1)).flatten()

    # order rows in Y from highest to lowest
    conditions["score"] = score
    conditions = conditions.sort_values(by="score", ascending=False)

    if num_samples is None:
        return conditions
    else:
        return conditions.head(num_samples)



def sample(conditions: Union[pd.DataFrame, np.ndarray],
           models: List,
           num_samples: int = 1):
    """
    A experimentalist that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        X: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool
    """

    selected_conditions = score_sample(conditions, models, num_samples)
    selected_conditions.drop(columns=["score"], inplace=True)

    return selected_conditions


model_disagreement_sample = sample
model_disagreement_score_sample = score_sample
model_disagreement_sampler = deprecated_alias(
    model_disagreement_sample, "model_disagreement_sampler")
