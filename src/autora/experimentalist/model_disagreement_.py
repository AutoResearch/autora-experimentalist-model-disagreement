import itertools
from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from autora.state.delta import Result, State, wrap_to_use_state


def model_disagreement_sample_on_state(s: State, **kwargs) -> State:
    """Wrapper on [model_disagreement_sample][] which uses the `State` mechanism."""
    return wrap_to_use_state(model_disagreement_sample)(s, **kwargs)


def model_disagreement_sample(
    conditions: Union[pd.DataFrame, Iterable, Dict, np.ndarray, np.recarray],
    models: List[BaseEstimator],
    num_samples: int = 1,
):
    """
    A sampler that returns selected samples for independent variables
    for which the models disagree the most in terms of their predictions.

    Args:
        conditions: pool of IV conditions to evaluate in terms of model disagreement
        models: List of Scikit-learn (regression or classification) models to compare
        num_samples: number of samples to select

    Returns: Sampled pool

    Examples:
        >>> from sklearn.dummy import DummyRegressor
        >>> from sklearn.linear_model import LinearRegression

        We compare two models – y = 1, and y = x + 1, which intercept at
        x=0 (minimum disagreement) and increase linearly in disagreement away from 0.
        >>> models = [
        ...     DummyRegressor(strategy="constant", constant=1).fit([(0,)], [1]),
        ...     LinearRegression(fit_intercept=True).fit([(-1,), (0,), (1,)], [0, 1, 2])
        ... ]

        The function returns a `Result` object:
        >>> model_disagreement_sample(
        ...     conditions={"x": [-2, -1, 0, 1]},
        ...     models=models,
        ...     num_samples=3
        ... )
        {'conditions':    x
        0 -2
        1 -1
        3  1}

        In an obvious case where we have conditions at 0 (agreement) and 1 (disagreement),
        1 is chosen.
        >>> model_disagreement_sample(
        ...     conditions={"x": [0, 1]}, models=models)["conditions"]
           x
        1  1

        In a less obvious case where we have conditions at -1 and 1 (equal disagreement),
        the first (-1) is chosen
        >>> model_disagreement_sample(
        ...     conditions={"x": [-1, 1]}, models=models)["conditions"]
           x
        0 -1

        If we reorder the conditions, the first is still chosen:
        >>> model_disagreement_sample(
        ...     conditions={"x": [1, -1]}, models=models)["conditions"]
           x
        0  1

        If we ask for as many samples as there are potential conditions, we get them all:
        >>> model_disagreement_sample(
        ...     conditions={"x": [1, -1]}, models=models, num_samples=2
        ... )["conditions"]
           x
        0  1
        1 -1

        If we ask for more samples, we get all the conditions (fewer than requested):
        >>> model_disagreement_sample(
        ...     conditions={"x": [1, -1]}, models=models, num_samples=3
        ... )["conditions"]
           x
        0  1
        1 -1

        The conditions are returned in order of their magnitude of disagreement:
        >>> model_disagreement_sample(
        ...     conditions={"x": [-1, 0, 2]}, models=models, num_samples=3
        ... )["conditions"]
           x
        2  2
        0 -1
        1  0

        Requesting zero samples returns an empty dataframe:
        >>> model_disagreement_sample(
        ...     conditions={"x": [-1, 0, 2]}, models=models, num_samples=0
        ... )["conditions"]
        Empty DataFrame
        Columns: [x]
        Index: []

        For a function with two variables, we provide two dimentsional conditions:
        >>> x1, x2 = [g.ravel() for g in np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))]
        >>> conditions = pd.DataFrame({"x1": x1, "x2": x2})
        >>> conditions
            x1   x2
        0 -1.0 -1.0
        1  0.0 -1.0
        2  1.0 -1.0
        3 -1.0  0.0
        4  0.0  0.0
        5  1.0  0.0
        6 -1.0  1.0
        7  0.0  1.0
        8  1.0  1.0

        The models are y = 0 and y = 2 * x1 + 3 * x2 + 2
        >>> models = [
        ...     DummyRegressor(strategy="constant", constant=0).fit([(0, 0)], [1]),
        ...     LinearRegression(fit_intercept=True).fit(
        ...         [(-1, -1), (0, 0), (0, 1), (1, 1)],
        ...         [       -3,      2,      5,     7])
        ... ]

        For ease of understanding, the values of the two models at the grid positions are:
        >>> conditions.assign(m0 = models[0].predict(conditions)
        ...          ).assign(m1 = models[1].predict(conditions)).round(1)
            x1   x2  m0   m1
        0 -1.0 -1.0   0 -3.0
        1  0.0 -1.0   0 -1.0
        2  1.0 -1.0   0  1.0
        3 -1.0  0.0   0  0.0
        4  0.0  0.0   0  2.0
        5  1.0  0.0   0  4.0
        6 -1.0  1.0   0  3.0
        7  0.0  1.0   0  5.0
        8  1.0  1.0   0  7.0

        The largest disagreement m1 - m0 is at (x1, x2) = (1, 1) -> m1 - m0 = 7
        >>> model_disagreement_sample(
        ...     conditions=conditions, models=models, num_samples=1
        ... )["conditions"]
            x1   x2
        8  1.0  1.0

        The 2nd-largest disagreement is at (x1, x2) = (0, 1) -> m1 - m0 = 5
        The 3rd-largest disagreement is at (x1, x2) = (1, 0) -> m1 - m0 = 4
        The equal 4th- largest disagreement is at (x1, x2) = {(-1, -1), (-1, 1)} -> m1 - m0 = 3
        >>> model_disagreement_sample(
        ...     conditions=conditions, models=models, num_samples=5
        ... )["conditions"]
            x1   x2
        8  1.0  1.0
        7  0.0  1.0
        5  1.0  0.0
        6 -1.0  1.0
        0 -1.0 -1.0

        Note here that although (-1, -1) comes first in the condition array
            and the values at (-1, -1) and (-1, 1) should be identical,
            due to floating point errors the disagreement is calculated to be
            (-1,  1) -> 9.00000000000001
            (-1, -1) -> 8.999999999999984
            so the value (-1, -1) comes first.

        If we try to retrieve more samples than there are conditions, we get all the conditions:
        >>> model_disagreement_sample(
        ...     conditions=conditions, models=models, num_samples=1_000
        ... )["conditions"]
            x1   x2
        8  1.0  1.0
        7  0.0  1.0
        5  1.0  0.0
        6 -1.0  1.0
        0 -1.0 -1.0
        4  0.0  0.0
        1  0.0 -1.0
        2  1.0 -1.0
        3 -1.0  0.0

    """
    conditions_ = pd.DataFrame(conditions)

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
        y_a = model_a_predict(conditions_)
        y_b = model_b_predict(conditions_)

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

    # sort the summed disagreements and select the top n
    idx = (-summed_disagreement).argsort()[:num_samples]

    return Result(conditions=conditions_.loc[idx])
