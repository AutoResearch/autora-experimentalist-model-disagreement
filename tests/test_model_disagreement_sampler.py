import numpy as np
import pandas as pd

from autora.experimentalist.model_disagreement import (
    model_disagreement_sample,
    model_disagreement_score_sample,
)
from autora.theorist.bms import BMSRegressor
from autora.theorist.darts import DARTSRegressor

BMSRegressor()


DARTSRegressor()


def test_output_dimensions():
    # Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X**2).reshape(-1, 1)
    n = 5

    # Theorists
    bms_theorist = BMSRegressor(epochs=10)
    darts_theorist = DARTSRegressor(max_epochs=10)

    bms_theorist.fit(X, y)
    darts_theorist.fit(X, y)

    # Sampler
    X_new = model_disagreement_sample(X, [bms_theorist, darts_theorist], n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])


def test_pandas():
    # Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X**2).reshape(-1, 1)
    n = 5

    X = pd.DataFrame(X)

    # Theorists
    bms_theorist = BMSRegressor(epochs=10)
    darts_theorist = DARTSRegressor(max_epochs=10)

    bms_theorist.fit(X, y)
    darts_theorist.fit(X, y)

    # Sampler
    X_new = model_disagreement_sample(X, [bms_theorist, darts_theorist], n)

    # Check that the sampler returns n experiment conditions
    assert isinstance(X_new, pd.DataFrame)
    assert X_new.shape == (n, X.shape[1])


def test_scoring():
    # Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X**2).reshape(-1, 1)
    n = 5

    X = pd.DataFrame(X)

    # Theorists
    bms_theorist = BMSRegressor(epochs=10)
    darts_theorist = DARTSRegressor(max_epochs=10)

    bms_theorist.fit(X, y)
    darts_theorist.fit(X, y)

    # Sampler
    X_new = model_disagreement_score_sample(X, [bms_theorist, darts_theorist], n)

    # Check that the sampler returns n experiment conditions
    assert isinstance(X_new, pd.DataFrame)
    assert "score" in X_new.columns
    assert X_new.shape == (n, X.shape[1] + 1)
