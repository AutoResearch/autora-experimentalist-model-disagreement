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

def test_multi_dimensional_prediction():

    num_samples = 2

    X = list()
    X.append(np.array([[0, 0], [1, 0], [0, 1]]))
    X.append(np.array([[1, 0], [0, 1], [1, 0]]))
    X.append(np.array([[0, 0], [1, 0], [0, 1]]))
    X.append(np.array([[1, 0], [1, 0], [0, 1]]))

    class dummy_theorist_A():

        def __init__(self):
            return

        def predict(self, X):
            Y = np.array(X)
            Y[0, :] = [0, 0]
            return Y

        def predict_proba(self, X):
            Y = np.array(X)
            Y[0, :] = [0.1, 0.1]
            return Y

    class dummy_theorist_B():
        def __init__(self):
            return

        def predict(self, X):
            Y = np.array(X)
            return Y

        def predict_proba(self, X):
            Y = np.array(X)
            return Y

    model_a = dummy_theorist_A()
    model_b = dummy_theorist_B()

    X_new = model_disagreement_sample(X, [model_a, model_b], num_samples)

    assert isinstance(X_new, pd.DataFrame)

    X_new_list = X_new['X'].tolist()

    assert len(X_new) == num_samples
    assert X_new_list[0].shape == (3, 2) and X_new_list[1].shape == (3, 2)
    assert (np.array_equal(X_new_list[0], [[1, 0], [0, 1], [1, 0]]) or
            np.array_equal(X_new_list[0], [[1, 0], [1, 0], [0, 1]]))
    assert (np.array_equal(X_new_list[1], [[1, 0], [0, 1], [1, 0]]) or
            np.array_equal(X_new_list[1], [[1, 0], [1, 0], [0, 1]]))


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
