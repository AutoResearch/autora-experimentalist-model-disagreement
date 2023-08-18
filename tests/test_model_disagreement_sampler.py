from src.autora.experimentalist.model_disagreement import model_disagreement_sample
from autora.theorist.bms import BMSRegressor; BMSRegressor()
from autora.theorist.darts import DARTSRegressor; DARTSRegressor()
import numpy as np
import pandas as pd

def test_output_dimensions():
    #Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X**2).reshape(-1, 1)
    n = 5
    
    #Theorists
    bms_theorist = BMSRegressor()
    darts_theorist = DARTSRegressor()
    
    bms_theorist.fit(X,y)
    darts_theorist.fit(X,y)

    #Sampler
    X_new = model_disagreement_sample(X, [bms_theorist, darts_theorist], n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])

def test_pandas():
    # Meta-Setup
    X = np.linspace(start=-3, stop=6, num=10).reshape(-1, 1)
    y = (X ** 2).reshape(-1, 1)
    n = 5

    X = pd.DataFrame(X)

    # Theorists
    bms_theorist = BMSRegressor()
    darts_theorist = DARTSRegressor()

    bms_theorist.fit(X, y)
    darts_theorist.fit(X, y)

    # Sampler
    X_new = model_disagreement_sample(X, [bms_theorist, darts_theorist], n)

    # Check that the sampler returns n experiment conditions
    assert X_new.shape == (n, X.shape[1])
