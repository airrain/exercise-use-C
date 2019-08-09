import numpy as np
import matplotlib.pyplot as plt
n_dots = 200

X = np.linspace(0, 1, n_dots)                   
y = np.sqrt(X) + 0.2*np.random.rand(n_dots) - 0.1
X = X.reshape(-1,1)
y = y.reshape(-1,1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression

def polynomial_model(degree = 1):
    polynomial_features = PolynomialFeatures(degree = degree,include_bias = False)
    linear_regression= LinearRegression()
    pipeline = Pipeline([("PolynomialFeatures",PolynomialFeatures),("linear_regression",linear_regression)
    ])
    return pipeline
    
    

