"""This module defines the functions used to do a linear regression.

The functions are meant to use a dataset of YOLO detections as explanatory variables and
BIC operators' ratings as scalar response to establish and illustrate a linearregression
with multiple variables model.

"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from .dataloader import _create_dataset_regression


def _fit_regression_model(inputs, outputs):
    reg = LinearRegression()
    reg.fit(inputs, outputs)
    return (reg.intercept_, reg.coef_)


def _plot_regression_model(inputs, outputs):
    plt.title("Linear regression with multiple variables")
    plt.scatter(inputs, outputs, alpha=0.2)
    plt.xlabel("regression_coefficients ∙ explanatory_variables")
    plt.ylabel("rating")
    x = np.array(range(0, 11))
    plt.plot(x, x, color="black")
    plt.savefig("regression.svg")
    plt.show()


def regression(path: str) -> None:
    """Function to fit a linear regression between traces' defects and ratings.

    The function aims to establish if a linear relation between the YOLO defects
    detection outputs and the ratings given by BIC operators exists. The function will
    plot the calculated model and save it in a file "regression.svg".

    Parameters
    ----------
    path : str
        The path to the folder containing YOLO's outputs (one .txt file for each trace).

    """

    if not isinstance(path, str):
        raise TypeError("path must be a string")

    x, y = _create_dataset_regression(path)
    intercept, coef = _fit_regression_model(x, y)
    x_scalar = np.sum(x * coef, axis=1) + intercept
    _plot_regression_model(x_scalar, y)
