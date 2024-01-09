import numpy as np
import xgboost as xgb
from xgboost import DMatrix


def CustomL1Loss(predt: np.ndarray, dtrain: DMatrix):
    """Custom L1 loss function for XGBoost.

    Parameters
    ----------
    predt : np.ndarray
        The predicted values.
    dtrain : DMatrix
        The training data.

    Returns
    -------
    grad : np.ndarray
        The first order gradients.
    hess : np.ndarray
        The second order gradients.
    """
    predt = predt.reshape(-1, 1)
    label = dtrain.get_label()
    grad = 2 * (predt - label)
    hess = np.repeat(2, label.shape[0])
    return grad, hess
