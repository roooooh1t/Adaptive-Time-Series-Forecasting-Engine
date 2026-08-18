import numpy as np

# Mean absolute error
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Root Mean Square Error
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Weighted Absolute Percentage Error
def WAPE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

# Weighted  Root Mean Square Error
def WRMSE(y_true, y_pred):
    return np.sqrt(np.sum(len(y_true) * (y_true - y_pred) ** 2) / np.sum(len(y_true)))

# Bias / Mean Error
def mean_error(y_true, y_pred):
    return np.mean(y_pred - y_true)