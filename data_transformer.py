import numpy as np
import pandas as pd

def Portfolio_Performance(weights, mean_returns, Cov_Matrix, Time):
    """
      - returns = sum_i w_i * mean_i * Time
      - port_std = sqrt(w' Σ w * Time)
    
    """
    #  Series
    if isinstance(weights, dict):
        w = pd.Series({str(k).upper(): float(v) for k, v in weights.items()})
    elif isinstance(weights, (list, tuple, np.ndarray)):
        w = pd.Series(weights, index=mean_returns.index, dtype=float)
    elif isinstance(weights, pd.Series):
        w = weights.astype(float)

    else:
        raise TypeError("weights doit être un dict, une Series ou un ndarray.")

    w = w.reindex(mean_returns.index, fill_value=0.0)

    # Normalisation
    s = w.sum()
    if s != 0:
        w = w / s
    else:
        w = pd.Series(1.0 / len(mean_returns), index=mean_returns.index)

    port_ret = float((w * mean_returns).sum() * Time)
    port_var = float(np.dot(w.values.T, np.dot(Cov_Matrix.values, w.values)) * Time)
    port_std = float(np.sqrt(port_var))
    return port_ret, port_std
