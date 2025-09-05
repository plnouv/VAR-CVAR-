import numpy as np
import pandas as pd

# simulation log price
def mc_logsum(returns: pd.DataFrame, weights, horizon=20, sims=20000):
    """
    Simule des log-retours corrélés (Normal) sur 'horizon' jours
    et renvoie un échantillon (Series) des retours agrégés (somme des log-retours).
    - returns : DataFrame de rendements simples quotidiens (colonnes = tickers)
    - weights : array-like (somme ≈ 1), aligné à returns.columns
    """
    log_ret = np.log1p(returns).dropna()
    mu  = log_ret.mean().values              # (n,)
    cov = log_ret.cov().values               # (n,n)
    w   = np.asarray(weights, float).reshape(-1)
    n   = len(w)

    # Cholesky
    L = np.linalg.cholesky(cov)              # (n,n)
    Z = np.random.normal(size=(n, horizon, sims))            # (n,h,s)
    daily = (L @ Z.reshape(n, horizon*sims)).reshape(n, horizon, sims) + mu[:, None, None]
    port_daily = np.tensordot(w, daily, axes=(0, 0))         # (h,s)
    mc = port_daily.sum(axis=0)                               # (s,)
    return pd.Series(mc)
