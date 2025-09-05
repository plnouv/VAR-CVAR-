import pandas as pd
import numpy as np
from scipy.stats import norm, t

# HISTORICAL
def Historical_VAR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(Historical_VAR, alpha=alpha)
    
    else:
        raise TypeError('Returns need to be in Series or df format')

def Historical_CVAR(returns, alpha):
    if isinstance(returns, pd.Series):
        below_var = returns <= Historical_VAR(returns, alpha=alpha)
        return returns[below_var].mean()

    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(Historical_CVAR, alpha=alpha)
    
    else:
        raise TypeError('Returns need to be in Series or df format')
    
# PARAMETRICS
def Parametric_VAR(port_returns, port_std, distribution='normal', alpha=5, deg_free=6):
    if distribution == 'normal':
        var = port_returns + norm.ppf(alpha/100) * port_std

    elif distribution == 'student':
        nu = deg_free
        var = port_returns + t.ppf(alpha/100, nu) * (port_std * np.sqrt((nu - 2) / nu) if nu > 2 else port_std)

    else:
        raise TypeError('Distribution need to be normal or student')
    return var

def Parametric_CVAR(port_returns, port_std, distribution='normal', alpha=5, deg_free=6):
    if distribution == 'normal':
        cvar = port_returns - port_std * (norm.pdf(norm.ppf(alpha/100)) / (alpha/100))

    elif distribution == 'student':
        nu = deg_free
        x = t.ppf((alpha/100), nu)  
        cvar = port_returns - (port_std * (np.sqrt((nu - 2) / nu) if nu > 2 else 1.0)) * (((nu + x**2) / ((nu - 1) * (alpha/100))) * t.pdf(x, nu))

    else:
        raise TypeError('Distribution need to be normal or student')
    
    return cvar

# MONTE-CARLO
def Monte_Carlo_VAR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return float(np.percentile(returns.values, alpha))
    
    else:
        raise TypeError("Returns need to be a pandas Series")

def Monte_Carlo_CVAR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return float(returns[returns <= np.percentile(returns.values, alpha)].mean())
    
    else:
        raise TypeError("Returns need to be a pandas Series")