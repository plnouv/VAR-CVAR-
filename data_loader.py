import io
import numpy as np
import pandas as pd
import yfinance as yf


def _parse_tickers(tickers):
    
    if tickers is None:
        return None
    if isinstance(tickers, str):
        s = tickers.replace(",", " ")
        lst = [t.strip().upper() for t in s.split() if t.strip()]
        return lst
    
    return [str(t).upper() for t in tickers if str(t).strip()]


def get_data(tickers=None, start=None, end=None, csv_bytes=None):
    
    if tickers is not None and (isinstance(tickers, str) or len(tickers) > 0):
        tick_list = _parse_tickers(tickers)

        data = yf.download(
            tickers=tick_list,
            start=start,
            end=end,
            auto_adjust=True,   
            progress=False,
        )
        # MultiIndex Df (ticker, close...)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                px = data["Close"].copy()
            else:
                px = data.xs("Adj Close", level=0, axis=1)
            
            cols = [t for t in _parse_tickers(tick_list) if t in px.columns]
            if cols:
                px = px[cols]
        else:
            # 1 ticker
            if "Close" in data.columns:
                px = data["Close"].to_frame()
            elif "Adj Close" in data.columns:
                px = data["Adj Close"].to_frame()
            else:
                raise ValueError("Colonnes 'Close'/'Adj Close' introuvables dans les données yfinance.")
            
            px.columns = [_parse_tickers(tick_list)[0]]

    elif csv_bytes is not None:
        # load from csv
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

        
        if {"Ticker", "Close"}.issubset(set(df.columns)):
            px = df.pivot(index=df.index, columns="Ticker", values="Close")
        
        elif [c.lower() for c in df.columns] == ["close"]:
            px = df.rename(columns={df.columns[0]: "ASSET"})
        
        else:
            px = df

        px.columns = [str(c).upper() for c in px.columns]

    else:
        raise ValueError("Fournir soit 'tickers' (Yahoo) soit 'csv_bytes' (CSV).")

    # Cleaning data
    px = px.sort_index().dropna(how="all")
    returns = px.pct_change().dropna(how="all")
    mean_returns = returns.mean()
    Cov_Matrix = returns.cov()
    return returns, mean_returns, Cov_Matrix


def Portfolio_Performance(weights, mean_returns, Cov_Matrix, Time=1):
    w = np.asarray(weights, dtype=float)
    mu = float(np.sum(w * mean_returns.values)) * float(Time)
    sigma_daily = float(np.sqrt(w.T @ Cov_Matrix.values @ w))
    sigma_h = sigma_daily * np.sqrt(float(Time))
    return mu, sigma_h
