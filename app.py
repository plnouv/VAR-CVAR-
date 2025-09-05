from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
from data_loader import get_data
from var_cvar import Historical_VAR, Historical_CVAR, Parametric_VAR, Parametric_CVAR, Monte_Carlo_VAR, Monte_Carlo_CVAR
from monte_carlo import mc_logsum
from plot import plot_hist_with_lines, compute_xlim, plot_value_panels

# UI streamlit and sidebar
st.set_page_config(page_title="VaR Distributions", layout="wide")
st.title("Value at Risk (VaR) and Conditional value at Risk (CVAR) distributions for a given confidence level.")
st.sidebar.header("Data")
src = st.sidebar.radio("Source", ["Yahoo Finance", "CSV"], horizontal=True)

if src == "Yahoo Finance":
    tickers_str = st.sidebar.text_input("Tickers (add comma between stocks)", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    start = st.sidebar.date_input("Start", value=date.today() - timedelta(days=3*365))
    end   = st.sidebar.date_input("End",   value=date.today())
    load  = st.sidebar.button("Load Data")
else:
    up   = st.sidebar.file_uploader("Price CSV (wide or Date/Ticker/Close)", type=["csv"])
    load = st.sidebar.button("Load Data")

# Show weights afte load
if st.session_state.get("returns") is not None and not st.session_state["returns"].empty:
    cols = list(st.session_state["returns"].columns)
    st.sidebar.markdown("### Portfolio Weights (%)")

    if "weights_map" not in st.session_state or set(st.session_state["weights_map"].keys()) != set(cols):
        eq = 100.0 / max(1, len(cols))
        st.session_state["weights_map"] = {t: eq for t in cols}

    use_equal = st.sidebar.checkbox("Use equal weights", value=True)
    if use_equal:
        eq = 100.0 / max(1, len(cols))
        for t in cols:
            st.sidebar.slider(f"{t} weight (%)", 0, 100, int(round(eq)), 1, disabled=True, key=f"w_{t}_disabled")
        weights_map = {t: eq for t in cols}
    else:
        tmp = {}
        for t in cols:
            tmp[t] = st.sidebar.slider(f"{t} weight (%)", 0, 100, int(round(st.session_state["weights_map"].get(t, 0.0))), 1, key=f"w_{t}")
        tot = sum(tmp.values())
        st.sidebar.write(f"**Sum:** {tot:.0f}%")
        if tot > 0 and st.sidebar.checkbox("Auto-normalize to 100%", value=True):
            weights_map = {t: (v / tot) * 100.0 for t, v in tmp.items()}
            st.sidebar.caption("Auto-normalized to 100%.")
        else:
            weights_map = tmp
        st.session_state["weights_map"] = weights_map

    st.sidebar.dataframe(
        pd.DataFrame({"Ticker": cols, "Weight (%)": [round(weights_map[t], 2) for t in cols]}),
        hide_index=True, use_container_width=True
    )

# streamlir sidebar
st.sidebar.markdown("---")
rolling_window  = st.sidebar.slider("Rolling window (days)", 1, 60, 20, 1,
                                    help="Days over which loss is measured.")
confidence      = st.sidebar.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
alpha_pct       = int(round((1 - confidence) * 100))
notional        = st.sidebar.number_input("Portfolio value ($)", min_value=1000, value=100000, step=1000)
df_student      = st.sidebar.number_input("Student-t deg_free", min_value=3, value=6, step=1)
sims            = st.sidebar.number_input("Monte-Carlo simulations", min_value=200, value=200, step=100)
axis_mode       = st.sidebar.selectbox("Axis", ["Dollars ($)", "Return (frac)"], index=0)
st.sidebar.markdown("---")
calc = st.sidebar.button("Compute & Plot")

# -Loading process
if load:
    try:
        if src == "Yahoo Finance":
            returns, _, _ = get_data(tickers=tickers, start=str(start), end=str(end), csv_bytes=None)
        else:
            if up is None:
                st.error("Please load a CSV first.")
                st.stop()
            returns, _, _ = get_data(tickers=None, start=None, end=None, csv_bytes=up.read())
        st.session_state["returns"] = returns
        st.success("Data successfully loaded.")
        st.rerun()
    except Exception as e:
        st.error(f"Loading error: {e}")

returns = st.session_state.get("returns")
if returns is None or returns.empty:
    st.info("Load some data to continue.")
    st.stop()

# setting up the weights
if st.session_state.get("weights_map"):
    w = np.array([st.session_state["weights_map"].get(t, 0.0) / 100.0 for t in returns.columns], float)
    s = w.sum()
    w = w / s if s > 0 else np.ones(len(w)) / len(w)
else:
    w = np.ones(returns.shape[1], float) / returns.shape[1]

# pd.Series
port_simple = pd.Series(returns.values @ w, index=returns.index, name="port_ret_simple")
agg_ret     = np.log1p(port_simple).rolling(rolling_window).sum().dropna()

if axis_mode == "Dollars ($)":
    series_values = (agg_ret * notional).values
    x_label = "Aggregate P&L ($)"
else:
    series_values = agg_ret.values
    x_label = "Aggregate return (log-sum)"

# End computation
if calc:
    horizon = rolling_window  # align h and mc steps

    # Historical
    h_var  = float(Historical_VAR(agg_ret, alpha=alpha_pct))
    h_cvar = float(Historical_CVAR(agg_ret, alpha=alpha_pct))

    # Parametric
    mu_p = float(agg_ret.mean())
    sd_p = float(agg_ret.std(ddof=1))
    p_var_norm  = float(Parametric_VAR(mu_p, sd_p, distribution="normal",  alpha=alpha_pct, deg_free=df_student))
    p_cvar_norm = float(Parametric_CVAR(mu_p, sd_p, distribution="normal",  alpha=alpha_pct, deg_free=df_student))
    p_var_t     = float(Parametric_VAR(mu_p, sd_p, distribution="student", alpha=alpha_pct, deg_free=df_student))
    p_cvar_t    = float(Parametric_CVAR(mu_p, sd_p, distribution="student", alpha=alpha_pct, deg_free=df_student))

    # Monte Carlo 
    mc_sum = mc_logsum(returns=returns, weights=w, horizon=int(horizon), sims=int(sims))
    mc_var  = Monte_Carlo_VAR(mc_sum, alpha=alpha_pct)
    mc_cvar = Monte_Carlo_CVAR(mc_sum, alpha=alpha_pct)

    # dollars and reurns
    if axis_mode == "Dollars ($)":
        to_dollars = lambda x: x * notional
        h_var_x,  h_cvar_x        = to_dollars(h_var),       to_dollars(h_cvar)
        p_var_norm_x, p_cvar_norm_x = to_dollars(p_var_norm), to_dollars(p_cvar_norm)
        p_var_t_x,  p_cvar_t_x    = to_dollars(p_var_t),     to_dollars(p_cvar_t)
        mc_var_x,  mc_cvar_x      = to_dollars(mc_var),      to_dollars(mc_cvar)
        mc_values = mc_sum.values * notional
    else:
        h_var_x,  h_cvar_x        = h_var,       h_cvar
        p_var_norm_x, p_cvar_norm_x = p_var_norm, p_cvar_norm
        p_var_t_x,  p_cvar_t_x    = p_var_t,     p_cvar_t
        mc_var_x,  mc_cvar_x      = mc_var,      mc_cvar
        mc_values = mc_sum.values


    # Final computation (param/mc)
    xlim = compute_xlim(series_values, mc_values,
                        [h_var_x, h_cvar_x, p_var_norm_x, p_cvar_norm_x, p_var_t_x, p_cvar_t_x, mc_var_x, mc_cvar_x])

    
    c1, c2 = st.columns(2)
    # Hist
    with c1:
        st.pyplot(
            plot_hist_with_lines(
                series_values,
                [h_var_x, h_cvar_x],
                [f"Historical VaR {100-alpha_pct}%", "Historical CVaR"],
                ["red", "black"],                
                ["--", ":"],
                f"Historical ({rolling_window} d)",
                x_label=x_label, xlim=xlim
        ),
        use_container_width=True
    )
    # Param
    with c2:
        st.pyplot(
            plot_hist_with_lines(
                series_values,
                [p_var_norm_x, p_cvar_norm_x, p_var_t_x, p_cvar_t_x],
                ["Parametric Normal VaR", "Parametric Normal CVaR",
                "Parametric Student VaR", "Parametric Student CVaR"],
                ["tab:red", "tab:red", "tab:purple", "tab:purple"],  
                ["--", ":", "--", ":"],
                f"Parametric — Normal vs Student ({rolling_window} d)",
                x_label=x_label, xlim=xlim
        ),
        use_container_width=True
    )

    # MC
    st.pyplot(
        plot_hist_with_lines(
            mc_values,
            [mc_var_x, mc_cvar_x],
            [f"MC VaR {100-alpha_pct}%", "MC CVaR"],
            ["red", "black"],                 # <- red / black
            ["--", ":"],
            f"Monte Carlo ({horizon} d, {sims} sims)",
        x_label=x_label, xlim=xlim
    ),
    use_container_width=True
)

    
    aligned = returns.dropna()
    steps   = int(rolling_window)
    n_paths = min(int(sims), 250)

    
    ps = aligned.values @ w
    steps = min(steps, len(ps))
    hist_vals  = float(notional) * np.cumprod(1.0 + ps[-steps:])

    
    mu_d = float(np.log1p(aligned.values @ w).mean())
    param_vals = float(notional) * np.exp(np.cumsum(np.full(steps, mu_d)))

    # Monte Carlo value 
    asset_log = np.log1p(aligned)
    L = np.linalg.cholesky(asset_log.cov().values + 1e-12 * np.eye(returns.shape[1]))
    Z = np.random.normal(size=(n_paths, steps, returns.shape[1]))
    R = (Z @ L.T) + asset_log.mean().values
    mc_vals = float(notional) * np.exp(np.cumsum(R @ w, axis=1))

    st.pyplot(plot_value_panels(hist=hist_vals, param=param_vals, mc=mc_vals,
                                y0=float(notional),
                                titles=("Historical", "Parametric", "Monte Carlo"),
                                y_label="Portfolio value", x_label="Number of steps"),
              use_container_width=True)

    # Summary
    st.markdown("### Summary")
    recap = pd.DataFrame({
        "Method": ["Historical", "Parametric Normal", "Parametric Student", "Monte Carlo"],
        "VaR (return)": [h_var, p_var_norm, p_var_t, mc_var],
        "CVaR (return)": [h_cvar, p_cvar_norm, p_cvar_t, mc_cvar],
        "VaR ($)":  [-h_var*notional,  -p_var_norm*notional,  -p_var_t*notional,  -mc_var*notional],
        "CVaR ($)": [-h_cvar*notional, -p_cvar_norm*notional, -p_cvar_t*notional, -mc_cvar*notional],
    })
    st.dataframe(recap.style.format({
        "VaR (return)":"{:.4%}", "CVaR (return)":"{:.4%}",
        "VaR ($)":"${:,.0f}",   "CVaR ($)":"${:,.0f}"
    }), use_container_width=True)
