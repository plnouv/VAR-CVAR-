import numpy as np
import matplotlib.pyplot as plt


# var and cvar dips
def compute_xlim(series_values, mc_values=None, extra_points=None,
                 lo_q=0.3, hi_q=99.7, left_pad=0.18, right_pad=0.07):
    
    vals = np.asarray(series_values, dtype=float)
    if mc_values is not None:
        vals = np.concatenate([vals, np.asarray(mc_values, dtype=float)])
    if extra_points is not None and len(extra_points) > 0:
        vals = np.concatenate([vals, np.asarray(extra_points, dtype=float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    x_min = np.percentile(vals, lo_q)
    x_max = np.percentile(vals, hi_q)
    span = max(1e-12, x_max - x_min)
    return (x_min - left_pad*span, x_max + right_pad*span)


def plot_hist_with_lines(values, lines=(), labels=(), colors=(), styles=(),
                         title="", x_label="", bins=50, xlim=None,
                         figsize=(7.6, 4.3), dpi=120):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(values, bins=bins, density=True, alpha=0.8, edgecolor="white")
    for i, x in enumerate(lines):
        lab = labels[i] if i < len(labels) else None
        col = colors[i] if i < len(colors) else None
        sty = styles[i] if i < len(styles) else "--"
        ax.axvline(x, color=col, linestyle=sty, linewidth=2, label=lab)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    if labels:
        ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig



# line, col (3,1)
def plot_3_graĥ(a):
    a = np.asarray(a)
    return a[None, :] if a.ndim == 1 else a

def plot_value_panels(
    hist=None,
    param=None,
    mc=None,
    *,
    y0=None,                       
    titles=("Historical", "Parametric", "Monte Carlo"),
    y_label="Portfolio value",
    x_label="Number of steps",
    max_lines=200,
    figsize=(15, 4.2),
    dpi=120,
    sharey=True,
):
    
    
    panels = [hist, param, mc]
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharex=True, sharey=sharey)

    for ax, arr, title in zip(axes, panels, titles):
        if arr is None:
            ax.axis("off")
            continue

        P = plot_3_graĥ(arr)
        n_show = min(P.shape[0], max_lines)
        x = np.arange(P.shape[1])

        ax.plot(x, P[:n_show].T, lw=0.9, alpha=0.7)
        if y0 is not None:
            ax.axhline(y0, color="tab:blue", lw=1.2, alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(y_label)
    fig.tight_layout()
    return fig
