"""
Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) finite-sample correction
and Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors.

References:
    - Diebold, F.X. & Mariano, R.S. (1995). "Comparing predictive accuracy."
      Journal of Business & Economic Statistics, 13(3), 253-263.
    - Harvey, D., Leybourne, S., & Newbold, P. (1997). "Testing the equality
      of prediction mean squared errors." International Journal of Forecasting, 13(2), 281-291.
    - Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite,
      Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
      Econometrica, 55(3), 703-708.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def newey_west_variance(series: np.ndarray, bandwidth: int | None = None, h: int = 1) -> float:
    """
    Compute the Newey-West HAC variance estimator for a 1D series.
    
    If bandwidth is None, uses optimal bandwidth = floor(4*(n/100)^(2/9)) or h-1, 
    whichever is larger. Uses Bartlett kernel weights.
    
    Args:
        series: 1D array of values.
        bandwidth: Number of lags to include in HAC. If None, uses heuristic.
        h: Forecast horizon, used for bandwidth heuristic.
        
    Returns:
        The HAC variance estimate.
    """
    series = np.asarray(series)
    n = len(series)
    if n <= 1:
        return np.nan
        
    if bandwidth is None:
        opt_b = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        bandwidth = max(opt_b, h - 1)
        
    bandwidth = min(bandwidth, n - 1)
    
    y = series - np.mean(series)
    gamma_0 = np.sum(y ** 2) / n
    var = gamma_0
    
    for lag in range(1, bandwidth + 1):
        gamma_lag = np.sum(y[lag:] * y[:-lag]) / n
        weight = 1.0 - (lag / (bandwidth + 1.0))
        var += 2.0 * weight * gamma_lag
        
    return var

def hln_correction(dm_stat: float, n: int, h: int = 1) -> tuple[float, float]:
    """
    Apply the Harvey-Leybourne-Newbold (1997) finite-sample correction to the DM statistic.
    
    Args:
        dm_stat: The uncorrected Diebold-Mariano statistic.
        n: Sample size.
        h: Forecast horizon.
        
    Returns:
        tuple containing (corrected_statistic, adjusted_p_value_two_sided)
    """
    if pd.isna(dm_stat) or n <= 1:
        return np.nan, np.nan
        
    multiplier = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    hln_stat = dm_stat * multiplier
    
    p_value = float(2 * stats.t.sf(np.abs(hln_stat), df=n - 1))
    return hln_stat, p_value

def diebold_mariano_test(errors_model: np.ndarray, errors_baseline: np.ndarray, 
                         loss: str = 'mse', h: int = 1, method: str = 'hac') -> dict:
    """
    Diebold-Mariano test for predictive accuracy.
    
    Args:
        errors_model: 1D array of forecast errors from the proposed model.
        errors_baseline: 1D array of forecast errors from the baseline model.
        loss: 'mse' for squared loss, 'mae' for absolute loss, 'custom' for pre-computed differentials.
        h: Forecast horizon.
        method: 'hac' for Newey-West standard errors, 'simple' for basic paired-t.
        
    Returns:
        dict with test statistics, p-values, and interpretation.
    """
    errors_model = np.asarray(errors_model)
    errors_baseline = np.asarray(errors_baseline)
    
    if len(errors_model) != len(errors_baseline):
        raise ValueError("Error arrays must have the same length.")
        
    n = len(errors_model)
    if n == 0:
        raise ValueError("Error arrays are empty.")
        
    if loss == 'mse':
        d = errors_model**2 - errors_baseline**2
    elif loss == 'mae':
        d = np.abs(errors_model) - np.abs(errors_baseline)
    elif loss == 'custom':
        d = errors_model - errors_baseline
    else:
        raise ValueError("Loss must be 'mse', 'mae', or 'custom'.")
        
    mean_d = np.mean(d)
    
    bandwidth_used = None
    if method == 'hac':
        opt_b = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        bandwidth_used = max(opt_b, h - 1)
        bandwidth_used = min(bandwidth_used, n - 1)
        
        var = newey_west_variance(d, bandwidth=bandwidth_used, h=h)
        if var == 0 or np.isnan(var):
            se = np.nan
        else:
            se = np.sqrt(var / n)
    elif method == 'simple':
        if n > 1:
            se = np.std(d, ddof=1) / np.sqrt(n)
        else:
            se = np.nan
    else:
        raise ValueError("Method must be 'hac' or 'simple'.")
        
    if se is None or np.isnan(se) or se == 0:
        dm_stat = np.nan
        p_two_sided = np.nan
        p_one_sided = np.nan
    else:
        dm_stat = mean_d / se
        p_two_sided = float(2 * stats.t.sf(np.abs(dm_stat), df=n - 1))
        # One sided: H1 that model is better (mean_d < 0)
        p_one_sided = float(stats.t.cdf(dm_stat, df=n - 1))
        
    if not np.isnan(p_one_sided) and p_one_sided < 0.05:
        interpretation = "Model is significantly better"
    elif not np.isnan(p_two_sided) and dm_stat > 0 and p_two_sided < 0.05:
        interpretation = "Baseline is significantly better"
    else:
        interpretation = "No significant difference"
        
    return {
        "dm_statistic": dm_stat,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "mean_loss_diff": mean_d,
        "se_loss_diff": se,
        "n_observations": n,
        "bandwidth": bandwidth_used,
        "method": method,
        "interpretation": interpretation
    }

def run_dm_tests(y_true_flat: np.ndarray, errors_dict: dict[str, np.ndarray], h: int = 1) -> pd.DataFrame:
    """
    Batch runner for Diebold-Mariano tests against all baselines.
    
    Args:
        y_true_flat: 1D array of true values (included for interface completeness).
        errors_dict: Dict mapping model_name -> 1D array of forecast errors.
                     Must include a 'ConvLSTM' key.
        h: Forecast horizon.
        
    Returns:
        DataFrame containing test results.
    """
    if 'ConvLSTM' not in errors_dict:
        raise ValueError("errors_dict must contain 'ConvLSTM'")
        
    conv_errors = errors_dict['ConvLSTM']
    results = []
    
    for base_name, base_errors in errors_dict.items():
        if base_name == 'ConvLSTM':
            continue
            
        for loss_type in ['mse', 'mae']:
            res = diebold_mariano_test(conv_errors, base_errors, loss=loss_type, h=h, method='hac')
            
            dm_stat = res['dm_statistic']
            p_val = res['p_value_two_sided']
            mean_diff = res['mean_loss_diff']
            bw = res['bandwidth']
            n = res['n_observations']
            
            dm_hln, p_hln = hln_correction(dm_stat, n, h=h)
            
            results.append({
                'Baseline': base_name,
                'Loss_Type': loss_type.upper(),
                'DM_Statistic': dm_stat,
                'DM_HLN_Statistic': dm_hln,
                'P_Value': p_val,
                'P_Value_HLN': p_hln,
                'Mean_Loss_Diff': mean_diff,
                'Bandwidth': bw,
                'N': n,
                'Significant_5pct': not np.isnan(p_hln) and p_hln < 0.05,
                'Significant_10pct': not np.isnan(p_hln) and p_hln < 0.10
            })
            
    return pd.DataFrame(results)

def fig_dm_results(dm_df: pd.DataFrame, path: str, show: bool = False):
    """
    Generate and save a bar chart of DM statistics.
    
    Args:
        dm_df: DataFrame with test results from run_dm_tests.
        path: Output path for the figure.
        show: Whether to display the plot interactively.
    """
    loss_types = dm_df['Loss_Type'].unique()
    n_losses = len(loss_types)
    
    fig, axes = plt.subplots(1, n_losses, figsize=(6 * n_losses, 5), squeeze=False)
    
    for i, loss in enumerate(loss_types):
        ax = axes[0, i]
        subset = dm_df[dm_df['Loss_Type'] == loss]
        
        baselines = subset['Baseline'].tolist()
        stats_vals = subset['DM_HLN_Statistic'].tolist()
        pvals = subset['P_Value_HLN'].tolist()
        
        colors = []
        for p in pvals:
            if pd.isna(p):
                colors.append('gray')
            elif p < 0.05:
                colors.append('green')
            elif p < 0.10:
                colors.append('gold')
            else:
                colors.append('red')
                
        bars = ax.bar(baselines, stats_vals, color=colors, edgecolor='black', alpha=0.7)
        
        for bar, p in zip(bars, pvals):
            height = bar.get_height()
            y_offset = max(abs(height) * 0.05, 0.1)
            y_pos = height + y_offset if height > 0 else height - y_offset
            ha = 'center'
            va = 'bottom' if height > 0 else 'top'
            if not pd.isna(p):
                ax.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f'p={p:.3f}', 
                        ha=ha, va=va, fontsize=9, rotation=0)
                
        ax.set_title(f"{loss} Loss")
        ax.set_ylabel("DM HLN Statistic")
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=45)
        
    fig.suptitle("Diebold-Mariano Tests: ConvLSTM vs Baselines", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)
