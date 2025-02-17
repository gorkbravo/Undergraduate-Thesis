import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import log, sqrt, exp
from scipy.stats import norm
from datetime import datetime  # For timestamp


def sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T):
    """Extended Hagan formula for SABR implied vol, with second-order corrections."""
    if np.isclose(F, K, atol=1e-14):
        leading = alpha / (F**(1.0 - beta))
        A1 = ((1-beta)**2 / 24.0) * (alpha**2 / (F**(2*(1-beta))))
        A2 = 0.25 * rho * beta * nu * alpha / (F**(1.0-beta))
        A3 = ((2.0 - 3.0*rho**2)/24.0) * (nu**2)
        correction = 1.0 + (A1 + A2 + A3)*T
        return max(leading*correction, 1e-8)

    z = (nu / alpha)*(F**(1.0 - beta) - K**(1.0 - beta))
    eps = 1e-14
    numerator = np.sqrt(1.0 - 2.0*rho*z + z*z) + z - rho
    denominator = (1.0 - rho)
    x_z = np.log((numerator + eps)/(denominator + eps))

    denom = (F*K)**((1.0 - beta)/2.0)
    fk_1minusbeta = (F*K)**(1.0 - beta)
    A1 = ((1.0-beta)**2 / 24.0) * (alpha**2 / fk_1minusbeta)
    A2 = 0.25 * rho * beta * nu * alpha / ((F*K)**((1.0-beta)/2.0))
    A3 = ((2.0 - 3.0*rho**2)/24.0) * (nu**2)
    correction = 1.0 + (A1 + A2 + A3)*T

    sabr_vol = (alpha/denom)*(z/x_z)*correction
    return max(sabr_vol, 1e-8)


def sabr_objective_extended(x, beta, F, T, strikes, market_vols):
    alpha, rho, nu = x
    model_vols = [
        sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T)
        for K in strikes
    ]
    return np.array(market_vols) - np.array(model_vols)


def black_76_call_price(F, K, r, T, sigma):
    """
    Black-76 call price formula:
      Price = e^(-r*T) [ F * Phi(d1) - K * Phi(d2) ]
      where
      d1 = [ln(F/K) + 0.5 * sigma^2 * T] / (sigma * sqrt(T))
      d2 = d1 - sigma * sqrt(T)
    """
    if T <= 0:
        return max(F - K, 0.0)

    d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    discount = np.exp(-r * T)
    call_price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price


def plot_sabr_skew_and_errors(strikes, market_vols, sabr_vols):
    """
    Plots:
      1) Market vs. SABR Vol Skew
      2) (MarketVol - SABRVol) error scatter
    """
    # 1) Vol Skew
    plt.figure(figsize=(8,5))
    plt.scatter(strikes, market_vols, color='blue', label='Market Vol', alpha=0.7)
    idx_sort = np.argsort(strikes)
    plt.plot(strikes[idx_sort], np.array(sabr_vols)[idx_sort],
             color='red', label='SABR Fit')
    plt.title("Market vs. SABR Implied Vol Skew")
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2) Error Plot
    errors = np.array(market_vols) - np.array(sabr_vols)
    plt.figure(figsize=(8,5))
    plt.axhline(y=0, color='black', linestyle='--')
    plt.scatter(strikes, errors, color='purple', alpha=0.7)
    plt.title("Vol Error: Market − SABR")
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol Error")
    plt.grid(True)
    plt.show()


def pdf_statistics(K_grid, pdf_grid):
    """
    Compute mean, variance, skewness, kurtosis from a discrete PDF on [K_min, K_max].
    pdf_grid[i] approximates f(K_i).
    We assume sum(pdf_grid * deltaK) ~ 1.
    """
    mask = ~np.isnan(pdf_grid)
    K_vals = K_grid[mask]
    f_vals = pdf_grid[mask]

    if len(K_vals) < 3:
        return {}

    total_prob = 0.0
    first_moment = 0.0
    second_moment = 0.0
    third_moment = 0.0
    fourth_moment = 0.0

    for i in range(len(K_vals)-1):
        K_i = K_vals[i]
        K_i1 = K_vals[i+1]
        f_i = f_vals[i]
        f_i1 = f_vals[i+1]
        dK = (K_i1 - K_i)

        # trapezoid for PDF
        avg_f = 0.5*(f_i + f_i1)
        total_prob += avg_f * dK

        # 1st moment
        avg_fK = 0.5*(K_i*f_i + K_i1*f_i1)
        first_moment += avg_fK * dK

        # 2nd moment
        avg_fK2 = 0.5*(K_i*K_i*f_i + K_i1*K_i1*f_i1)
        second_moment += avg_fK2 * dK

        # 3rd moment
        avg_fK3 = 0.5*(K_i**3 * f_i + K_i1**3 * f_i1)
        third_moment += avg_fK3 * dK

        # 4th moment
        avg_fK4 = 0.5*(K_i**4 * f_i + K_i1**4 * f_i1)
        fourth_moment += avg_fK4 * dK

    if total_prob < 1e-12:
        return {}

    mean = first_moment / total_prob
    var = second_moment / total_prob - mean**2

    if var < 1e-12:
        return {
            "mean": mean,
            "variance": var,
            "stdev": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "total_prob": total_prob
        }

    stdev = np.sqrt(var)
    # approximate approach for skew & kurt:
    skew = (third_moment / total_prob - 3*mean*var - mean**3) / (stdev**3)
    E_K3 = third_moment / total_prob
    E_K4 = fourth_moment / total_prob
    cent4 = E_K4 - 4*mean*E_K3 + 6*(mean**2)*(second_moment / total_prob) - 3*(mean**4)
    kurt = cent4 / (stdev**4)

    return {
        "mean": mean,
        "variance": var,
        "stdev": stdev,
        "skewness": skew,
        "kurtosis": kurt,
        "total_prob": total_prob
    }


def main(csv_path, F, T, beta):
    """Main SABR engine that dynamically accepts cleaned file, F, T, and beta."""
    # A) Load data & filter
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "call"].copy()

    # B) Prepare strikes & vols
    strikes = df["strike"].values
    market_vols = df["impliedVolatility"].values

    # *** Risk-free rate set to 4.163% (0.04163) ***
    r = 0.04163

    # D) Calibrate alpha, rho, nu
    x0 = [0.5, 0.0, 0.5]
    lower_bounds = [0.0, -1.0, 0.0]
    upper_bounds = [np.inf, 1.0, 1000.0]
    res = least_squares(
        sabr_objective_extended,
        x0,
        bounds=(lower_bounds, upper_bounds),
        args=(beta, F, T, strikes, market_vols),
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )
    alpha_calib, rho_calib, nu_calib = res.x
    print("===== Extended SABR Calibration =====")
    print(f" Converged: {res.success}, {res.message}")
    print(f"  Beta = {beta}")
    print(f"  Alpha = {alpha_calib:.6f},  Rho = {rho_calib:.6f},  Nu = {nu_calib:.6f}")

    # Calculate MSE & RMSE
    sabr_vols_market = [
        sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, F, K, T)
        for K in strikes
    ]
    errors = np.array(market_vols) - np.array(sabr_vols_market)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    print(f"\nSABR Fit MSE = {mse:.6f}")
    print(f"SABR Fit RMSE = {rmse:.6f}\n")

    # E) Plot the Market vs. SABR Skew & Errors
    plot_sabr_skew_and_errors(strikes, market_vols, sabr_vols_market)

    # F) Dense strike grid for 'continuous' call prices
    strike_range = min(F * 0.8, 150)  # Use either 80% of F or 150, whichever is smaller
    K_min = max(F - strike_range, 1)  # Ensure minimum strike is positive
    K_max = F + strike_range
    n_points = 4000
    K_grid = np.linspace(K_min, K_max, n_points)

    call_prices = []
    for K in K_grid:
        sabr_vol = sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, F, K, T)
        c_price = black_76_call_price(F, K, r, T, sabr_vol)
        call_prices.append(c_price)

    # G) Approx second derivative => PDF
    pdf_grid = np.full(n_points, np.nan)
    for i in range(1, n_points-1):
        K_minus = K_grid[i-1]
        K_i = K_grid[i]
        K_plus = K_grid[i+1]

        C_minus = call_prices[i-1]
        C_i = call_prices[i]
        C_plus = call_prices[i+1]

        # central difference for non-uniform spacing
        dC_plus = (C_plus - C_i)/(K_plus - K_i)
        dC_minus = (C_i - C_minus)/(K_i - K_minus)
        second_deriv = 2.0*(dC_plus - dC_minus)/(K_plus - K_minus)

        pdf_grid[i] = np.exp(r*T)*second_deriv

    # H) Plot call price & PDF in one figure
    fig, ax1 = plt.subplots(figsize=(8,6))
    color_call = "gray"
    color_pdf = "darkred"

    ax1.plot(K_grid, call_prices, color=color_call, label="Call Price")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Call price", color=color_call)
    ax1.tick_params(axis='y', labelcolor=color_call)

    ax2 = ax1.twinx()
    ax2.plot(K_grid, pdf_grid, color=color_pdf, label="f(K)")
    ax2.set_ylabel("f(K)", color=color_pdf)
    ax2.tick_params(axis='y', labelcolor=color_pdf)

    plt.axvline(x=F, color='k', linestyle='--')
    plt.title("Call Price & Implied PDF from SABR (Black-76)")
    fig.tight_layout()
    plt.show()

    # I) Plot PDF alone
    plt.figure(figsize=(8,6))
    plt.plot(K_grid, pdf_grid, color=color_pdf)
    plt.title("Risk-Neutral PDF vs. Strike (Black-76)")
    plt.xlabel("Strike")
    plt.ylabel("f(K)")
    plt.grid(True)
    plt.show()

    # J) PDF statistics
    stats = pdf_statistics(K_grid, pdf_grid)
    print("===== PDF Statistics =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # K) Append PDF statistics + run info to CSV
    stats_csv_path = "Add output File path"

    # If stats dict is empty, create a row of None
    if not stats:
        stats = {
            "mean": None, "variance": None, "stdev": None,
            "skewness": None, "kurtosis": None, "total_prob": None
        }

    # Build a row with the desired order so the first column is the timestamp.
    # Also use only the *filename* for 'csv_file'.
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": os.path.basename(csv_path),
        "alpha": alpha_calib,
        "rho": rho_calib,
        "nu": nu_calib,
        "beta": beta,
        "MSE": mse,
        "RMSE": rmse,
        "mean": stats["mean"],
        "variance": stats["variance"],
        "stdev": stats["stdev"],
        "skewness": stats["skewness"],
        "kurtosis": stats["kurtosis"],
        "totalprob": stats["total_prob"],
    }

    # Convert to DataFrame
    stats_df = pd.DataFrame([row_data])

    # Check if file exists; if not, write header. If yes, append without header.
    file_exists = os.path.isfile(stats_csv_path)
    stats_df.to_csv(stats_csv_path, mode="a", index=False, header=not file_exists)


if __name__ == "__main__":
    # Example usage
    main(
        csv_path="Add specific file for testing",
        F=71.98,
        T=4.0 / 52.0,
        beta=0.25
    )

