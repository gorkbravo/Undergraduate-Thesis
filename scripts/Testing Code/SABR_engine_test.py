import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import log, sqrt, exp
from scipy.stats import norm

###############################################################################
# 1) Extended Hagan SABR
###############################################################################
def sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T):
    """
    Extended Hagan formula for SABR implied vol, with second-order corrections.
    """
    if np.isclose(F, K, atol=1e-14):
        # near ATM
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
    # second-order expansions
    fk_1minusbeta = (F*K)**(1.0 - beta)
    A1 = ((1.0-beta)**2 / 24.0) * (alpha**2 / fk_1minusbeta)
    A2 = 0.25 * rho * beta * nu * alpha / ((F*K)**((1.0-beta)/2.0))
    A3 = ((2.0 - 3.0*rho**2)/24.0) * (nu**2)
    correction = 1.0 + (A1 + A2 + A3)*T

    sabr_vol = (alpha/denom)*(z/x_z)*correction
    return max(sabr_vol, 1e-8)

###############################################################################
# 1b) Weighted Objective Function
###############################################################################
def sabr_objective_extended(params, beta, F, T, strikes, market_vols, spot=None):
    """
    Weighted residuals for SABR calibration.
    
    We emphasize near-ATM strikes by weighting their residuals more.
    For each strike K:

      residual_i = w_i * (market_vol_i - model_vol_i)

    Where w_i = 1 / (|K - spot| + 1.0)

    If spot is None, we use unweighted residuals.
    """
    alpha, rho, nu = params

    # Compute model vols
    model_vols = [
        sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T)
        for K in strikes
    ]
    residuals = np.array(market_vols) - np.array(model_vols)

    # If we want to emphasize near-ATM, define a weighting that is larger near S
    if spot is not None:
        dist = np.abs(strikes - spot)
        # Example weighting: 1 / (dist + 1). So if dist=0, weight=1, if dist=10, weight=1/11, etc.
        weights = 1.0 / (dist + 1.0)
    else:
        # No weighting
        weights = np.ones_like(residuals)

    return weights * residuals

###############################################################################
# 2) Black–Scholes Call
###############################################################################
def black_scholes_call_price(S, K, r, T, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

###############################################################################
# 3) Vol Skew & Error Plots
###############################################################################
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

###############################################################################
# 4) Statistics from the PDF
###############################################################################
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
        K_i   = K_vals[i]
        K_i1  = K_vals[i+1]
        f_i   = f_vals[i]
        f_i1  = f_vals[i+1]
        dK    = (K_i1 - K_i)

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
    var = (second_moment / total_prob) - mean**2
    if var < 1e-12:
        return {"mean": mean, "variance": var, "skewness": 0, "kurtosis": 0}

    stdev = np.sqrt(var)
    # approximate approach for skew & kurt:
    skew = (third_moment/total_prob - 3*mean*var - mean**3)/(stdev**3)
    E_K3 = third_moment/total_prob
    E_K4 = fourth_moment/total_prob
    cent4 = E_K4 - 4*mean*E_K3 + 6*(mean**2)*(second_moment/total_prob) - 3*(mean**4)
    kurt = cent4 / (stdev**4)

    return {
        "mean": mean,
        "variance": var,
        "stdev": stdev,
        "skewness": skew,
        "kurtosis": kurt,
        "total_prob": total_prob
    }

###############################################################################
# 5) Main script
###############################################################################
def main():
    # A) Load data & filter
    csv_path = "C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo_cleaned.csv"
    df = pd.read_csv(csv_path)

    # For this example, let's calibrate to calls only
    df = df[df["type"] == "call"].copy()

    # B) Prepare strikes & vols
    strikes = df["strike"].values
    market_vols = df["impliedVolatility"].values

    # C) SABR calibration inputs
    T = 4/52.0
    S = 607      # Spot (or forward) for weighting
    r = 0.00
    beta = 0.25  # Chosen beta; can be changed or tested across multiple values

    # D) Calibrate alpha, rho, nu using Weighted Residuals
    x0 = [0.5, 0.0, 0.5]  # initial guess
    lower_bounds = [0.0, -1.0, 0.0]
    upper_bounds = [np.inf, 1.0, 100.0]

    # Weighted objective: emphasis near S
    res = least_squares(
        sabr_objective_extended,
        x0,
        bounds=(lower_bounds, upper_bounds),
        args=(beta, S, T, strikes, market_vols, S),  # passing S as 'spot'
        method='trf',
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )

    alpha_calib, rho_calib, nu_calib = res.x
    print("===== Extended SABR Calibration (Weighted) =====")
    print(f" Converged: {res.success}, {res.message}")
    print(f"  Beta = {beta}")
    print(f"  Alpha = {alpha_calib:.6f},  Rho = {rho_calib:.6f},  Nu = {nu_calib:.6f}")

    # (1) Compute MSE & RMSE for these parameters
    sabr_vols_market = [
        sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, S, K, T)
        for K in strikes
    ]
    errors = np.array(market_vols) - np.array(sabr_vols_market)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    print(f"\nSABR Fit MSE = {mse:.6f}")
    print(f"SABR Fit RMSE = {rmse:.6f}\n")

    # E) Plot Market vs. SABR Skew & Errors
    plot_sabr_skew_and_errors(strikes, market_vols, sabr_vols_market)

    # F) Dense strike grid for 'continuous' call prices
    K_min = 400
    K_max = 800
    n_points = 500
    K_grid = np.linspace(K_min, K_max, n_points)

    call_prices = []
    for K in K_grid:
        sabr_vol = sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, S, K, T)
        c_price  = black_scholes_call_price(S, K, r, T, sabr_vol)
        call_prices.append(c_price)

    # G) Approx second derivative => PDF
    pdf_grid = np.full(n_points, np.nan)
    for i in range(1, n_points-1):
        K_minus = K_grid[i-1]
        K_i     = K_grid[i]
        K_plus  = K_grid[i+1]

        C_minus = call_prices[i-1]
        C_i     = call_prices[i]
        C_plus  = call_prices[i+1]

        # central difference for non-uniform spacing
        dC_plus  = (C_plus - C_i)/(K_plus - K_i)
        dC_minus = (C_i - C_minus)/(K_i - K_minus)
        second_deriv = 2.0*(dC_plus - dC_minus)/(K_plus - K_minus)

        pdf_grid[i] = np.exp(r*T)*second_deriv

    # H) Plot call price & PDF in one figure
    fig, ax1 = plt.subplots(figsize=(8,6))
    color_call = "gray"
    color_pdf  = "darkred"

    ax1.plot(K_grid, call_prices, color=color_call, label="Call Price")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Call price", color=color_call)
    ax1.tick_params(axis='y', labelcolor=color_call)

    ax2 = ax1.twinx()
    ax2.plot(K_grid, pdf_grid, color=color_pdf, label="f(K)")
    ax2.set_ylabel("f(K)", color=color_pdf)
    ax2.tick_params(axis='y', labelcolor=color_pdf)

    plt.axvline(x=S, color='k', linestyle='--')
    plt.title("Call Price & Implied PDF from SABR")
    fig.tight_layout()
    plt.show()

    # I) Plot PDF alone
    plt.figure(figsize=(8,6))
    plt.plot(K_grid, pdf_grid, color=color_pdf)
    plt.title("Risk-Neutral PDF vs. Strike")
    plt.xlabel("Strike")
    plt.ylabel("f(K)")
    plt.grid(True)
    plt.show()

    # J) PDF statistics
    stats = pdf_statistics(K_grid, pdf_grid)
    print("===== PDF Statistics =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
