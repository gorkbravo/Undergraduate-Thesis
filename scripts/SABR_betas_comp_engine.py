import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import math



################################################################################
# SABR Extended Hagan Formula
################################################################################
def sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T):
    """
    Extended Hagan SABR implied volatility approximation, including second-order
    correction terms (A1, A2, A3). 
    """

    if np.isclose(F, K, atol=1e-14):
        # Near ATM
        leading = alpha / (F**(1.0 - beta))
        A1 = ((1-beta)**2 / 24.0) * (alpha**2 / (F**(2*(1-beta))))
        A2 = 0.25 * rho * beta * nu * alpha / (F**(1-beta))
        A3 = ((2.0 - 3.0*(rho**2))/24.0) * (nu**2)
        correction = 1.0 + (A1 + A2 + A3)*T
        return max(leading*correction, 1e-8)

    z = (nu / alpha) * ((F**(1.0 - beta)) - (K**(1.0 - beta)))
    eps = 1e-14
    numerator   = np.sqrt(1.0 - 2*rho*z + z*z) + z - rho
    denominator = 1.0 - rho
    x_z = np.log((numerator + eps)/(denominator + eps))

    denom = (F*K)**((1.0 - beta)/2.0)

    # Second-order expansions
    fk_1minusbeta = (F*K)**(1 - beta)
    A1 = ((1-beta)**2 / 24.0) * (alpha**2 / fk_1minusbeta)
    A2 = 0.25 * rho * beta * nu * alpha / ((F*K)**((1.0-beta)/2.0))
    A3 = ((2.0 - 3.0*rho**2)/24.0) * (nu**2)

    correction = 1.0 + (A1 + A2 + A3)*T

    sabr_vol = (alpha / denom) * (z / x_z) * correction
    return max(sabr_vol, 1e-8)

def sabr_objective_extended(x, beta, F, T, strikes, market_vols):
    """
    Objective function returning (market_vol - sabr_vol) for each strike.
    x = [alpha, rho, nu]
    """
    alpha, rho, nu = x
    sabr_vols = [
        sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T)
        for K in strikes
    ]
    return np.array(market_vols) - np.array(sabr_vols)


################################################################################
# Main: Test multiple Beta values
################################################################################
def main():
    # 1) Load your data
    csv_path = "C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo.csv"
    df = pd.read_csv(csv_path)
    
    # Optionally filter for calls, etc.
    df = df[df["type"] == "call"].copy()

    # 2) Extract needed columns
    strikes = df["strike"].values
    market_vols = df["impliedVolatility"].values

    # 3) Basic inputs
    T = 4 / 52.0   # about 1 month
    F = 52     # Spot or forward
    # If you have a different approach for F (like S*exp(r*T)), adapt here

    # 4) Range of Beta values to test
    beta_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]

    # 5) Bounds for [alpha, rho, nu]
    #    We'll allow alpha >=0, -1 <= rho <=1, 0 <= nu <= 10
    lower_bounds = [0.0, -1.0, 0.0]
    upper_bounds = [np.inf, 1.0, 1000.0]

    # 6) Initialize array/list to store results
    results = []

    # 7) Loop over each Beta
    for beta in beta_values:

        # Initial guess for alpha, rho, nu
        x0 = [0.5, 0.0, 0.5]

        # Run least_squares calibration
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

        # Compute SABR-fitted vols
        sabr_vols = [
            sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, F, K, T)
            for K in strikes
        ]

        # Compute MSE / RMSE
        errors = np.array(market_vols) - np.array(sabr_vols)
        mse = np.mean(errors**2)
        rmse = math.sqrt(mse)

        # Collect results
        results.append({
            "Beta": beta,
            "Alpha": alpha_calib,
            "Rho": rho_calib,
            "Nu": nu_calib,
            "MSE": mse,
            "RMSE": rmse,
            "Converged": res.success
        })

    # 8) Print a summary table
    print("                 ===== SABR Multi-Beta Comparison =====")
    print("{:>6} | {:>9} | {:>9} | {:>9} | {:>8} | {:>8} | {}".format(
        "Beta", "Alpha", "Rho", "Nu", "MSE", "RMSE", "Conv?"
    ))
    print("-"*70)
    for r in results:
        print("{:6.2f} | {:9.4f} | {:9.4f} | {:9.4f} | {:8.6f} | {:8.6f} | {}".format(
            r["Beta"],
            r["Alpha"],
            r["Rho"],
            r["Nu"],
            r["MSE"],
            r["RMSE"],
            "Yes" if r["Converged"] else "No"
        ))

if __name__ == "__main__":
    main()

