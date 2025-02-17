# SABR_betas_comp_engine_test.py

import numpy as np
import pandas as pd
import math
from scipy.optimize import least_squares


# 1) Extended Hagan SABR

def sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T):
    if np.isclose(F, K, atol=1e-14):
        leading = alpha / (F**(1.0 - beta))
        A1 = ((1-beta)**2 / 24.0) * (alpha**2 / (F**(2*(1-beta))))
        A2 = 0.25 * rho * beta * nu * alpha / (F**(1.0-beta))
        A3 = ((2.0 - 3.0*rho**2)/24.0) * (nu**2)
        correction = 1.0 + (A1 + A2 + A3)*T
        return max(leading*correction, 1e-8)

    z = (nu/alpha)*(F**(1.0 - beta) - K**(1.0 - beta))
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


# 2) SABR Objective 

def sabr_objective_extended(params, beta, F, T, strikes, market_vols):
    alpha, rho, nu = params
    model_vols = [
        sabr_hagan_black_vol_extended(alpha, beta, rho, nu, F, K, T)
        for K in strikes
    ]
    return np.array(market_vols) - np.array(model_vols)


# 3) Main Loop Over Beta

def main(csv_path, F, T):
    beta_values = [0.05, 0.10, 0.15 ,0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]

    # Bounds & initial guess
    lower_bounds = [0.0, -1.0, 0.0]
    upper_bounds = [np.inf, 1.0, 1000.0]
    x0 = [0.5, 0.0, 0.5]

    # Load Data
    df = pd.read_csv(csv_path)
    df = df[df["type"] == "call"].copy()  # Only calls
    strikes = df["strike"].values
    market_vols = df["impliedVolatility"].values

    # Store results
    results = []

    # Loop over Beta values
    for beta in beta_values:
        fun = lambda p: sabr_objective_extended(
            p, beta, F, T, strikes, market_vols
        )

        # Perform calibration
        res = least_squares(
            fun,
            x0,
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12
        )

        alpha_calib, rho_calib, nu_calib = res.x

        # Compute MSE & RMSE
        sabr_vols = [
            sabr_hagan_black_vol_extended(alpha_calib, beta, rho_calib, nu_calib, F, K, T)
            for K in strikes
        ]
        errors = np.array(market_vols) - np.array(sabr_vols)
        mse = np.mean(errors**2)
        rmse = math.sqrt(mse)

        # Save results
        results.append({
            "Beta": beta,
            "Alpha": alpha_calib,
            "Rho": rho_calib,
            "Nu": nu_calib,
            "MSE": mse,
            "RMSE": rmse,
            "Converged": res.success
        })

    # Print Summary
    print("============= SABR Multi-Beta Comparison (Unweighted) =============")
    print(f"{'Beta':>5}  {'Alpha':>9}  {'Rho':>9}  {'Nu':>9}  {'MSE':>12}  {'RMSE':>12}  Converged")
    for r in results:
        print(f"{r['Beta']:5.2f}  {r['Alpha']:9.4f}  {r['Rho']:9.4f}  {r['Nu']:9.4f}"
              f"  {r['MSE']:12.6e}  {r['RMSE']:12.6e}  {r['Converged']}")

if __name__ == "__main__":
    # Example for standalone testing
    main(
        csv_path="C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo_cleaned.csv",
        F=607,
        T=4.0 / 52.0
    )

