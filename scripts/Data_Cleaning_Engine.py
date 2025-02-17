import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import re
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

def black76_call_price(F, K, r, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(F - K, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discountedK = K * np.exp(-r * T)
    return F * norm.cdf(d1) - discountedK * norm.cdf(d2)

def black76_put_price(F, K, r, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - F, 0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discountedK = K * np.exp(-r * T)
    return discountedK * norm.cdf(-d2) - F * norm.cdf(-d1)

def black76_implied_vol(market_price, F, K, r, T, is_call=True):
    if market_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return float('nan')

    def objective(sigma):
        price_model = (
            black76_call_price(F, K, r, T, sigma)
            if is_call else
            black76_put_price(F, K, r, T, sigma)
        )
        return price_model - market_price

    try:
        iv = brentq(objective, 1e-6, 5)  # bracket for sigma between near-zero and 5
        return iv
    except:
        return float('nan')

def clean_options_data(input_file, output_file, F, T):
    """
    Steps:
    1) Load CSV and rename columns to lowercase
    2) Parse strike column (remove trailing 'C'/'P')
    3) Compute impliedVolatility via Black-76 using user-passed F and T
    4) Filter out near-zero IV and negative bid/ask
    5) Fill volume NaN with 0 (so we keep zero-volume options)
    6) Drop any truly negative volumes if that ever occurs (optional)
    7) Drop unnecessary columns, add midprice, etc.
    """

    # 1. Load the data
    try:
        options_data = pd.read_csv(input_file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"[ERROR] The file '{input_file}' was not found.")
        return

    # 2. Rename columns if present
    rename_map = {
        'Strike': 'strike_raw',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Last': 'last',
        'Change': 'change',
        'Bid': 'bid',
        'Ask': 'ask',
        'Volume': 'volume',
        'Open Int': 'openInterest',
        'Premium': 'premium',
        'Time': 'time',
        'Type': 'option_type'
    }
    for old_name, new_name in list(rename_map.items()):
        if old_name not in options_data.columns:
            rename_map.pop(old_name)

    options_data.rename(columns=rename_map, inplace=True)

    # 3. Parse the strike column to remove trailing 'C'/'P'
    def parse_strike(str_str):
        if isinstance(str_str, str):
            match = re.match(r'^\s*(\d+(\.\d+)?)([CPcp])?\s*$', str_str.strip())
            if match:
                return float(match.group(1))  # numeric part
        return float('nan')

    if 'strike_raw' in options_data.columns:
        options_data['strike'] = options_data['strike_raw'].apply(parse_strike)

    # 4. Standardize 'type' column
    if 'option_type' in options_data.columns:
        options_data['type'] = options_data['option_type'].str.lower()
    else:
        # default if missing
        options_data['type'] = "call"

    
    r = 0.00
    implied_vols = []
    for _, row in options_data.iterrows():
        K = row.get('strike', np.nan)
        last_price = row.get('last', np.nan)
        is_call = (row.get('type', 'call') == 'call')
        iv_est = black76_implied_vol(last_price, F, K, r, T, is_call)
        implied_vols.append(iv_est)

    options_data['impliedVolatility'] = implied_vols

    # 6. Remove near-zero IV
    iv_threshold = 1e-4
    size_before = len(options_data)
    options_data = options_data[options_data['impliedVolatility'] > iv_threshold]
    size_after = len(options_data)
    print(f"Dropped {size_before - size_after} rows with impliedVolatility <= {iv_threshold}.")

    # 7. Drop rows with missing or negative bid/ask (allows zero)
    size_before = len(options_data)
    options_data = options_data.dropna(subset=['bid', 'ask'])
    options_data = options_data[(options_data['bid'] >= 0) & (options_data['ask'] >= 0)]
    size_after = len(options_data)
    print(f"Dropped {size_before - size_after} rows due to negative bid/ask or NaNs.")

    # 8. Handle volume column: fill NaN with 0
    if 'volume' in options_data.columns:
        # Fill missing volume with 0
        options_data['volume'] = options_data['volume'].fillna(0)
        # Optional: remove rows with negative volume if that ever appears
        size_before = len(options_data)
        options_data = options_data[options_data['volume'] >= 0]
        size_after = len(options_data)
        print(f"Dropped {size_before - size_after} rows due to negative volume (if any).")

    # 9. Drop unnecessary columns
    columns_to_remove = [
        'inTheMoney', 'contractSize', 'currency',
        'strike_raw', 'option_type'
    ]
    options_data = options_data.drop(columns=columns_to_remove, errors='ignore')

    # 10. Add midprice & smoothed_midprice
    options_data['midprice'] = (
        (options_data['bid'].fillna(0) + options_data['ask'].fillna(0)) / 2
    )
    options_data['smoothed_midprice'] = gaussian_filter1d(
        options_data['midprice'].fillna(0), sigma=3
    )

    # 11. Plot calls vs. puts IV (optional)
    if all(col in options_data.columns for col in ['impliedVolatility', 'strike', 'type']):
        calls_data = options_data[options_data['type'] == 'call']
        puts_data = options_data[options_data['type'] == 'put']

        plt.figure(figsize=(10, 5))
        plt.scatter(
            calls_data['strike'], calls_data['impliedVolatility'],
            s=10, alpha=0.5, color='blue', label='Calls'
        )
        plt.scatter(
            puts_data['strike'], puts_data['impliedVolatility'],
            s=10, alpha=0.5, color='red', label='Puts'
        )
        plt.title("Implied Volatility vs. Strike")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.legend()
        plt.show()

    # 12. Save cleaned data
    try:
        options_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the cleaned data: {e}")

    # 13. Missing values summary
    missing_values = options_data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # 14. Print extra info
    if 'strike' in options_data.columns:
        print("Min strike:", options_data['strike'].min())
        print("Max strike:", options_data['strike'].max())

if __name__ == "__main__":
    test_input = "Add some file"
    test_output = test_input.replace(".csv", "_cleaned.csv")
    F_test = 72.85 # example
    T_test = 30/365
    clean_options_data(test_input, test_output, F_test, T_test)

