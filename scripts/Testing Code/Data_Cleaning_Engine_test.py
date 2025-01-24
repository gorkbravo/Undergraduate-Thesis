import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib.pyplot as plt

INPUT_FILE = "C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo.csv"
OUTPUT_FILE = "C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo_cleaned.csv"

def clean_options_data(input_file, output_file):
    """
    Cleans the options data, including:
    1) Plotting impliedVolatility vs. strike,
    2) Dropping outliers in impliedVolatility,
    3) Saving the cleaned data.
    """

    # 1. Load the data
    try:
        options_data = pd.read_csv(input_file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return

    # 2. Visualize impliedVolatility vs. strike and drop outliers
    if 'impliedVolatility' in options_data.columns and 'strike' in options_data.columns:
        # --- (a) Scatter Plot (Before Outlier Removal) ---
        plt.figure(figsize=(8, 4))
        plt.scatter(options_data['strike'], options_data['impliedVolatility'], s=10, alpha=0.5)
        plt.title("Implied Volatility vs. Strike (Before Outlier Removal)")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.show()

        # --- (b) Drop Outliers Using IQR (Interquartile Range) ---
        Q1 = options_data['impliedVolatility'].quantile(0.25)
        Q3 = options_data['impliedVolatility'].quantile(0.75)
        IQR = Q3 - Q1
        # Define bounds (you can adjust the multiplier below)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out rows that are outside of this range
        original_size = len(options_data)
        options_data = options_data[
            (options_data['impliedVolatility'] >= lower_bound) &
            (options_data['impliedVolatility'] <= upper_bound)
        ]
        filtered_size = len(options_data)
        print(f"Dropped {original_size - filtered_size} rows based on impliedVolatility outliers.")

        # --- (c) Scatter Plot (After Outlier Removal) ---
        plt.figure(figsize=(8, 4))
        plt.scatter(options_data['strike'], options_data['impliedVolatility'], s=10, alpha=0.5, color='green')
        plt.title("Implied Volatility vs. Strike (After Outlier Removal)")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.show()

    # 3. Drop unnecessary columns
    columns_to_remove = ['inTheMoney', 'contractSize', 'currency']
    options_data = options_data.drop(columns=columns_to_remove, errors='ignore')

    # 4. Extract expiration date from contractSymbol (if present)
    if 'contractSymbol' in options_data.columns:
        options_data['expirationDate'] = options_data['contractSymbol'].str.extract(r'(\d{6})')[0]
        options_data['expirationDate'] = pd.to_datetime(options_data['expirationDate'], format='%y%m%d', errors='coerce')

    # 5. Remove rows with zero or NaN bid/ask
    options_data = options_data.dropna(subset=['bid', 'ask'])
    options_data = options_data[(options_data['bid'] > 0) & (options_data['ask'] > 0)]

    # 6. Calculate mid-prices
    options_data['midprice'] = (options_data['bid'] + options_data['ask']) / 2

    # 7. Remove options with excessive bid-ask spread
    options_data['spread'] = options_data['ask'] - options_data['bid']
    # Example threshold: 10% of midprice
    options_data = options_data[options_data['spread'] / options_data['midprice'] < 0.1]

    # 8. Filter by volume (if volume column exists)
    if 'volume' in options_data.columns:
        options_data = options_data[options_data['volume'] > 0]

    # 9. Smooth midprice (if you want to keep this step)
    options_data['smoothed_midprice'] = gaussian_filter1d(options_data['midprice'], sigma=3)

    # 10. Save cleaned data to new CSV
    try:
        options_data.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"Error saving the cleaned data: {e}")

    # 11. Check missing values
    missing_values = options_data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # 12. Describe relevant columns
    if 'expirationDate' in options_data.columns:
        print(options_data['expirationDate'].describe())

    if 'spread' in options_data.columns:
        print(options_data['spread'].describe())
        # Plot spread distribution
        options_data['spread'].plot.hist(bins=50, title="Spread Distribution")
        plt.show()

    # 13. (Optional) Find outlier strikes based on a certain threshold (for example, ±100 from spot price)
    spot_price = 607  # Example SPY spot price
    if 'strike' in options_data.columns:
        outliers = options_data[abs(options_data['strike'] - spot_price) > 100]
        print(f"Outlier strikes: {len(outliers)}")
        # Plot Strike vs Midprice
        options_data.plot.scatter(x='strike', y='midprice', title="Strike vs Midprice")
        plt.show()

if __name__ == "__main__":
    clean_options_data(INPUT_FILE, OUTPUT_FILE)
