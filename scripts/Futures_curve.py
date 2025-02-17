# Futures_curve.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import numpy as np
import re
import os
from datetime import datetime, timedelta

# ======================= CONFIGURATION ========================
HISTORY_CSV_PATH = r"C:/Users/User/Desktop/UPF/TGF/Data/Stats/stats.csv"
COLOR_CONTANGO = '#2ecc71'  # Green
COLOR_BACKWARDATION = '#e74c3c'  # Red
COLOR_NEUTRAL = '#f1c40f'
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
# =============================================================

plt.style.use(PLOT_STYLE)

MONTH_CODES = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

def save_index_to_csv(input_path, index_value, components=None):
    """
    Save index value with metadata to history file.
    Now also accepts a dictionary 'components' for storing extra fields
    (like 1Y spread, 3Y spread, near_term, long_term, etc.).
    """
    # Build the row dict
    row_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_file': os.path.basename(input_path),
        'term_structure_index': round(index_value, 4),
        'calculation_date': datetime.now().strftime('%Y-%m-%d'),
        'market_state': 'Contango' if index_value >= 0 else 'Backwardation'
    }
    
    # Merge in additional components (if provided)
    if components:
        # e.g. {'year1_spread': 0.0213, 'year3_spread': 0.045, ...}
        for k, v in components.items():
            row_dict[k] = round(v, 4) if isinstance(v, (int, float)) else v

    new_entry = pd.DataFrame([row_dict])

    # Append or create the HISTORY_CSV_PATH
    if os.path.exists(HISTORY_CSV_PATH):
        new_entry.to_csv(HISTORY_CSV_PATH, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(HISTORY_CSV_PATH, index=False)

    print(f"Index {index_value:.2f} and components saved to {HISTORY_CSV_PATH}")

def parse_contract(contract_str):
    """Robust futures contract parser with error handling"""
    try:
        clean_contract = re.split(r'[ .()]', contract_str)[0]
        if len(clean_contract) < 5 or not clean_contract.startswith('CL'):
            return pd.NaT

        month_code = clean_contract[2]
        if month_code not in MONTH_CODES:
            return pd.NaT

        # e.g. "CLH25" -> year=2025 if 25 < 50 else 1925
        year_digits = int(clean_contract[3:5])
        year = 2000 + year_digits if year_digits < 50 else 1900 + year_digits

        return pd.Timestamp(year=year, month=MONTH_CODES[month_code], day=1)
    except Exception as e:
        print(f"Parse error for {contract_str}: {str(e)}")
        return pd.NaT

def validate_inputs(df):
    """Comprehensive data validation checks"""
    if df.empty:
        raise ValueError("Empty dataframe after filtering")

    if not df['Expiration'].is_monotonic_increasing:
        raise ValueError("Contract dates not chronological")

    price_changes = df['Last'].pct_change().dropna().abs()
    if (price_changes > 0.25).any():
        raise ValueError(f"Implausible price jump: {price_changes.idxmax()}")

    if (df['Last'] < 5).any():
        raise ValueError("Prices below $5 detected")

def calculate_term_structure_index(df):
    """
    Calculate composite index using 1-month, 3-month, and 1-year spreads.
    Returns:
      composite_index, components_dict
    """
    if len(df) < 2:
        return 0.0, {}

    front_price = df['Last'].iloc[0]
    front_date = df['Expiration'].iloc[0]

    # Identify time anchors for 1-month, 3-month, and 1-year contracts
    month1_idx = next(
        (i for i, row in df.iterrows() if (row['Expiration'] - front_date).days >= 30),
        len(df)-1
    )
    month3_idx = next(
        (i for i, row in df.iterrows() if (row['Expiration'] - front_date).days >= 90),
        len(df)-1
    )
    year1_idx = next(
        (i for i, row in df.iterrows() if (row['Expiration'] - front_date).days >= 365),
        len(df)-1
    )

    # Get prices for each anchor
    month1_price = df['Last'].iloc[month1_idx]
    month3_price = df['Last'].iloc[month3_idx]
    year1_price  = df['Last'].iloc[year1_idx]

    # Calculate raw spreads
    month1_spread = (month1_price - front_price) / front_price
    month3_spread = (month3_price - front_price) / front_price
    year1_spread  = (year1_price  - front_price) / front_price

    # Normalize spreads based on assumed typical levels (adjust these as needed)
    month1_severity = month1_spread / 0.01  # Typical 1-month spread ~1%
    month3_severity = month3_spread / 0.03  # Typical 3-month spread ~3%
    year1_severity  = year1_spread  / 0.05  # Typical 1-year spread ~5%

    # Apply nonlinear normalization
    norm_month1 = np.tanh(month1_severity)
    norm_month3 = np.tanh(month3_severity)
    norm_year1  = np.tanh(year1_severity)

    # Directional persistence (as before)
    consecutive_changes = np.diff(df['Last']) < 0
    persistence = np.mean(consecutive_changes)
    persistence_impact = (2 * persistence - 1) * min(abs(year1_spread / 0.05), 1)

    # Diagnostic output (optional)
    print(f"\nComposite index components:")
    print(f"1M Spread: {month1_spread:.4f} -> Severity: {month1_severity:.4f} -> Norm: {norm_month1:.4f}")
    print(f"3M Spread: {month3_spread:.4f} -> Severity: {month3_severity:.4f} -> Norm: {norm_month3:.4f}")
    print(f"1Y Spread: {year1_spread:.4f} -> Severity: {year1_severity:.4f} -> Norm: {norm_year1:.4f}")
    print(f"Persistence: {persistence:.4f} -> Impact: {persistence_impact:.4f}")

    # Combine components using chosen weights (tweak these weights as needed)
    composite = (
        0.25 * norm_month1 +
        0.35 * norm_month3 +
        0.25 * norm_year1 +
        0.15 * persistence_impact
    )
    composite = np.clip(composite, -1, 1)

    # Build a dictionary with all components
    components_dict = {
        'month1_spread': month1_spread,
        'month3_spread': month3_spread,
        'year1_spread': year1_spread,
        'norm_month1': norm_month1,
        'norm_month3': norm_month3,
        'norm_year1': norm_year1,
        'persistence': persistence,
        'persistence_impact': persistence_impact
    }

    return composite, components_dict


def create_visualization(df, index_value):
    """Create professional-grade visualization of the term structure"""
    plt.rcParams['font.family'] = 'Segoe UI'
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('#f5f6fa')

    # Main price curve
    line_color = COLOR_CONTANGO if index_value >= 0 else COLOR_BACKWARDATION
    ax.plot(df['Expiration'], df['Last'],
            marker='o', markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=line_color,
            color=line_color, linewidth=2.5, alpha=0.9,
            path_effects=[pe.withStroke(linewidth=4, foreground='#ffffff55')])

    # Spread bars
    if 'Annualized_Spread' in df:
        ax2 = ax.twinx()
        bar_color = np.where(df['Annualized_Spread'] >= 0, COLOR_CONTANGO, COLOR_BACKWARDATION)
        bars = ax2.bar(df['Expiration'], df['Annualized_Spread'],
                       width=20, color=bar_color, alpha=0.15)
        ax2.axhline(0, color='#7f8c8d', linewidth=1, linestyle='--')
        ax2.set_ylabel('Annualized Spread (%)', fontsize=10, labelpad=15)
        ax2.spines['right'].set_position(('outward', 60))

    # Index gauge
    gauge_color = COLOR_CONTANGO if index_value >= 0 else COLOR_BACKWARDATION
    gauge_text = f"Term Structure Index\n{index_value:.2f}"
    ax.annotate(gauge_text, xy=(0.03, 0.88), xycoords='axes fraction',
                fontsize=14, color=gauge_color, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='#ffffff',
                          edgecolor=gauge_color, pad=0.5))

    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Price axis styling
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.tick_params(axis='both', colors='#7f8c8d', labelsize=10)

    # Dynamic Y-axis scaling
    y_min, y_max = df['Last'].min(), df['Last'].max()
    ax.set_ylim(y_min - (y_max - y_min)*0.05, y_max + (y_max - y_min)*0.1)
    ax.set_ylabel('Futures Price (USD)', fontsize=12, color=line_color, labelpad=15)

    # Title annotation
    ax.set_title(f"WTI Crude Oil Term Structure - {datetime.now().strftime('%d %b %Y')}",
                 fontsize=16, pad=20, color='#2c3e50', fontweight='semibold')

    plt.tight_layout()
    plt.show()

def main(csv_path):
    """Main processing pipeline"""
    try:
        # 1) Load and pre-process
        df = pd.read_csv(csv_path)
        df = df[df['Contract'].str.contains(r'^CL[A-Z]\d{2}', regex=True, na=False)]
        df['Expiration'] = df['Contract'].apply(parse_contract)
        df = df.dropna(subset=['Expiration']).sort_values('Expiration')

        # 2) Validate inputs
        validate_inputs(df)

        # 3) Calculate extra columns
        df['Days_to_Next'] = (df['Expiration'].shift(-1) - df['Expiration']).dt.days
        df['Annualized_Spread'] = (df['Last'].shift(-1) - df['Last']) / df['Last'] * (365/df['Days_to_Next']) * 100
        df = df.iloc[:-1]  # Remove last contract

        # 4) Calculate the term structure index & detailed components
        ts_index, components = calculate_term_structure_index(df)
        print(f"\nCalculated Index: {ts_index:.2f}")

        # 5) Save index and component details to CSV
        save_index_to_csv(csv_path, ts_index, components=components)

        # 6) Visualization
        create_visualization(df, ts_index)

    except Exception as e:
        print(f"Processing failed: {str(e)}")


# Standalone test
if __name__ == "__main__":
    test_csv = r"C:/Users/User/Desktop/UPF/TGF/Data/Futures/crude-oil-wti-prices-intraday-01-31-2025 (0).csv"
    main(test_csv)
