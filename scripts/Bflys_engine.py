# Bflys_engine_test.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

def bfly_main(input_csv="Add Input file path"):
    # 1) Load Data
    options_data = pd.read_csv(input_csv)
    calls = options_data[options_data['type'].str.lower() == 'call']
    puts  = options_data[options_data['type'].str.lower() == 'put']

    # Quick strike vs. midprice scatter
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.scatter(calls.strike, calls.midprice)
    ax0.set_title("Calls: Strike vs. Midprice")
    ax1.scatter(puts.strike, puts.midprice)
    ax1.set_title("Puts: Strike vs. Midprice")
    plt.show()

    # 2) Butterfly spreads
    data = []
    for (_, left), (_, center), (_, right) in zip(
            calls.iterrows(),
            calls.iloc[1:].iterrows(),
            calls.iloc[2:].iterrows()
        ):
        if not any(vol > 0 for vol in [left.volume, center.volume, right.volume]):
            continue
        if not all(oi > 0 for oi in [left.openInterest, center.openInterest, right.openInterest]):
            continue
        if center.strike - left.strike != right.strike - center.strike:
            continue

        butterfly_price = left.midprice - 2 * center.midprice + right.midprice
        max_profit = center.strike - left.strike
        prob = butterfly_price / max_profit
        data.append([center.strike, butterfly_price, max_profit, prob])

    bflys = pd.DataFrame(data, columns=["strike", "price", "max_profit", "prob"])

    # Ignore negative probabilities
    bflys = bflys[bflys['prob'] >= 0]

    plt.figure(figsize=(9, 6))
    plt.scatter(bflys.strike, bflys.prob)
    plt.xlabel("Strike")
    plt.ylabel("Probability")
    plt.show()

    # 3) Smoothed probability
    smoothed_prob = gaussian_filter1d(bflys.prob, sigma=2)
    plt.figure(figsize=(9, 6))
    plt.plot(bflys.strike, bflys.prob, "o", label="raw prob")
    plt.plot(bflys.strike, smoothed_prob, "rx", label="smoothed prob")
    plt.xlabel("Strike")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    # 4) Fitted curve
    pdf = interp1d(bflys.strike, smoothed_prob, kind="cubic", fill_value="extrapolate")
    x_new = np.linspace(bflys.strike.min(), bflys.strike.max(), 1000)
    plt.figure(figsize=(9, 6))
    plt.plot(bflys.strike, smoothed_prob, "rx", x_new, pdf(x_new), "k-")
    plt.xlabel("K")
    plt.ylabel("f(K)")
    plt.legend(["smoothed prob", "fitted PDF"], loc="best")
    plt.tight_layout()
    plt.show()

    # 5) Integration
    raw_total_prob = trapezoid(smoothed_prob, bflys.strike)
    print(f"Raw total probability: {raw_total_prob}")
    normalised_prob = smoothed_prob / raw_total_prob if raw_total_prob != 0 else smoothed_prob
    total_prob = trapezoid(normalised_prob, bflys.strike)
    print(f"Normalised total probability: {total_prob}")

    # 6) Normalised PDF
    pdf_norm = interp1d(bflys.strike, normalised_prob, kind="cubic", fill_value="extrapolate")
    plt.figure(figsize=(9, 6))
    plt.plot(bflys.strike, normalised_prob, "rx", x_new, pdf_norm(x_new), "k-")
    plt.xlabel("K")
    plt.ylabel("f(K)")
    plt.legend(["normalised prob", "fitted PDF"], loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    bfly_main()


