import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# =====================
# 1) FETCH DATA
# =====================

csv_path = "C:/Users/User/Desktop/UPF/TGF/Data/SPY_opt_1mo_cleaned.csv"

options_data = pd.read_csv(csv_path)


    #---------------------------------
    #Filter calls and puts separately
    #---------------------------------

calls = options_data[options_data['type'] == 'call']
puts = options_data[options_data['type'] == 'put']

    #--------------
    # Sanity Check
    #--------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
ax0.scatter(calls.strike, calls.midprice)
ax1.scatter(puts.strike, puts.midprice)
plt.show()

# =====================
# 2) BFLYS PROCESSING
# =====================

    #-----------------
    # Bflys Creation
    #-----------------

data = []

#Not sure about this part, as the data has already been filtered :/

for (_, left) ,(_,centre), (_, right) in zip(calls.iterrows(), calls.iloc[1:].iterrows(), calls.iloc[2:].iterrows()):
    # Filter out all zero volume
    if not any(vol > 0 for vol in {left.volume, centre.volume, right.volume}):
        continue
    # Filter out any zero open interest
    if not all(oi > 0 for oi in {left.openInterest, centre.openInterest, right.openInterest}):
        continue
    # Equidistant on either end
    if centre.strike - left.strike != right.strike - centre.strike:
        continue
    butterfly_price = left.midprice - 2* centre.midprice + right.midprice
    max_profit = centre.strike - left.strike
    data.append([centre.strike, butterfly_price, max_profit])


bflys = pd.DataFrame(data, columns=["strike", "price", "max_profit"])
bflys["prob"] = bflys.price / bflys.max_profit

    #-------------------------------------------
    # Bflys Scatter Plot, Strike vs Probability
    #-------------------------------------------

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(9,6))
plt.scatter(bflys.strike, bflys.prob);
plt.xlabel("Strike")
plt.ylabel("Probability")
plt.show()



    #------------------------------------------------------------
    # Bflys Scatter Plot, Strike vs Probability, with smoothing
    #------------------------------------------------------------


from scipy.ndimage import gaussian_filter1d

smoothed_prob = gaussian_filter1d(bflys.prob, 2)

plt.figure(figsize=(9,6))
plt.plot(bflys.strike, bflys.prob, "o", bflys.strike, smoothed_prob, "rx")
plt.legend(["raw prob", "smoothed prob"], loc="best")
plt.xlabel("Strike")
plt.ylabel("Probability")
plt.show()



    #-------------------------------------
    # Bflys PDF smoothed and fitted probs
    #-------------------------------------


plt.figure(figsize=(9,6))
pdf = scipy.interpolate.interp1d(bflys.strike, smoothed_prob, kind="cubic",
                                 fill_value="extrapolate")
x_new = np.linspace(bflys.strike.min(), bflys.strike.max(), 1000)
plt.plot(bflys.strike, smoothed_prob, "rx", x_new, pdf(x_new), "k-");
plt.legend(["smoothed prob", "fitted PDF"], loc="best")
plt.xlabel("K")
plt.ylabel("f(K)")
plt.tight_layout()
plt.show()


    #----------------------------
    # Bflys Integration over PDF
    #----------------------------


raw_total_prob = scipy.integrate.trapezoid(smoothed_prob, bflys.strike)
print(f"Raw total probability: {raw_total_prob}")
normalised_prob = smoothed_prob / raw_total_prob
total_prob = scipy.integrate.trapezoid(normalised_prob, bflys.strike)
print(f"Normalised total probability: {total_prob}")

    #---------------------------------
    # Bflys PDF with normalised probs
    #---------------------------------


plt.figure(figsize=(9,6))
pdf = scipy.interpolate.interp1d(bflys.strike, normalised_prob, kind="cubic",
                                 fill_value="extrapolate")
x_new = np.linspace(bflys.strike.min(), bflys.strike.max(), 1000)
plt.plot(bflys.strike, normalised_prob, "rx", x_new, pdf(x_new), "k-");
plt.legend(["normalised prob", "fitted PDF"], loc="best")
plt.xlabel("K")
plt.ylabel("f(K)")
plt.tight_layout()
plt.show()


