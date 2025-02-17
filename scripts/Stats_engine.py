import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ---------------------------
# 1. Data Loading and Preparation
# ---------------------------
# Load the CSV files. (No need to parse dates since merging is by order.)
stats_df = pd.read_csv('C:/Users/User/Desktop/UPF/TGF/Data/Stats/stats.csv')
stats2_df = pd.read_csv('C:/Users/User/Desktop/UPF/TGF/Data/Stats/stats2.csv')

# Reset the indices to ensure proper alignment.
stats_df = stats_df.reset_index(drop=True)
stats2_df = stats2_df.reset_index(drop=True)

# Create an 'order' column in each dataframe to serve as a merge key.
stats_df['order'] = stats_df.index
stats2_df['order'] = stats2_df.index

# Merge the two dataframes on the 'order' column.
merged_df = pd.merge(stats_df, stats2_df, on='order', how='inner')

# ---------------------------
# 2. Recalculate and Compare the Composite Index
# ---------------------------
# Our new composite index is defined as:
#     composite = 0.25 * norm_month1 + 0.35 * norm_month3 + 0.25 * norm_year1 + 0.15 * persistence_impact
# We check if the new component columns exist before recalculating.
required_columns = ['norm_month1', 'norm_month3', 'norm_year1', 'persistence_impact']
if all(col in merged_df.columns for col in required_columns):
    merged_df['composite_calculated'] = np.clip(
        0.25 * merged_df['norm_month1'] +
        0.35 * merged_df['norm_month3'] +
        0.25 * merged_df['norm_year1'] +
        0.15 * merged_df['persistence_impact'],
        -1, 1
    )
    index_version = "NEW"
else:
    print("Warning: New component columns not found. Falling back to the legacy 'term_structure_index'.")
    merged_df['composite_calculated'] = merged_df['term_structure_index']
    index_version = "LEGACY"

print(f"\nComposite index ({index_version} version) sample comparison:")
print(merged_df[['order', 'term_structure_index', 'composite_calculated']].head())

# ---------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------
# Select PDF moments (assumed to be output by the implied PDF engine).
pdf_moments = ['mean', 'variance', 'skewness', 'kurtosis']

# Display summary statistics.
print("\nSummary Statistics:")
print(merged_df.describe())

# Compute and print the correlation matrix between the composite index and PDF moments.
corr_columns = ['composite_calculated'] + pdf_moments
corr_matrix = merged_df[corr_columns].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the relationships with a pairplot.
sns.pairplot(merged_df[corr_columns])
plt.suptitle("Pairplot: Composite Index vs. PDF Moments", y=1.02)
plt.show()

# Example scatter plot: Composite index vs. PDF skewness.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='composite_calculated', y='skewness', data=merged_df)
plt.title('Composite Index vs. PDF Skewness')
plt.xlabel('Composite Index (Calculated)')
plt.ylabel('PDF Skewness')
plt.show()

# ---------------------------
# 4. Preliminary Regression Analysis
# ---------------------------
# Here we illustrate a basic regression: predicting PDF skewness from the composite index.
X = merged_df[['composite_calculated']]
y = merged_df['skewness']

# Add a constant (intercept term).
X = sm.add_constant(X)

# Fit an Ordinary Least Squares (OLS) regression model.
model = sm.OLS(y, X).fit()

print("\nRegression Results: Predicting PDF Skewness from Composite Index")
print(model.summary())

# ---------------------------
# 5. Expanded Analysis: Regression and Visualization for Each PDF Moment
# ---------------------------
for moment in pdf_moments:
    # Scatter plot for each PDF moment against the composite index.
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='composite_calculated', y=moment, data=merged_df)
    plt.title(f'Composite Index vs. PDF {moment.capitalize()}')
    plt.xlabel('Composite Index (Calculated)')
    plt.ylabel(f'PDF {moment.capitalize()}')
    plt.show()
    
    # Regression analysis for each PDF moment.
    X = merged_df[['composite_calculated']]
    y = merged_df[moment]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"\nRegression Results: Predicting PDF {moment.capitalize()} from Composite Index")
    print(model.summary())

# ---------------------------
# 6. Next Steps for Further Analysis
# ---------------------------
# Consider expanding the regression models to include individual futures curve components such as:
#    - norm_month1, norm_month3, norm_year1
#    - persistence, persistence_impact
# Also, you might explore non-linear relationships, interactions, or even time series dynamics if your data evolves over time.
