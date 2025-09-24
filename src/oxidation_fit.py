import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

R = 8.314  # Gas constant (J/mol·K)
T_25 = 298.15  # Temperature 25°C (K)

# [Step 1]: Data preprocessing
def mad(series):
    """Calculate the Median Absolute Deviation (MAD) of a series."""
    medium = np.median(series)
    return np.median(np.abs(series - medium))

def robust_z(series):
    """Calculate the robust z-score of a series."""
    medium = np.median(series)
    m = mad(series)
    if m == 0:
        return np.zeros_like(series, dtype=float)
    return 0.6745 * (series - medium) / m 

def remove_outliers(group, z_thresh=3.5, min_n=6):
    """Remove outliers from a group based on robust z-scores."""
    df = group.copy()
    if len(df) < min_n:
        return df
    ln_ip = np.log(df["ip_h"].astype(float).values)
    z = robust_z(ln_ip)
    mask = np.abs(z) <= z_thresh
    return df.loc[mask]

def clean_oxidation_data(oxidation_df, z_thresh=3.5, min_n=6):
    """Clean oxidation data by removing outliers."""
    oxidation_clean = (
        oxidation_df
        .groupby(["category", "set_point_t"], group_keys=False)
        .apply(lambda g: remove_outliers(g, z_thresh=z_thresh, min_n=min_n))
        .reset_index(drop=True)
    )
    return oxidation_clean

# Read data and clean
input_excel = 'data/experimental_data.xlsx'
oxidation_df = pd.read_excel(input_excel, sheet_name='oxidation')
cleaned = clean_oxidation_data(oxidation_df)

# [Step 2]: Fit Arrhenius parameters by category
def arrhenius_linear_model(x, a, b):
    """Arrhenius model in linear form: ln(1/IP) = a + b * (1/T)"""
    return a + b * x

def fit_arrhenius_model(temps_C, ips):
    """Fit the Arrhenius model to temperature and IP data."""
    temps_K = temps_C + 273.15
    x = 1 / temps_K
    y = np.log(1 / ips)
    try:
        params, _ = curve_fit(arrhenius_linear_model, x, y)
        a, b = params
        Ea = -b * R / 1000       # kJ/mol
        k_0 = np.exp(a)          # 1/h
        ln_1_IP_25 = arrhenius_linear_model(1 / T_25, a, b) # ln(1/IP) at 25°C
        IP_25 = np.exp(-ln_1_IP_25) # IP at 25°C (hours)

        # Predictions for fit quality
        y_pred = arrhenius_linear_model(x, a, b)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        rss = np.sum((y - y_pred) ** 2)

        return [Ea, k_0, IP_25, r2, rmse, mae, rss, x, y, y_pred]
    except Exception:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, None, None]

# Loop by category
os.makedirs("outputs/oxidation", exist_ok=True)
ox_results = []

for category, group in cleaned.groupby('category'):
    grouped = group.groupby('set_point_t')['ip_h'].mean().reset_index() # Average IP per temperature
    temps_C = grouped['set_point_t'].astype(float).values
    ips = grouped['ip_h'].astype(float).values
    
    E_a, k_0, IP_25, r2, rmse, mae, rss, x, y, y_pred = fit_arrhenius_model(temps_C, ips)
    IP_25_days = IP_25 / 24 if not np.isnan(IP_25) else np.nan  # Convert hours to days

    # Print summary for each category
    print(f"\nCategory: {category}")
    print(f"  E_a = {E_a:.4f} kJ/mol, k_0 = {k_0:.4e} 1/h, IP_25C = {IP_25_days:.2f} days")
    print(f"  R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, RSS = {rss:.4f}")

    # Plotting
    if x is not None:
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.figure(figsize=(5, 5))
        plt.scatter(x*1000, y, color="#C54E4A", s=8)
        plt.plot(x*1000, y_pred, color="#3A6EA5", linewidth=1.5)
        plt.xlabel("1000/T")
        plt.ylabel("ln(1/IP)")
        plt.title(f"Arrhenius line for category {category}", fontweight='bold')
        # Plot the equation
        equation = (
            r"$y = {:.4f} - {:.4f}x$"
            "\n\n"
            r"$R^2 = {:.4f}$"
        ).format(
            np.log(k_0),
            E_a / R,
            r2
        )

        plt.text(
            0.9, 0.8,
            equation,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right"
        )
        plt.tight_layout()

        plot_path = f"outputs/oxidation/arrhenius_fit_{category}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved plot to: {plot_path}")

    #  Save results
    ox_results.append([category, E_a, k_0, IP_25_days])

# Combined plot with all categories
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.figure(figsize=(6, 5))
colors = plt.colormaps.get_cmap('tab10').colors

for (category, group), color in zip(cleaned.groupby('category'), colors):
    grouped = group.groupby('set_point_t')['ip_h'].mean().reset_index()
    temps_C = grouped['set_point_t'].astype(float).values
    ips = grouped['ip_h'].astype(float).values

    E_a, k_0, IP_25, r2, rmse, mae, rss, x, y, y_pred = fit_arrhenius_model(temps_C, ips)

    if x is not None:
        plt.plot(x*1000, y_pred, label=f"{category}", color=color, linewidth=1.5)

plt.xlabel("1000/T")
plt.ylabel("ln(1/IP)")
plt.title("Arrhenius Lines for All Categories", fontweight="bold")
plt.legend(fontsize=8, frameon=False)
plt.tight_layout()

combined_plot_path = "outputs/oxidation/arrhenius_fit_all.png"
plt.savefig(combined_plot_path, dpi=300)
plt.close()
print(f"\nCombined plot saved to: {combined_plot_path}")

# [Step 3]: Save results to CSV
ox_df = pd.DataFrame(ox_results, columns=['category', 'e_a', 'k_0', 'ip_days_25c'])
ox_output_path = 'outputs/oxidation/oxidation_params_by_category.csv'
ox_df.to_csv(ox_output_path, index=False)
print(f"Arrhenius parameters saved to: {ox_output_path}")