import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# GAB Model (Moisture)
def gab_model(aw, W_m, C, K):
    """
    GAB isotherm: moisture content M(aw) (g water / g dry solid).
    Parameters:
        aw: Water activity (dimensionless)
        W_m: Monolayer moisture content (g water / g dry solid)
        C: GAB constant (dimensionless)
        K: GAB constant (dimensionless)
    Returns:
        Moisture content M(aw) (g water / g dry solid)
    """
    return W_m * C * K * aw / ((1 - K * aw) * (1 - K * aw + C * K * aw))

def fit_gab_model(aw, moisture, init_params=[0.05, 10.0, 0.8]):
    """
    Fit GAB parameters W_m, C, K by non-linear regression.
    Parameters:
        aw: Water activity (array-like)
        moisture: Moisture content (array-like)
        init_params: Initial guess for parameters [W_m, C, K]
    Returns:
        Fitted parameters [W_m, C, K] or NaN
    """
    try:
        params, _ = curve_fit(gab_model, aw, moisture, p0=init_params, maxfev=10000)
        return params
    except RuntimeError:
        return [np.nan, np.nan, np.nan]

# Read data
input_excel = 'data/experimental_data.xlsx'
moisture_df = pd.read_excel(input_excel, sheet_name='moisture')

gab_results = []
os.makedirs("outputs/moisture", exist_ok=True)

# Fit GAB parameters by category
print("Fitting GAB parameters by category...")
for category, group in moisture_df.groupby('category'):
    aw = pd.to_numeric(group['water_activity'], errors='coerce')
    moisture = pd.to_numeric(group['moisture_content'], errors='coerce')
    valid = (~aw.isna()) & (~moisture.isna())
    aw_clean = aw[valid].values
    moisture_clean = moisture[valid].values

    # Fit GAB params
    params = fit_gab_model(aw_clean, moisture_clean)

    if not np.isnan(params[0]):
        # Predictions
        moisture_pred = gab_model(aw_clean, *params)

        # Evaluation metrics
        r2 = r2_score(moisture_clean, moisture_pred)
        rmse = np.sqrt(mean_squared_error(moisture_clean, moisture_pred))
        mae = mean_absolute_error(moisture_clean, moisture_pred)
        rss = np.sum((moisture_clean - moisture_pred) ** 2)

        # Print summary
        print(f"\nCategory: {category}")
        print(f"  W_m={params[0]:.4f}, C={params[1]:.4f}, K={params[2]:.4f}")
        print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, RSS: {rss:.4f}")

        # Plot fitted curve
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10
        plt.figure(figsize=(5, 5))
        aw_range = np.linspace(0.1, 0.95, 200)
        plt.plot(aw_range, gab_model(aw_range, params[0],params[1],params[2]), label=f"Category {category}")

        plt.xlabel("Water activity")
        plt.ylabel("Moisture content")
        plt.title(f"GAB Isotherm for category {category}", fontweight='bold')

        # Plot the equation
        coef = params[1] * params[2] - params[2]

        equation = (
            r"$w = \frac{{{:.2f}\, a_w}}{{(1 - {:.2f}a_w)(1 {} {:.2f}a_w)}}$"
            "\n\n"
            r"$R^2 = {:.4f}$"
        ).format(
            params[0] * params[1] * params[2],
            params[2],
            '+' if coef >= 0 else '-',
            abs(coef),
            r2
        )

        plt.text(
            0.10, 0.80,
            equation,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='left'
        )

        plt.legend()
        plt.tight_layout()

        plot_output_path = f"outputs/moisture/gab_isotherm_{category}.png"
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        print(f"  Saved plot to: {plot_output_path}")
    else:
        print(f"\nCategory: {category} - GAB fitting failed.")

    # Save GAB parameters
    gab_results.append([category] + list(params))

# Save to CSV
gab_df = pd.DataFrame(gab_results, columns=['category', 'W_m', 'C', 'K'])
gab_output_path = 'outputs/moisture/gab_params_by_category.csv'
gab_df.to_csv(gab_output_path, index=False)

# Print summary
print(f"\nGAB parameters saved to: {gab_output_path}")