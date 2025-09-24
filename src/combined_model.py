import os
import numpy as np
import pandas as pd
import argparse

R = 8.314  # J/mol·K

# GAB model -- Moisture sorption
def gab_model(aw, W_m, C, K):
    """GAB isotherm: moisture content M(aw) (g water / g dry solid)."""
    return W_m * C * K * aw / ((1 - K * aw) * (1 - K * aw + C * K * aw))

def predict_shelf_life_moisture(W_m, C, K,
                            W_s=200,
                            A=0.1,
                            WVTR=0.5,
                            aw_c=0.6,
                            aw_0=0.3):
    """
    Shelf life t_s = (W_s / (A * WVTR)) * [ M(aw_c) - M(aw_0) ]
    where M(aw) is moisture content from GAB model.

    Parameters:
       W_m: monolayer moisture content (g water / g dry solid)
       C: GAB constant
       K: GAB constant
       W_s: dry solid weight (g)
       A: package area (m²)
       WVTR: water vapor transmission rate (g/(m²·day))
       aw_c: critical water activity (dimensionless)
       aw_0: initial water activity (dimensionless)
    Returns:
       Shelf life (day)
    """
    M_c = gab_model(aw_c, W_m, C, K)
    M_0 = gab_model(aw_0, W_m, C, K)
    return (W_s / (A * WVTR)) * (M_c - M_0)



# Arrhenius model -- Lipid oxidation
def predict_shelf_life_oxidation(E_a, k_0, T_C=25):
    """
    Shelf life (days) from Arrhenius oxidation model.
    Parameters:
       E_a: activation energy (kJ/mol)
       k_0: pre-exponential factor (1/h)
       T_C: temperature (°C)
    Returns:
       Shelf life (days)
    """
    if np.isnan(E_a) or np.isnan(k_0):
        return np.nan
    T_K = T_C + 273.15
    k_T = k_0 * np.exp(-E_a * 1000 / (R * T_K))  # 1/h
    IP_h = 1 / k_T
    return IP_h / 24.0  # convert to days



# Default parameters and category mapping
CATEGORY_MAP = {
    "C": "Crackers",
    "D": "Sweet biscuit",
    "F": "Cream filling",
    "O": "Snack (oil)",
    "P": "Baked snack",
    "S": "Sandwich biscuit",
    "W": "Sugar Wafers"
}

def get_aw_values(category):
    """Return aw_c and aw_0 depending on category rules"""
    if category in ["C", "D", "P", "S"]:
        aw_c = 0.6
    else:
        aw_c = 0.5  # fallback
    
    if category in ["D", "S"]:
        aw_0 = 0.4
    else:
        aw_0 = 0.3
    return aw_c, aw_0



# Combined model -- minimum of both shelf lives
def predict_shelf_life_combined(category, moisture_df, oxidation_df, T=25, W_s=200, A=0.1, WVTR=0.5):
    W_m = C = K = np.nan
    E_a = k_0 = np.nan

    if category in moisture_df["category"].values:
        row = moisture_df[moisture_df["category"] == category].iloc[0]
        W_m, C, K = row["W_m"], row["C"], row["K"]

    if category in oxidation_df["category"].values:
        row = oxidation_df[oxidation_df["category"] == category].iloc[0]
        E_a, k_0 = row["e_a"], row["k_0"]

    # Predict shelf life from both models
    aw_c, aw_0 = get_aw_values(category)
    shelf_life_moisture = np.nan
    shelf_life_oxidation = np.nan

    if not np.isnan(W_m):
        shelf_life_moisture = predict_shelf_life_moisture(W_m, C, K,
                                                 W_s=W_s, A=A, WVTR=WVTR,
                                                 aw_c=aw_c, aw_0=aw_0)
    if not np.isnan(E_a):
        shelf_life_oxidation = predict_shelf_life_oxidation(E_a, k_0, T)

    # Decide dominant model
    if not np.isnan(shelf_life_moisture) and not np.isnan(shelf_life_oxidation):
        if shelf_life_moisture <= shelf_life_oxidation:
            return shelf_life_moisture, "moisture gain"
        else:
            return shelf_life_oxidation, "lipid oxidation"
    elif not np.isnan(shelf_life_moisture):
        return shelf_life_moisture, "moisture gain"
    elif not np.isnan(shelf_life_oxidation):
        return shelf_life_oxidation, "lipid oxidation"
    else:
        return np.nan, "none"



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Shelf life prediction from moisture & oxidation models.")
    parser.add_argument("-c", "--category", type=str, default= 'C',
                        help="Category code (C,D,F,O,P,S,W), where C=Crackers, D=Sweet biscuit, F=Cream filling, O=Snack (oil), P=Baked snack, S=Sandwich biscuit, W=Sugar Wafers. If not provided, default predictions for all categories will be generated.")
    parser.add_argument("-t", "--temperature", type=float, default=25.0,
                        help="Storage temperature in °C (for oxidation model)")
    parser.add_argument("-ws", "--dry_weight", type=float, default=200.0,
                        help="Dry solid weight (g)")
    parser.add_argument("-a", "--area", type=float, default=0.1,
                        help="Package area (m²)")
    parser.add_argument("-wvtr", "--wvtr", type=float, default=0.5,
                        help="Packaging Water Vapour Transmission Rate(WVTR) (g/(m²·day))")

    args = parser.parse_args()

    # Load fitted parameters
    moisture_path = "outputs/moisture/gab_params_by_category.csv"
    oxidation_path = "outputs/oxidation/oxidation_params_by_category.csv"
    if not (os.path.exists(moisture_path) and os.path.exists(oxidation_path)):
        print("Missing input CSV files. \nPlease run moisture_fit.py and oxidation_fit.py first.")
        return
    
    moisture_df = pd.read_csv(moisture_path)
    oxidation_df = pd.read_csv(oxidation_path)

    # Output to CSV for all categories with default params
    results = []
    for cat in CATEGORY_MAP.keys():
        shelf_life, model = predict_shelf_life_combined(cat, moisture_df, oxidation_df, T=25, W_s=200, A=0.1, WVTR=0.5)
        results.append([cat, CATEGORY_MAP[cat], shelf_life, model])

    out_path = "outputs/shelf_life_predictions_by_category.csv"
    os.makedirs("outputs", exist_ok=True)
    df_out = pd.DataFrame(results, columns=["category_code", "category_name", "shelf_life_pred_days", "dominant_model"])
    df_out.to_csv(out_path, index=False)
    print(f"Default predictions by category saved to {out_path}")

    # Output to command line for specific category with custom params
    if args.category:
        cat = args.category.upper()
        if cat in CATEGORY_MAP:
            shelf, model = predict_shelf_life_combined(cat, moisture_df, oxidation_df,
                                                T=args.temperature,
                                                W_s=args.dry_weight,
                                                A=args.area,
                                                WVTR=args.wvtr)
            print("\n--- Shelf Life Prediction ---")
            print(f"Category: {cat} ({CATEGORY_MAP[cat]})")
            print(f"Temperature: {args.temperature} °C, W_s={args.dry_weight}, A={args.area}, WVTR={args.wvtr}")
            print(f"Shelf life: {shelf:.2f} days")
            print(f"Dominant model: {model}\n")
        else:
            print(f"Invalid category code: {cat}. Must be one of {list(CATEGORY_MAP.keys())}")



if __name__ == "__main__":
    main()
