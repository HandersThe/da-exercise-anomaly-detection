import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sys
import os

def load_data(file_path):
    """Load CSV file and validate format."""
    try:
        df = pd.read_csv(file_path)
        if list(df.columns) != ["Total Sales", "Date"]:
            raise ValueError("CSV must have exactly two columns: 'Total Sales' and 'Date'")
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df = df.drop_duplicates(subset="Date", keep="first").sort_values("Date")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def process_sales_data(df, contamination):
    """Process data: compute daily sales, fill gaps, detect and correct anomalies."""
    # Compute initial daily sales/delta and flag true deltas
    df["Daily_Sales"] = df["Total Sales"].diff().fillna(0)
    df["Notes"] = "Delta"  # Mark original deltas
    df.loc[df.index == 0, "Notes"] = "Assumed"

    # Fill missing dates
    full_dates = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="D")
    df_full = df.set_index("Date").reindex(full_dates).reset_index()
    df_full = df_full.rename(columns={"index": "Date"})

    # Fill gaps in daily sales by averaging deltas, flag all as Assumed
    df_final = df_full.copy()
    for i in range(1, len(df_final)):
        if pd.isna(df_final.loc[i, "Total Sales"]) or (i == len(df_final)-1):
            start_idx = i - 1
            while start_idx >= 0 and pd.isna(df_final.loc[start_idx, "Total Sales"]):
                start_idx -= 1
            if start_idx < 0:
                start_idx = 0
            end_idx = i
            while end_idx < len(df_final) and pd.isna(df_final.loc[end_idx, "Total Sales"]):
                end_idx += 1
            if end_idx < len(df_final):
                start_value = df_final.loc[start_idx, "Total Sales"]
                end_value = df_final.loc[end_idx, "Total Sales"]
                if not pd.isna(start_value) and not pd.isna(end_value):
                    total_delta = end_value - start_value
                    gap_size = end_idx - start_idx
                    avg_delta = round(total_delta / gap_size)  # Integer
                    for j in range(start_idx + 1, end_idx + 1):
                        df_final.loc[j, "Daily_Sales"] = avg_delta
                        df_final.loc[j, "Notes"] = "Assumed"

    df_final["Daily_Sales"] = df_final["Daily_Sales"].fillna(0).astype(int)
    df_final["Notes"] = df_final["Notes"].fillna("Assumed")  # initial missing days

    # Feature engineering for IsolationForest
    df_final["Lag1"] = df_final["Daily_Sales"].shift(1).fillna(0).astype(int)
    df_final["Rolling_Mean"] = df_final["Daily_Sales"].rolling(window=3, min_periods=1).mean().fillna(0)
    features = df_final[["Daily_Sales", "Lag1", "Rolling_Mean"]]

    # Detect anomalies with IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    df_final["Anomaly_Score"] = iso_forest.fit_predict(features)
    iso_anomalies = df_final["Anomaly_Score"] == -1

    # Explicitly flag negatives
    negatives = df_final["Daily_Sales"] < 0
    anomalies = iso_anomalies | negatives

    # Store original anomalies for visualization
    anomaly_dates = df_final[anomalies]["Date"]
    original_anomalies = df[df["Date"].isin(anomaly_dates)][["Date", "Daily_Sales"]]

    # Interpolate all anomalous daily sales and ensure integers
    df_final.loc[anomalies, "Daily_Sales"] = np.nan
    df_final["Daily_Sales"] = df_final["Daily_Sales"].interpolate(method="linear").fillna(0)
    df_final["Daily_Sales"] = df_final["Daily_Sales"].round().astype(int)
    df_final.loc[anomalies, "Notes"] = "Assumed"  # Mark interpolated values

    print(f"Isolation Forest anomalies: {iso_anomalies.sum()}")
    print(f"Negative anomalies: {negatives.sum()}")
    print(f"Total anomalies corrected: {anomalies.sum()}")
    print(f"Negatives remaining: {(df_final["Daily_Sales"] < 0).sum()}")

    return df_final, original_anomalies

def save_outputs(df_final, original_anomalies, output_csv, output_plot):
    """Save corrected CSV and visualization."""
    # Save corrected CSV with Notes
    df_final[["Date", "Total Sales", "Daily_Sales", "Notes"]].to_csv(output_csv, index=False)
    print(f"Corrected data saved to: {output_csv}")

    # Plot and save visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df_final["Date"], df_final["Daily_Sales"], label="Corrected Daily Sales", color="blue")
    plt.scatter(original_anomalies["Date"], original_anomalies["Daily_Sales"], 
                color="red", label="Original Anomalies", zorder=5)
    plt.axhline(0, color="black", linestyle="--", label="Zero Line")
    plt.title("Daily Sales with Anomalies Corrected (Isolation Forest)")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"Visualization saved to: {output_plot}")

def main():
    # Get input file
    input_file = input("Enter the path to your CSV file (e.g., 'sales_data.csv'): ").strip()
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # Get contamination value
    while True:
        contamination_input = input("Enter contamination value for IsolationForest (0.01-0.5, or 'auto', e.g., 0.1): ").strip().lower()
        if contamination_input == "auto":
            contamination = "auto"
            break
        try:
            contamination = float(contamination_input)
            if 0.01 <= contamination <= 0.5:
                break
            print("Please enter a numeric value between 0.01 and 0.5, or 'auto'.")
        except ValueError:
            print("Invalid input. Please enter a numeric value between 0.01 and 0.5, or 'auto'.")

    # Define output paths
    base_name = os.path.splitext(input_file)[0]
    output_csv = f"{base_name}_corrected.csv"
    output_plot = f"{base_name}_plot.png"

    # Process data
    df = load_data(input_file)
    df_final, original_anomalies = process_sales_data(df, contamination)
    save_outputs(df_final, original_anomalies, output_csv, output_plot)

if __name__ == "__main__":
    main()