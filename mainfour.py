import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Read Excel file ===
excel_path = input("Enter the path to your Excel file (e.g., random_data.xlsx): ")

try:
    df = pd.read_excel(excel_path)

    # === Step 2: Check required columns (excluding 'SL' and 'sigma_demand' which can be defaulted) ===
    required_columns = {"price", "demand", "LT"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Handle 'h' if missing
    if "h" not in df.columns:
        print("Column 'h' is missing. Default value of 0.15 will be used for all rows.")
        df["h"] = 0.15

    # Handle 'SL' if missing
    if "SL" not in df.columns:
        try:
            sl_input = float(input("SL column not found. Please enter a default Service Level (e.g., 95): "))
        except ValueError:
            raise ValueError("Invalid SL input. Please enter a numeric value.")
        df["SL"] = sl_input
        print(f"All rows will use SL = {sl_input}")

    # === Step 3: Z-score mapping with rounding ===
    z_score_lookup = {
        50: 0.00, 60: 0.25, 70: 0.52, 75: 0.67, 80: 0.84, 85: 1.04,
        90: 1.28, 92: 1.41, 95: 1.65, 96: 1.75, 97: 1.88, 98: 2.05,
        99: 2.33, 99.5: 2.58, 99.9: 3.08, 99.99: 3.72
    }

    valid_levels = np.array(list(z_score_lookup.keys()))
    df["SL_rounded"] = df["SL"].apply(lambda x: valid_levels[np.abs(valid_levels - x).argmin()])
    df["Z_score"] = df["SL_rounded"].map(z_score_lookup)

    if df["Z_score"].isnull().any():
        invalid_sl = df[df["Z_score"].isnull()]["SL"].unique()
        raise ValueError(f"Could not assign Z-score for SL values: {invalid_sl}")

    # === Step 4: Expanded probability matrix for 2–75 companies ===
    prob_matrix = {
        2: [0.8509, 0.1491, 0.0, 0.0, 0.0, 0.0],
        3: [0.72374, 0.25394, 0.02232, 0.0, 0.0, 0.0],
        4: [0.61374, 0.32507, 0.0578, 0.00339, 0.0, 0.0],
        5: [0.52183, 0.36639, 0.09947, 0.01173, 0.00058, 0.0],
        6: [0.44425, 0.39142, 0.13752, 0.02462, 0.00215, 0.00004],
        7: [0.37522, 0.4006, 0.17671, 0.04176, 0.00541, 0.0003],
        8: [0.31938, 0.39553, 0.21159, 0.06093, 0.01132, 0.00125],
        9: [0.2722, 0.38379, 0.23837, 0.08505, 0.01779, 0.0028],
        10: [0.23169, 0.36614, 0.2592, 0.10879, 0.02839, 0.00579],
        11: [0.19551, 0.34463, 0.27758, 0.13092, 0.04134, 0.01002],
        12: [0.16797, 0.32258, 0.28665, 0.1522, 0.05461, 0.01599],
        13: [0.14272, 0.30025, 0.29426, 0.17124, 0.06824, 0.02329],
        14: [0.12129, 0.27869, 0.2934, 0.18944, 0.08249, 0.03469],
        15: [0.10248, 0.25634, 0.29232, 0.20196, 0.09904, 0.04786],
        16: [0.08668, 0.23304, 0.2859, 0.21608, 0.11669, 0.06161],
        17: [0.07558, 0.2077, 0.27789, 0.22797, 0.13156, 0.0793],
        18: [0.06365, 0.18939, 0.267, 0.23477, 0.14526, 0.09993],
        19: [0.05463, 0.1716, 0.25247, 0.2428, 0.158, 0.1205],
        20: [0.04428, 0.15124, 0.24446, 0.24503, 0.17066, 0.14433],
        21: [0.03849, 0.13629, 0.22926, 0.24364, 0.18236, 0.16996],
        22: [0.03277, 0.12016, 0.21586, 0.2432, 0.19181, 0.1962],
        23: [0.02877, 0.10904, 0.2016, 0.23718, 0.19772, 0.22569],
        24: [0.02405, 0.09767, 0.18682, 0.23228, 0.20259, 0.25659],
        25: [0.02092, 0.08562, 0.17606, 0.22385, 0.20658, 0.28697],
        26: [0.01693, 0.07628, 0.15962, 0.215, 0.21168, 0.32049],
        27: [0.01457, 0.0678, 0.14757, 0.21125, 0.21021, 0.3486],
        28: [0.01251, 0.05879, 0.13511, 0.19896, 0.2103, 0.38433],
        29: [0.01026, 0.05114, 0.12447, 0.19187, 0.20877, 0.41349],
        30: [0.00885, 0.04656, 0.11315, 0.18078, 0.20513, 0.44553],
        31: [0.00808, 0.04035, 0.10361, 0.16851, 0.20282, 0.47663],
        32: [0.0062, 0.03558, 0.09457, 0.1593, 0.19684, 0.50751],
        33: [0.0057, 0.03055, 0.08496, 0.15123, 0.19459, 0.53297],
        34: [0.00488, 0.02678, 0.07669, 0.14015, 0.18764, 0.56386],
        35: [0.00434, 0.02411, 0.06967, 0.13282, 0.17773, 0.59133],
        36: [0.0036, 0.02091, 0.06239, 0.123, 0.17038, 0.61972],
        37: [0.00305, 0.01819, 0.05639, 0.11199, 0.16427, 0.64611],
        38: [0.00261, 0.0166, 0.04979, 0.10629, 0.15571, 0.669],
        39: [0.00229, 0.01463, 0.04596, 0.09651, 0.14924, 0.69137],
        40: [0.00177, 0.01167, 0.04168, 0.08971, 0.14188, 0.71329],
        41: [0.00146, 0.01041, 0.03677, 0.0816, 0.13208, 0.73768],
        42: [0.00132, 0.00887, 0.03152, 0.07595, 0.12636, 0.75598],
        43: [0.00122, 0.00797, 0.02808, 0.0672, 0.11663, 0.7789],
        44: [0.00087, 0.00709, 0.02584, 0.06329, 0.11191, 0.791],
        45: [0.00076, 0.00616, 0.02253, 0.0573, 0.10447, 0.80878],
        46: [0.00075, 0.00503, 0.02023, 0.05206, 0.09707, 0.82486],
        47: [0.00057, 0.00483, 0.0181, 0.04772, 0.0875, 0.84128],
        48: [0.00065, 0.00399, 0.01693, 0.04228, 0.08204, 0.85411],
        49: [0.00037, 0.00367, 0.01417, 0.03916, 0.07676, 0.86587],
        50: [0.00039, 0.00279, 0.013, 0.03503, 0.07011, 0.87868],
        51: [0.00021, 0.00245, 0.01117, 0.0317, 0.06476, 0.88971],
        52: [0.00031, 0.00227, 0.01057, 0.02886, 0.06049, 0.8975],
        53: [0.00028, 0.00177, 0.00909, 0.02566, 0.05577, 0.90743],
        54: [0.00014, 0.00163, 0.0076, 0.02351, 0.05165, 0.91547],
        55: [0.00022, 0.00156, 0.00684, 0.02194, 0.04741, 0.92203],
        56: [0.00014, 0.00122, 0.0058, 0.01844, 0.0426, 0.9318],
        57: [0.00017, 0.00098, 0.00549, 0.01666, 0.03902, 0.93768],
        58: [0.00006, 0.00094, 0.00475, 0.01487, 0.03688, 0.9425],
        59: [0.00005, 0.00094, 0.00419, 0.01333, 0.03317, 0.94832],
        60: [0.00014, 0.00067, 0.00372, 0.01258, 0.03013, 0.95276],
        61: [0.00007, 0.00067, 0.00299, 0.01107, 0.02791, 0.95729],
        62: [0.00006, 0.00072, 0.0028, 0.00982, 0.02494, 0.96166],
        63: [0.00001, 0.00038, 0.0022, 0.00871, 0.02287, 0.96583],
        64: [0.00003, 0.00042, 0.00237, 0.00768, 0.0202, 0.9693],
        65: [0.00002, 0.00036, 0.00202, 0.00726, 0.01919, 0.97115],
        66: [0.00003, 0.0003, 0.00169, 0.0063, 0.01675, 0.97493],
        67: [0.00003, 0.00034, 0.00134, 0.00593, 0.01574, 0.97662],
        68: [0.00003, 0.00022, 0.00137, 0.00512, 0.01422, 0.97904],
        69: [0.00003, 0.00015, 0.00131, 0.0042, 0.01248, 0.98183],
        70: [0.00002, 0.00016, 0.00114, 0.00373, 0.01098, 0.98397],
        71: [0.0, 0.00013, 0.00082, 0.00352, 0.00993, 0.9856],
        72: [0.0, 0.00012, 0.00084, 0.00295, 0.00969, 0.9864],
        73: [0.0, 0.00017, 0.00076, 0.00307, 0.00796, 0.98804],
        74: [0.00001, 0.00014, 0.00054, 0.00245, 0.00693, 0.98993],
        75: [0.0, 0.00011, 0.00045, 0.00192, 0.00644, 0.99108]
    }

    # === Step 5: User input for number of companies ===
    try:
        num_companies = int(input("Enter the number of companies in the pooling system (2–75): "))
    except ValueError:
        raise ValueError("Invalid input. Please enter an integer.")

    if num_companies not in prob_matrix:
        raise ValueError(f"Number of companies {num_companies} not supported. Choose between 2 and 75.")

    # === Step 6: Perform calculations ===
    df["H"] = df["price"] * df["h"]
    df["DLT"] = df["demand"] * df["LT"]  # Demand during lead time

    # Handle sigma_demand (row-wise calculation if missing)
    if "sigma_demand" in df.columns:
        print("Using 'sigma_demand' values from Excel (row-wise).")
    else:
        print("'sigma_demand' not found. Calculating sigma_demand per row as sqrt(demand).")
        df["sigma_demand"] = df["demand"].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)

    # Initial Safety Stock calculation: SS_initial = sigma_demand * sqrt(lead time) * z-score
    df["SS_initial"] = df["sigma_demand"] * np.sqrt(df["LT"]) * df["Z_score"]

    # Total initial cost: TC_initial = H × (DLT + SS_initial)
    df["TC_initial"] = df["H"] * (df["DLT"] + df["SS_initial"])

    # Calculate TC_new using probability-weighted safety stock
    probs = prob_matrix[num_companies]

    def calculate_tc_new(row):
        ss_initial = row["SS_initial"]
        H = row["H"]
        DLT = row["DLT"]
        weighted_ss = sum([
            probs[k] * ss_initial / np.sqrt(k + 1)
            for k in range(5)
        ]) + probs[5] * ss_initial / np.sqrt(6)
        return H * (DLT + weighted_ss)

    df["TC_new"] = df.apply(calculate_tc_new, axis=1)
    df["TC_savings_per_item"] = df["TC_initial"] - df["TC_new"]

    # === Step 7: Totals ===
    total_initial_cost = df["TC_initial"].sum()
    total_new_cost = df["TC_new"].sum()
    total_savings = df["TC_savings_per_item"].sum()
    percent_savings = (total_savings / total_initial_cost) * 100 if total_initial_cost > 0 else 0

    # === Step 8: Output ===
    print("\nCalculation complete. Preview:")
    print(df[[
        "price", "h", "demand", "LT", "SL", "SL_rounded", "sigma_demand",
        "Z_score", "H", "DLT", "SS_initial",
        "TC_initial", "TC_new", "TC_savings_per_item"
    ]].head())

    print("\nSummary:")
    print(f"Total Initial Cost:    {total_initial_cost:,.2f}")
    print(f"Total New Cost:        {total_new_cost:,.2f}")
    print(f"Total Cost Savings:    {total_savings:,.2f}")
    print(f"Percentual Savings:    {percent_savings:.2f}%")

    # === Step 9: Export ===
    export_path = excel_path.replace(".xlsx", "_with_probabilistic_costs.xlsx")
    df.to_excel(export_path, index=False)
    print(f"\nResults exported to: {export_path}")

    # === Step 10: Plotting ===
    plt.figure(figsize=(6, 4))
    costs = [total_initial_cost, total_new_cost]
    labels = ["Initial Total Cost", "New Total Cost"]
    colors = ["gray", "green"]

    plt.bar(labels, costs, color=colors)
    plt.title("Total Cost Comparison")
    plt.ylabel("Total Cost")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i, v in enumerate(costs):
        plt.text(i, v + max(costs)*0.01, f"{v:,.2f}", ha='center', fontweight='bold')

    plt.text(0.5, min(costs) * 0.95,
             f"Cost Reduction: {percent_savings:.2f}%",
             ha='center', fontsize=11, bbox=dict(facecolor='lightblue', edgecolor='black'))

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("File not found. Please check the path.")
except Exception as e:
    print(f"Error: {e}")
