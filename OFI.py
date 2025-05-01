import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# ==============================================================================
# 1. Data Loading and Preprocessing
# ==============================================================================
df = pd.read_csv("first_25000_rows.csv")
df["ts_event"] = pd.to_datetime(df["ts_event"])
df = df.sort_values("ts_event").reset_index(drop=True)

# ==============================================================================
# 2. Configuration Parameters
# ==============================================================================
bucket_size = pd.Timedelta("1min")  # Time aggregation window
M = 10  # Depth: number of LOB levels to compute OFI on

# ==============================================================================
# 3. OFI Computation at Individual Depth Level
# ==============================================================================
def compute_ofi_level(df, m, prev_bid_px, prev_bid_sz, prev_ask_px, prev_ask_sz):
    bid_px_col = f"bid_px_{m:02d}"
    ask_px_col = f"ask_px_{m:02d}"
    bid_sz_col = f"bid_sz_{m:02d}"
    ask_sz_col = f"ask_sz_{m:02d}"

    bid_px = df[bid_px_col].values
    ask_px = df[ask_px_col].values
    bid_sz = df[bid_sz_col].values
    ask_sz = df[ask_sz_col].values

    bid_ofi = np.where(bid_px > prev_bid_px, bid_sz,
                       np.where(bid_px < prev_bid_px, -bid_sz,
                                bid_sz - prev_bid_sz))

    ask_ofi = np.where(ask_px > prev_ask_px, -ask_sz,
                       np.where(ask_px < prev_ask_px, ask_sz,
                                ask_sz - prev_ask_sz))

    return bid_ofi - ask_ofi

# ==============================================================================
# 4. Time Bucketing and Multi-Level OFI Aggregation
# ==============================================================================
start_time = df["ts_event"].iloc[0]
end_time = df["ts_event"].iloc[-1]
time_buckets = pd.date_range(start=start_time, end=end_time, freq=bucket_size)

ofi_vectors = []
timestamps = []

for i in range(len(time_buckets) - 1):
    t_start = time_buckets[i]
    t_end = time_buckets[i + 1]
    window = df[(df["ts_event"] > t_start) & (df["ts_event"] <= t_end)].copy()

    if len(window) < 2:
        continue

    ofi_multi = []

    for m in range(M):
        bid_px_col = f"bid_px_{m:02d}"
        ask_px_col = f"ask_px_{m:02d}"
        bid_sz_col = f"bid_sz_{m:02d}"
        ask_sz_col = f"ask_sz_{m:02d}"

        prev_bid_px = window[bid_px_col].shift(1).bfill()
        prev_bid_sz = window[bid_sz_col].shift(1).bfill()
        prev_ask_px = window[ask_px_col].shift(1).bfill()
        prev_ask_sz = window[ask_sz_col].shift(1).bfill()

        ofi = compute_ofi_level(window, m, prev_bid_px, prev_bid_sz, prev_ask_px, prev_ask_sz)
        ofi_multi.append(ofi.sum())

    ofi_vectors.append(ofi_multi)
    timestamps.append(t_end)

# ==============================================================================
# 5. PCA-Based Integrated OFI Extraction
# ==============================================================================
ofi_matrix = np.vstack(ofi_vectors)

if np.all(ofi_matrix == 0):
    weights = np.ones(M) / M
else:
    pca = PCA(n_components=1)
    pca.fit(ofi_matrix)
    weights = np.abs(pca.components_[0])
    weights /= weights.sum()

# ==============================================================================
# 6. Final Output Construction
# ==============================================================================
results = []
for i in range(len(ofi_vectors)):
    vec = np.array(ofi_vectors[i])
    integrated_ofi = np.dot(vec, weights)
    results.append({
        "timestamp": timestamps[i],
        "ofi_best_level": vec[0],
        "ofi_integrated": integrated_ofi,
        **{f"ofi_level_{m + 1}": vec[m] for m in range(M)}
    })

ofi_all_df = pd.DataFrame(results)
print(ofi_all_df.head())

# ==============================================================================
# 7. Export Features to CSV
# ==============================================================================
output_path = "ofi_features_output.csv"
ofi_all_df.to_csv(output_path, index=False)
print(f"Features saved to: {output_path}")

# ==============================================================================
# 8. Cross-Asset OFI Modeling (To Be Enabled for Multi-Symbol Datasets)
# ==============================================================================
"""
# Add symbol identifier before aggregating multi-asset OFI features
ofi_all_df["symbol"] = current_symbol
ofi_all_list.append(ofi_all_df)

# Combine OFI features for all assets and pivot into wide format
ofi_all_combined = pd.concat(ofi_all_list)
ofi_wide = ofi_all_combined.pivot(index="timestamp", columns="symbol", values="ofi_integrated")

# Construct return matrix using mid-prices or last traded prices
returns = mid_price_data.pct_change().dropna()

# Estimate cross-impact using LASSO regression
from sklearn.linear_model import LassoCV

for symbol in ofi_wide.columns:
    X = ofi_wide.drop(columns=symbol).shift(1).fillna(0)
    y = returns[symbol]
    model = LassoCV().fit(X, y)
    beta = pd.Series(model.coef_, index=X.columns)
    print(f"Cross-Asset OFI impact on {symbol}:\n{beta.sort_values(ascending=False).head()}")
"""
