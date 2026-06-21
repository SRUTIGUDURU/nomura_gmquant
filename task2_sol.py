"""
Task 2: Client Profitability and Spread Recommendation
Computes expected PnL per horizon, classifies clients,
and recommends a minimum half‑spread to achieve non‑negative
expected aggregate PnL.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional

_DATA = None

def _get_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the trade data CSV once and caches it globally.
    If filepath is provided, uses it; otherwise falls back to a default.
    """
    global _DATA
    if _DATA is None:
        if filepath is None:
            # Use a relative path; adjust as needed or pass explicitly
            filepath = "trade_data.csv"
        _DATA = pd.read_csv(filepath)
        _DATA["Side"] = _DATA["Side"].astype(int)
    return _DATA


def expected_pnl(
    client: str,
    tau: List[int],
    filepath: Optional[str] = None
) -> Dict[str, Union[List[float], float]]:
    """
    Compute expected PnL per trade (LP perspective) for given client and horizons.
    
    Parameters:
        client: Client identifier (e.g., 'A', 'B')
        tau: List of horizons, e.g. [5, 10, 15, 20, 25, 30]
        filepath: Optional path to CSV data file
    
    Returns:
        Dictionary with keys:
            'per_horizon': List[float] – expected PnL at each tau (Eq.5)
            'aggregate':   float       – expected aggregate PnL (Eq.6)
    """
    df = _get_data(filepath)
    client_df = df[df["Name"] == client]
    if client_df.empty:
        raise ValueError(f"No trades found for client '{client}'")
    
    per_horizon = []
    for t in tau:
        mid_col = f"M{t}"
        pnl = client_df["Side"] * client_df["Volume"] * (client_df[mid_col] - client_df["Trade Price"])
        per_horizon.append(pnl.mean())
    
    # Aggregate PnL uses uniform weights over the six closing horizons
    mid_cols = [f"M{t}" for t in [5, 10, 15, 20, 25, 30]]
    avg_mid = client_df[mid_cols].mean(axis=1)
    agg_pnl = (client_df["Side"] * client_df["Volume"] * (avg_mid - client_df["Trade Price"])).mean()
    
    return {"per_horizon": per_horizon, "aggregate": agg_pnl}


def classify_client(client: str, filepath: Optional[str] = None) -> str:
    """
    Classify client as 'profitable' if expected aggregate PnL > 0, else 'costly'.
    """
    agg = expected_pnl(client, [5], filepath)["aggregate"]
    return "profitable" if agg > 0 else "costly"


def min_half_spread(client: str, filepath: Optional[str] = None) -> float:
    """
    Compute minimum half‑spread δ* such that expected aggregate PnL >= 0
    when quoting at M0 ± δ* for all trades.
    
    Derivation:
        If we quote at M0 - side·δ, then trade price = M0 - side·δ.
        Aggregate PnL = side·V·(avg_mid - (M0 - side·δ))
                     = side·V·(avg_mid - M0) + δ·V.
        Expectation: E[side·V·(avg_mid - M0)] + δ·E[V] >= 0
        => δ >= -E[side·V·(avg_mid - M0)] / E[V].
    """
    df = _get_data(filepath)
    client_df = df[df["Name"] == client]
    if client_df.empty:
        raise ValueError(f"No trades found for client '{client}'")
    
    mid_cols = [f"M{t}" for t in [5, 10, 15, 20, 25, 30]]
    avg_mid = client_df[mid_cols].mean(axis=1)
    raw_pnl = client_df["Side"] * client_df["Volume"] * (avg_mid - client_df["M0"])
    numerator = raw_pnl.mean()
    denominator = client_df["Volume"].mean()
    if denominator == 0:
        return 0.0
    delta = -numerator / denominator
    return max(0.0, delta)


if __name__ == "__main__":
    # Example usage: read data, compute results, save CSV
    DATA_PATH = "trade_data.csv"  # Adjust if needed
    df = _get_data(DATA_PATH)
    clients = df["Name"].unique()
    tau_list = [5, 10, 15, 20, 25, 30]
    
    rows = []
    for c in clients:
        exp = expected_pnl(c, tau_list, DATA_PATH)
        per_h = exp["per_horizon"]
        agg = exp["aggregate"]
        delta = min_half_spread(c, DATA_PATH)
        rows.append([c] + per_h + [agg, delta])
    
    # Use plain ASCII column names (Excel compatible)
    cols = ["client"] + [f"tau={t}" for t in tau_list] + ["agg_pnl", "delta*"]
    out_df = pd.DataFrame(rows, columns=cols)
    out_df.to_csv("task2_results.csv", index=False)
    print("Saved task2_results.csv")