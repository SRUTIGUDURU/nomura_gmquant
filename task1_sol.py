import pandas as pd
from typing import List

_DATA = None

def _get_data():
    global _DATA
    if _DATA is None:
        _DATA = pd.read_csv(r"trade_data.csv")
        _DATA["Side"] = _DATA["Side"].astype(int)
    return _DATA

def adversity_profile(client: str, tau: List[int]) -> List[float]:
    df = _get_data()
    client_df = df[df["Name"] == client]
    result = []
    for t in tau:
        mid_col = f"M{t}"
        pnl = client_df["Side"] * client_df["Volume"] * (client_df[mid_col] - client_df["Trade Price"])
        result.append((pnl < 0).mean() * 100.0)
    return result

if __name__ == "__main__":
    df = _get_data()
    clients = df["Name"].unique()
    tau_list = [5,10,15,20,25,30]
    rows = [[c] + adversity_profile(c, tau_list) for c in clients]
    out = pd.DataFrame(rows, columns=["client"] + [f"Tau={t}" for t in tau_list])
    out.to_csv(r"task1_results.csv", index=False)