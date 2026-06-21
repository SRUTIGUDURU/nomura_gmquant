"""
Task 4: Optimal Externalization Threshold
------------------------------------------
- Uses LightGBM + CatBoost ensemble (tuned with Optuna) to predict adversity probability.
- Finds threshold(s) θ that maximize validation PnL.
- Supports global and client‑specific thresholds (client‑specific used here).
- Outputs task4_results.csv (client, tau, theta_star, final_pnl).
- Plots PnL_vs_theta for tau=5 (both global and client‑specific).
"""

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. Load data and feature engineering (applied globally before split)
def load_data(filepath="trade_data.csv"):
    df = pd.read_csv(filepath)
    df["Side"] = df["Side"].astype(int)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["time"])
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["second"] = df["datetime"].dt.second
    le = LabelEncoder()
    df["client_code"] = le.fit_transform(df["Name"])
    return df, le

def add_microstructure_features(df):
    df = df.copy()
    df["mid_return"] = df["M0"].pct_change()
    for span in [1, 5, 10]:
        df[f"ema_mid_{span}"] = df["M0"].ewm(span=span, adjust=False).mean()
        df[f"ema_return_{span}"] = df[f"ema_mid_{span}"].pct_change()
    for window in [5, 20, 50]:
        df[f"vol_{window}"] = df["mid_return"].rolling(window, min_periods=1).std().fillna(0)
    df["spread_to_vol_20"] = df["Spread"] / (df["vol_20"] + 1e-8)
    df["spread_change"] = df["Spread"].diff().fillna(0)
    df["spread_skew"] = df["Spread"].rolling(20).apply(lambda x: x.skew() if len(x)>3 else 0).fillna(0)
    df["log_volume"] = np.log1p(df["Volume"])
    df["signed_volume"] = df["Side"] * df["Volume"]
    df["volume_ema_5"] = df["Volume"].ewm(span=5, adjust=False).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ema_5"] + 1e-8)
    df["cum_signed_volume"] = df["signed_volume"].cumsum()
    df["ofi_20"] = df["signed_volume"].rolling(20).sum()
    df["client_volume_z"] = df.groupby("client_code")["Volume"].transform(
        lambda x: (x - x.rolling(20, min_periods=1).mean()) / (x.rolling(20, min_periods=1).std() + 1e-8)
    ).fillna(0)
    df["time_since_last"] = df["datetime"].diff().dt.total_seconds().fillna(1)
    df["trade_burst"] = (df["time_since_last"] < df["time_since_last"].quantile(0.25)).astype(int)
    df["client_trade_count"] = df.groupby("client_code").cumcount()
    df["client_freq"] = df.groupby("client_code")["time_since_last"].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    ).fillna(30)
    df["tp_m0_norm"] = (df["Trade Price"] - df["M0"]) / (df["Spread"] + 1e-8)
    df["side_tp"] = df["Side"] * df["tp_m0_norm"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Drop intermediate column
    df = df.drop(columns=["mid_return"], errors='ignore')
    return df

def split_by_time_with_embargo(df, tau, train_ratio=0.6, val_ratio=0.2):
    """
    Chronological split with embargo: no training trade has its M_tau
    after the start of validation. Uses positional indexing.
    """
    n = len(df)
    # initial splits (using integer positions)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))
    val_start_time = df.iloc[train_end_idx]['datetime']

    # Find last safe training row: trade_time + tau < val_start_time
    safe_end = train_end_idx
    for i in range(train_end_idx - 1, -1, -1):
        if df.iloc[i]['datetime'] + pd.Timedelta(seconds=tau) < val_start_time:
            safe_end = i + 1
            break

    train_df = df.iloc[:safe_end].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()
    return train_df, val_df, test_df

def add_client_adversity(train_df, val_df, test_df, tau):
    mid_col = f"M{tau}"
    for d in [train_df, val_df, test_df]:
        d["adverse"] = (d["Side"] * d["Volume"] * (d[mid_col] - d["Trade Price"]) < 0).astype(int)
    # Train: expanding mean
    train_df["client_adv_rate"] = train_df.groupby("client_code")["adverse"].expanding().mean().values
    train_df["client_adv_rate"] = train_df["client_adv_rate"].fillna(train_df["adverse"].mean())
    # Val: expanding within val, fallback to train mean
    val_df["client_adv_rate"] = val_df.groupby("client_code")["adverse"].expanding().mean().values
    val_df["client_adv_rate"] = val_df["client_adv_rate"].fillna(train_df["adverse"].mean())
    # Test: frozen from last val value
    last_val_adv = val_df.groupby("client_code")["client_adv_rate"].last().to_dict()
    global_fallback = train_df["adverse"].mean()
    test_df["client_adv_rate"] = test_df["Name"].map(last_val_adv).fillna(global_fallback)
    return train_df, val_df, test_df

# Feature columns (identical to Task 3)
FEATURE_COLS = [
    "client_code", "Side", "Volume", "log_volume", "signed_volume",
    "Spread", "spread_to_vol_20", "spread_change", "spread_skew",
    "time_since_last", "trade_burst", "client_trade_count", "client_freq",
    "ema_mid_1", "ema_mid_5", "ema_mid_10",
    "ema_return_1", "ema_return_5", "ema_return_10",
    "vol_5", "vol_20", "vol_50",
    "volume_ema_5", "volume_ratio", "cum_signed_volume", "ofi_20",
    "client_volume_z", "tp_m0_norm", "side_tp",
    "hour", "minute", "second", "hour_sin", "hour_cos",
    "client_adv_rate"
]

# 2. Optuna tuning (LightGBM and CatBoost)
def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss',
              callbacks=[lgb.early_stopping(10, verbose=False)])
    y_pred = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, y_pred)

def objective_catboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': 42,
        'verbose': False
    }
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10, verbose=False)
    y_pred = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, y_pred)

def optimize_model(model_type, X_train, y_train, X_val, y_val, n_trials=10):
    if model_type == 'lgb':
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective_lgb(t, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best['n_estimators'] = best.get('n_estimators', 300)
        best['verbose'] = -1
        best['random_state'] = 42
        return LGBMClassifier(**best)
    else:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective_catboost(t, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best['verbose'] = False
        best['random_seed'] = 42
        return CatBoostClassifier(**best)
# 3. Manual Platt scaling (replaces CalibratedClassifierCV)
def fit_platt(raw_probs, labels):
    def neg_ll(ab):
        p = np.clip(expit(ab[0] * raw_probs + ab[1]), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    res = minimize(neg_ll, [1.0, 0.0], method='L-BFGS-B')
    return res.x[0], res.x[1]

def apply_platt(raw, a, b):
    return expit(a * raw + b)

def train_ensemble(train_df, val_df, tau):
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df["adverse"]
    X_v = val_df[FEATURE_COLS]
    y_v = val_df["adverse"]
    
    print("  Tuning LightGBM...")
    lgb_model = optimize_model('lgb', X_tr, y_tr, X_v, y_v, n_trials=10)
    prob_lgb_uncal = lgb_model.predict_proba(X_v)[:, 1]
    
    print("  Tuning CatBoost...")
    cb_model = optimize_model('catboost', X_tr, y_tr, X_v, y_v, n_trials=10)
    prob_cb_uncal = cb_model.predict_proba(X_v)[:, 1]
    
    # Ensemble weights based on uncalibrated validation log-loss
    ll_lgb = log_loss(y_v, prob_lgb_uncal)
    ll_cb  = log_loss(y_v, prob_cb_uncal)
    w_lgb = 1.0 / (ll_lgb + 1e-8)
    w_cb  = 1.0 / (ll_cb + 1e-8)
    total = w_lgb + w_cb
    w_lgb /= total
    w_cb  /= total
    print(f"  Ensemble weights: LGB={w_lgb:.3f}, CB={w_cb:.3f}")

    # Calibrate on validation set using manual Platt scaling
    raw_val_lgb = lgb_model.predict_proba(X_v)[:, 1]
    a_lgb, b_lgb = fit_platt(raw_val_lgb, y_v.values)
    raw_val_cb = cb_model.predict_proba(X_v)[:, 1]
    a_cb, b_cb = fit_platt(raw_val_cb, y_v.values)

    # Store everything needed for prediction
    models = {
        'lgb': lgb_model, 'lgb_a': a_lgb, 'lgb_b': b_lgb,
        'cb': cb_model, 'cb_a': a_cb, 'cb_b': b_cb
    }
    weights = (w_lgb, w_cb)
    return models, weights

def predict_ensemble(models, weights, X):
    lgb_model = models['lgb']
    cb_model  = models['cb']
    w_lgb, w_cb = weights
    raw_lgb = lgb_model.predict_proba(X)[:, 1]
    raw_cb  = cb_model.predict_proba(X)[:, 1]
    prob_lgb = apply_platt(raw_lgb, models['lgb_a'], models['lgb_b'])
    prob_cb  = apply_platt(raw_cb,  models['cb_a'],  models['cb_b'])
    return w_lgb * prob_lgb + w_cb * prob_cb

# 4. PnL and threshold search (unchanged)
def compute_pnl_for_threshold(df, probs, tau, theta):
    mid_col = f"M{tau}"
    pnl_if_internal = df["Side"] * df["Volume"] * (df[mid_col] - df["Trade Price"])
    externalize = (probs > theta).astype(bool)
    pnl = pnl_if_internal.copy()
    pnl[externalize] = 0.0
    return pnl.sum()

def find_optimal_threshold(df, probs, tau, n_points=100):
    thetas = np.linspace(0, 1, n_points)
    pnls = [compute_pnl_for_threshold(df, probs, tau, t) for t in thetas]
    idx = np.argmax(pnls)
    return thetas[idx], pnls[idx], thetas, pnls

def find_client_specific_thresholds(df, probs, tau, clients, n_points=100):
    thresholds = {}
    val_pnls = {}
    for client in clients:
        mask = df["Name"] == client
        if mask.sum() == 0:
            thresholds[client] = 0.5
            val_pnls[client] = 0.0
            continue
        client_df = df[mask]
        client_probs = probs[mask]
        theta_opt, pnl_opt, _, _ = find_optimal_threshold(client_df, client_probs, tau, n_points)
        thresholds[client] = theta_opt
        val_pnls[client] = pnl_opt
    total_val_pnl = sum(val_pnls.values())
    return thresholds, total_val_pnl

# 5. Main function for optimal threshold
def optimal_threshold(filepath="trade_data.csv", tau=5, client_specific=True, n_points=100):
    df, le = load_data(filepath)
    df = add_microstructure_features(df)
    train, val, test = split_by_time_with_embargo(df, tau)
    train, val, test = add_client_adversity(train, val, test, tau)
    
    models, weights = train_ensemble(train, val, tau)
    
    X_val = val[FEATURE_COLS]
    prob_val = predict_ensemble(models, weights, X_val)
    X_test = test[FEATURE_COLS]
    prob_test = predict_ensemble(models, weights, X_test)
    
    clients = val["Name"].unique()
    
    if client_specific:
        thresholds, val_pnl = find_client_specific_thresholds(val, prob_val, tau, clients, n_points)
        test_pnl_per_client = {}
        total_test_pnl = 0.0
        for client in clients:
            mask = test["Name"] == client
            if mask.sum() == 0:
                test_pnl_per_client[client] = 0.0
                continue
            theta_c = thresholds[client]
            pnl_c = compute_pnl_for_threshold(test[mask], prob_test[mask], tau, theta_c)
            test_pnl_per_client[client] = pnl_c
            total_test_pnl += pnl_c
        return {
            "theta": thresholds,
            "validation_pnl": val_pnl,
            "test_pnl": total_test_pnl,
            "test_pnl_per_client": test_pnl_per_client
        }
    else:
        theta_opt, val_pnl, _, _ = find_optimal_threshold(val, prob_val, tau, n_points)
        test_pnl = compute_pnl_for_threshold(test, prob_test, tau, theta_opt)
        return {
            "theta": theta_opt,
            "validation_pnl": val_pnl,
            "test_pnl": test_pnl,
            "test_pnl_per_client": None
        }

# 6. Plot function
def plot_pnl_vs_theta(filepath="trade_data.csv", tau=5, client_specific=False, save_path="pnl_vs_theta.png"):
    df, le = load_data(filepath)
    df = add_microstructure_features(df)
    train, val, test = split_by_time_with_embargo(df, tau)
    train, val, test = add_client_adversity(train, val, test, tau)
    models, weights = train_ensemble(train, val, tau)
    X_val = val[FEATURE_COLS]
    prob_val = predict_ensemble(models, weights, X_val)
    
    if client_specific:
        clients = val["Name"].unique()
        thetas = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 6))
        for client in clients:
            mask = val["Name"] == client
            if mask.sum() == 0:
                continue
            pnls = [compute_pnl_for_threshold(val[mask], prob_val[mask], tau, t) for t in thetas]
            plt.plot(thetas, pnls, label=f"Client {client}")
        plt.xlabel("Threshold θ")
        plt.ylabel("Validation PnL")
        plt.title(f"PnL vs θ (client-specific) - τ={tau}")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    else:
        thetas = np.linspace(0, 1, 100)
        pnls = [compute_pnl_for_threshold(val, prob_val, tau, t) for t in thetas]
        plt.figure(figsize=(8, 5))
        plt.plot(thetas, pnls, 'b-', linewidth=2)
        plt.xlabel("Threshold θ")
        plt.ylabel("Validation PnL")
        plt.title(f"Global PnL vs θ - τ={tau}")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

# 7. Generate task4_results.csv
if __name__ == "__main__":
    taus = [5, 10, 15, 20, 25, 30]
    all_rows = []
    for tau in taus:
        print(f"\nProcessing tau={tau}...")
        res = optimal_threshold("trade_data.csv", tau, client_specific=True)
        thresholds = res["theta"]
        test_pnl_per_client = res["test_pnl_per_client"]
        for client, theta_val in thresholds.items():
            pnl = test_pnl_per_client.get(client, 0.0)
            all_rows.append({
                "client": client,
                "tau": tau,
                "theta_star": theta_val,
                "final_pnl": pnl
            })
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv("task4_results.csv", index=False)
    print("\nSaved task4_results.csv")
    # Example plots for tau=5
    plot_pnl_vs_theta("trade_data.csv", tau=5, client_specific=False, save_path="pnl_vs_theta_global_tau5.png")
    plot_pnl_vs_theta("trade_data.csv", tau=5, client_specific=True, save_path="pnl_vs_theta_client_tau5.png")
    print("Saved PnL vs θ plots.")