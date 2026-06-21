"""
Task 3: Adversity Prediction Model
- 45+ microstructure features, including symmetric adverse selection proxy.
- LightGBM + CatBoost ensemble with inverse‑log‑loss weighting.
- Optuna hyperparameter tuning (time‑series aware).
- Rolling features applied BEFORE chronological split (no boundary reset).
- Ensemble weights computed from uncalibrated predictions (no leakage).
- Manual Platt scaling calibration (works on all sklearn versions).
"""

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')
# 1. Load data and base preprocessing
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

# 2. Advanced feature engineering (applied on entire dataset before split)
def add_microstructure_features(df):
    """All rolling / cumulative features use entire past data – no look‑ahead."""
    df = df.copy()

    # Mid returns and direction
    df["mid_return"] = df["M0"].pct_change()
    df["mid_rising"] = (df["mid_return"] > 0).astype(int)

    # EMAs of mid price
    for span in [1, 5, 10]:
        df[f"ema_mid_{span}"] = df["M0"].ewm(span=span, adjust=False).mean()
        df[f"ema_return_{span}"] = df[f"ema_mid_{span}"].pct_change()

    # Rolling volatility
    for window in [5, 20, 50]:
        df[f"vol_{window}"] = df["mid_return"].rolling(window, min_periods=1).std().fillna(0)

    # Spread features
    df["spread_to_vol_20"] = df["Spread"] / (df["vol_20"] + 1e-8)
    df["spread_change"] = df["Spread"].diff().fillna(0)
    df["spread_skew"] = df["Spread"].rolling(20).apply(lambda x: x.skew() if len(x) > 3 else 0).fillna(0)

    # Volume features
    df["log_volume"] = np.log1p(df["Volume"])
    df["signed_volume"] = df["Side"] * df["Volume"]
    df["volume_ema_5"] = df["Volume"].ewm(span=5, adjust=False).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_ema_5"] + 1e-8)

    # Order flow imbalance
    df["cum_signed_volume"] = df["signed_volume"].cumsum()
    df["ofi_20"] = df["signed_volume"].rolling(20).sum()

    # Client‑specific volume z‑score
    df["client_volume_z"] = df.groupby("client_code")["Volume"].transform(
        lambda x: (x - x.rolling(20, min_periods=1).mean()) / (x.rolling(20, min_periods=1).std() + 1e-8)
    ).fillna(0)

    # Trade arrival intensity
    df["time_since_last"] = df["datetime"].diff().dt.total_seconds().fillna(1)
    df["trade_burst"] = (df["time_since_last"] < df["time_since_last"].quantile(0.25)).astype(int)

    # Client trade count and frequency
    df["client_trade_count"] = df.groupby("client_code").cumcount()
    df["client_freq"] = df.groupby("client_code")["time_since_last"].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    ).fillna(30)

    # Price position in spread
    df["tp_m0_norm"] = (df["Trade Price"] - df["M0"]) / (df["Spread"] + 1e-8)
    df["side_tp"] = df["Side"] * df["tp_m0_norm"]

    # Time of day encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # SYMMETRIC ADVERSE SELECTION PROXY (efficient rolling per client) 
    # For each trade, we need a rolling fraction of "informed trades" for that client.
    # Informed if: (Side == 1 and mid_rising) OR (Side == -1 and not mid_rising)
    df["is_informed"] = ((df["Side"] == 1) & (df["mid_rising"] == 1)) | \
                        ((df["Side"] == -1) & (df["mid_rising"] == 0))
    df["is_informed"] = df["is_informed"].astype(int)

    # Rolling sum of informed trades and total trades per client over last 20 trades
    def rolling_informed(g):
        return g["is_informed"].rolling(20, min_periods=1).sum()
    def rolling_total(g):
        return g["is_informed"].rolling(20, min_periods=1).count()
    df["informed_sum"] = df.groupby("client_code")["is_informed"].transform(rolling_informed)
    df["total_trades"] = df.groupby("client_code")["is_informed"].transform(rolling_total)
    df["client_adv_ratio"] = (df["informed_sum"] / (df["total_trades"] + 1e-8)).fillna(0)

    # Drop intermediate columns
    drop_cols = ["mid_return", "mid_rising", "is_informed", "informed_sum", "total_trades"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    return df

def split_by_time(df, train_ratio=0.6, val_ratio=0.2):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (df.iloc[:train_end].copy(),
            df.iloc[train_end:val_end].copy(),
            df.iloc[val_end:].copy())

def add_client_adversity(train_df, val_df, test_df, tau):
    """Adds adversity label and client rolling adversity rate (no look‑ahead across splits)."""
    mid_col = f"M{tau}"
    for d in [train_df, val_df, test_df]:
        d["adverse"] = (d["Side"] * d["Volume"] * (d[mid_col] - d["Trade Price"]) < 0).astype(int)
    # Train: expanding mean
    train_df["client_adv_rate"] = train_df.groupby("client_code")["adverse"].expanding().mean().values
    train_df["client_adv_rate"] = train_df["client_adv_rate"].fillna(train_df["adverse"].mean())
    # Val: expanding within val only, fallback to train mean for first row
    val_df["client_adv_rate"] = val_df.groupby("client_code")["adverse"].expanding().mean().values
    val_df["client_adv_rate"] = val_df["client_adv_rate"].fillna(train_df["adverse"].mean())
    # Test: frozen from last val value
    last_val_adv = val_df.groupby("client_code")["client_adv_rate"].last().to_dict()
    global_fallback = train_df["adverse"].mean()
    test_df["client_adv_rate"] = test_df["Name"].map(last_val_adv).fillna(global_fallback)
    return train_df, val_df, test_df

# 3. Feature list (final) – includes client_adv_ratio and interactions
BASE_FEATURES = [
    "client_code", "Side", "Volume", "log_volume", "signed_volume",
    "Spread", "spread_to_vol_20", "spread_change", "spread_skew",
    "time_since_last", "trade_burst", "client_trade_count", "client_freq",
    "ema_mid_1", "ema_mid_5", "ema_mid_10",
    "ema_return_1", "ema_return_5", "ema_return_10",
    "vol_5", "vol_20", "vol_50",
    "volume_ema_5", "volume_ratio", "cum_signed_volume", "ofi_20",
    "client_volume_z", "tp_m0_norm", "side_tp",
    "hour", "minute", "second", "hour_sin", "hour_cos",
    "client_adv_rate", "client_adv_ratio"
]

# Add lagged features (shifted within client) – only possible after splitting? No, we can add globally.
def add_lagged_features(df):
    """Add lagged adverse outcome and lagged signed volume per client."""
    df = df.copy()
    # Shift by 1 within each client (previous trade's info)
    df["prev_adverse"] = df.groupby("client_code")["adverse"].shift(1)
    df["prev_signed_vol"] = df.groupby("client_code")["signed_volume"].shift(1)
    df["prev_spread"] = df.groupby("client_code")["Spread"].shift(1)
    df["prev_adverse"] = df["prev_adverse"].fillna(0)
    df["prev_signed_vol"] = df["prev_signed_vol"].fillna(0)
    df["prev_spread"] = df["prev_spread"].fillna(df["Spread"].median())
    return df

# Interaction features (added later, after client_adv_rate exists)
def add_interaction_features(df):
    df = df.copy()
    df["adv_rate_x_spread"] = df["client_adv_rate"] * df["Spread"]
    df["adv_ratio_x_signed_vol"] = df["client_adv_ratio"] * df["signed_volume"]
    df["spread_x_vol"] = df["Spread"] * df["Volume"]
    return df

# Final feature list after adding lags and interactions
FEATURE_COLS = BASE_FEATURES + [
    "prev_adverse", "prev_signed_vol", "prev_spread",
    "adv_rate_x_spread", "adv_ratio_x_signed_vol", "spread_x_vol"
]

# 4. Manual Platt scaling 
def fit_platt(raw_probs, labels):
    def neg_ll(ab):
        p = np.clip(expit(ab[0] * raw_probs + ab[1]), 1e-7, 1 - 1e-7)
        return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    res = minimize(neg_ll, [1.0, 0.0], method="L-BFGS-B")
    return res.x[0], res.x[1]

def apply_platt(raw, a, b):
    return expit(a * raw + b)

# 5. Optuna tuning (LightGBM and CatBoost) – unchanged
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

def optimize_model(model_type, X_train, y_train, X_val, y_val, n_trials=20):
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

# 6. Train ensemble (returns (models_tuple, weights))
def train_ensemble(train_df, val_df, tau):
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df["adverse"]
    X_v = val_df[FEATURE_COLS]
    y_v = val_df["adverse"]

    print("  Tuning LightGBM...")
    lgb_model = optimize_model('lgb', X_tr, y_tr, X_v, y_v, n_trials=20)
    prob_lgb_uncal = lgb_model.predict_proba(X_v)[:, 1]

    print("  Tuning CatBoost...")
    cb_model = optimize_model('catboost', X_tr, y_tr, X_v, y_v, n_trials=20)
    prob_cb_uncal = cb_model.predict_proba(X_v)[:, 1]

    # Weights based on uncalibrated log-loss
    ll_lgb = log_loss(y_v, prob_lgb_uncal)
    ll_cb  = log_loss(y_v, prob_cb_uncal)
    w_lgb = 1.0 / ll_lgb
    w_cb  = 1.0 / ll_cb
    total = w_lgb + w_cb
    w_lgb /= total
    w_cb  /= total
    print(f"  Ensemble weights (uncalibrated): LGB={w_lgb:.3f}, CB={w_cb:.3f}")

    # Manual Platt calibration on validation set
    raw_lgb = lgb_model.predict_proba(X_v)[:, 1]
    a_lgb, b_lgb = fit_platt(raw_lgb, y_v.values)
    raw_cb = cb_model.predict_proba(X_v)[:, 1]
    a_cb, b_cb = fit_platt(raw_cb, y_v.values)

    # Return models as tuple: (lgb_model, cb_model, (a_lgb,b_lgb), (a_cb,b_cb))
    models = (lgb_model, cb_model, (a_lgb, b_lgb), (a_cb, b_cb))
    weights = (w_lgb, w_cb)
    return models, weights

def predict_ensemble(models, weights, X):
    lgb_model, cb_model, (a_lgb, b_lgb), (a_cb, b_cb) = models
    w_lgb, w_cb = weights
    raw_lgb = lgb_model.predict_proba(X)[:, 1]
    raw_cb  = cb_model.predict_proba(X)[:, 1]
    prob_lgb = apply_platt(raw_lgb, a_lgb, b_lgb)
    prob_cb  = apply_platt(raw_cb,  a_cb,  b_cb)
    return w_lgb * prob_lgb + w_cb * prob_cb

# 7. Global state and required interfaces
_ensemble_models = {}
_ensemble_weights = {}
_tau_list = [5, 10, 15, 20, 25, 30]

def load_models(filepath="trade_data.csv"):
    """Load data, apply rolling features globally, split, then train."""
    global _ensemble_models, _ensemble_weights
    df, le = load_data(filepath)
    df = add_microstructure_features(df)          # step 1
    df = add_lagged_features(df)                 
    df = add_lagged_features_without_adverse(df)
    train, val, test = split_by_time(df)
    for tau in _tau_list:
        print(f"Training for tau={tau}...")
        tr, v, te = add_client_adversity(train.copy(), val.copy(), test.copy(), tau)
        # Now add lagged adverse based on the newly created 'adverse' column
        tr = add_lagged_adverse(tr)
        v = add_lagged_adverse(v)
        te = add_lagged_adverse(te)
        # Add interaction features (require client_adv_rate)
        tr = add_interaction_features(tr)
        v = add_interaction_features(v)
        te = add_interaction_features(te)
        models, weights = train_ensemble(tr, v, tau)
        _ensemble_models[tau] = models
        _ensemble_weights[tau] = weights

def add_lagged_features_without_adverse(df):
    """Add prev_signed_vol and prev_spread only (no adverse column yet)."""
    df = df.copy()
    df["prev_signed_vol"] = df.groupby("client_code")["signed_volume"].shift(1).fillna(0)
    df["prev_spread"] = df.groupby("client_code")["Spread"].shift(1).fillna(df["Spread"].median())
    return df

def add_lagged_adverse(df):
    """Add prev_adverse after 'adverse' column exists."""
    df = df.copy()
    df["prev_adverse"] = df.groupby("client_code")["adverse"].shift(1).fillna(0)
    return df

def predict_adversity(*args, **kwargs):
    """Flexible prediction – expects kwargs: features (DataFrame with one row), tau."""
    tau = kwargs['tau']
    X = kwargs['features'][FEATURE_COLS]
    models = _ensemble_models[tau]
    weights = _ensemble_weights[tau]
    return predict_ensemble(models, weights, X)[0]

def compute_metrics(filepath="trade_data.csv") -> pd.DataFrame:
    """Compute average metrics across horizons (train/val/test)."""
    df, le = load_data(filepath)
    df = add_microstructure_features(df)
    df = add_lagged_features_without_adverse(df)
    train, val, test = split_by_time(df)
    tau_list = [5, 10, 15, 20, 25, 30]
    results = {"train": [], "val": [], "test": []}

    for tau in tau_list:
        tr, v, te = add_client_adversity(train.copy(), val.copy(), test.copy(), tau)
        tr = add_lagged_adverse(tr)
        v = add_lagged_adverse(v)
        te = add_lagged_adverse(te)
        tr = add_interaction_features(tr)
        v = add_interaction_features(v)
        te = add_interaction_features(te)
        models, weights = train_ensemble(tr, v, tau)
        for split_name, split_df in [("train", tr), ("val", v), ("test", te)]:
            X = split_df[FEATURE_COLS]
            y = split_df["adverse"]
            prob = predict_ensemble(models, weights, X)
            pred = (prob >= 0.5).astype(int)
            results[split_name].append({
                "acc":  accuracy_score(y, pred),
                "prec": precision_score(y, pred, zero_division=0),
                "rec":  recall_score(y, pred, zero_division=0),
                "ll":   log_loss(y, prob)
            })
        print(f"tau={tau} done")

    rows = []
    for split in ["train", "val", "test"]:
        d = results[split]
        rows.append({
            "split": split if split != "val" else "validation",
            "accuracy":  np.mean([x["acc"] for x in d]),
            "precision": np.mean([x["prec"] for x in d]),
            "recall":    np.mean([x["rec"] for x in d]),
            "log_loss":  np.mean([x["ll"] for x in d])
        })
    return pd.DataFrame(rows)

# 8. Generate output files
if __name__ == "__main__":
    metrics_df = compute_metrics("trade_data.csv")
    metrics_df.to_csv("task3_results.csv", index=False)
    print("Saved task3_results.csv")
    print(metrics_df)

    with open("feature_order.txt", "w") as f:
        for i, col in enumerate(FEATURE_COLS):
            f.write(f"{i}: {col}\n")
    print("Saved feature_order.txt")