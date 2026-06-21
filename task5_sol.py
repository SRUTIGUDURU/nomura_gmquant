"""
Task 5: Dynamic Quoting Under Inventory Pressure
- Avellaneda‑Stoikov style with asymmetric skew.
- Online learning of fill parameters (γ, λ) from observed fills (only for positive spreads).
- Symmetric volume‑weighted adverse selection score per client:
    * client sells when mid rising (adverse)
    * client buys when mid falling (adverse)
  Updated only on filled trades.
- Regime detection (5‑day rolling window) adjusts base spread when φ spikes.
- Validation includes regime shifts in hidden penalty.
- Complies with spec: spreads >= 0.5 * sigma and <= 50 bps of mid.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class DynamicQuoter:
    def __init__(self,
                 initial_gamma: float = 5.0,
                 initial_lambda: float = 0.8,
                 learning_rate: float = 0.01,
                 inventory_p_gain: float = 0.3,
                 min_spread_mult: float = 0.5,
                 base_spread_mult: float = 2.0,
                 adv_multiplier: float = 1.5,
                 eta_power: float = 2.0,
                 adapt_interval: int = 50,
                 default_mid_price: float = 100.0,
                 ema_alpha: float = 0.2,
                 regime_sensitivity: float = 2.0,
                 max_spread_bps: float = 0.005):
        self.gamma = initial_gamma
        self.lambd = initial_lambda
        self.lr = learning_rate
        self.Kp = inventory_p_gain
        self.c_min = min_spread_mult
        self.base_spread = base_spread_mult
        self.adv_mult = adv_multiplier
        self.eta_power = eta_power
        self.adapt_interval = adapt_interval
        self.default_mid = default_mid_price
        self.ema_alpha = ema_alpha
        self.regime_sensitivity = regime_sensitivity
        self.max_spread_bps = max_spread_bps

        self.inv_ema = 0.0
        self.trade_counter = 0
        self.recent_spreads = deque(maxlen=100)
        self.recent_fills = deque(maxlen=100)
        self.phi_estimates = deque(maxlen=20)
        self.base_spread_adjust = 1.0
        self.client_adv = {}

    def update_adverse_score(self, client: str, side: int, volume: float, mid_rising: bool, was_filled: bool):
        if not was_filled:
            return
        if client not in self.client_adv:
            self.client_adv[client] = {'total_vol': 0.0, 'informed_vol': 0.0}
        self.client_adv[client]['total_vol'] += volume
        if (side == 1 and mid_rising) or (side == -1 and not mid_rising):
            self.client_adv[client]['informed_vol'] += volume

    def get_adverse_score(self, client: str) -> float:
        if client not in self.client_adv:
            return 0.0
        total = self.client_adv[client]['total_vol']
        if total == 0:
            return 0.0
        return self.client_adv[client]['informed_vol'] / total

    def update_regime(self, inventory_squared: float, sigma_daily: float, penalty_observed: float):
        if inventory_squared > 1e-6 and sigma_daily > 1e-6:
            phi_est = penalty_observed / (inventory_squared * sigma_daily)
            self.phi_estimates.append(phi_est)
            if len(self.phi_estimates) >= 5:
                mean_phi = np.mean(self.phi_estimates)
                std_phi = np.std(self.phi_estimates)
                if phi_est > mean_phi + self.regime_sensitivity * std_phi:
                    self.base_spread_adjust = min(self.base_spread_adjust * 1.5, 3.0)
                elif phi_est < mean_phi - self.regime_sensitivity * std_phi:
                    self.base_spread_adjust = max(self.base_spread_adjust / 1.2, 1.0)

    def quote(self, inventory: float, sigma: float, alpha: float, eta: float,
              client: str = None, adv_score_external: float = None, mid: float = None) -> Tuple[float, float]:
        if adv_score_external is not None:
            effective_alpha = max(alpha, adv_score_external)
        elif client is not None:
            effective_alpha = max(alpha, self.get_adverse_score(client))
        else:
            effective_alpha = alpha

        adjusted_base = self.base_spread * self.base_spread_adjust
        self.inv_ema = self.ema_alpha * inventory + (1 - self.ema_alpha) * self.inv_ema

        adv_factor = 1.0 + self.adv_mult * effective_alpha
        base_delta_sigma = max(adjusted_base * adv_factor, self.c_min)

        time_factor = 1.0 + eta ** self.eta_power
        skew_sigma = self.Kp * self.inv_ema * time_factor

        delta_bid_sigma = base_delta_sigma + skew_sigma
        delta_ask_sigma = base_delta_sigma - skew_sigma

        delta_bid = sigma * delta_bid_sigma
        delta_ask = sigma * delta_ask_sigma

        if mid is None:
            mid = self.default_mid
        delta_max_abs = self.max_spread_bps * mid
        delta_bid = np.clip(delta_bid, self.c_min * sigma, delta_max_abs)
        delta_ask = np.clip(delta_ask, self.c_min * sigma, delta_max_abs)

        return delta_bid, delta_ask

    def update_parameters(self, spread_norm: float, was_filled: bool):
        if spread_norm <= 0:
            return
        self.recent_spreads.append(spread_norm)
        self.recent_fills.append(1 if was_filled else 0)

        self.trade_counter += 1
        if self.trade_counter % self.adapt_interval != 0 or len(self.recent_spreads) < 20:
            return

        s = np.array(self.recent_spreads)
        y = np.array(self.recent_fills)
        for _ in range(5):
            p = self.lambd * np.exp(-self.gamma * s)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            grad_g = -self.lambd * s * np.exp(-self.gamma * s) * (y/p - (1-y)/(1-p))
            grad_l = np.exp(-self.gamma * s) * (y/p - (1-y)/(1-p))
            self.gamma += 0.1 * self.lr * np.mean(grad_g)
            self.lambd += 0.01 * self.lr * np.mean(grad_l)
            self.gamma = np.clip(self.gamma, 0.5, 50.0)
            self.lambd = np.clip(self.lambd, 0.1, 1.0)


_quoter = None

def quote(inventory: float, sigma: float, alpha: float, eta: float, **kwargs) -> Tuple[float, float]:
    global _quoter
    if _quoter is None:
        _quoter = DynamicQuoter()
    client = kwargs.get('client', None)
    adv_score = kwargs.get('adv_score', None)
    mid = kwargs.get('mid', None)
    return _quoter.quote(inventory, sigma, alpha, eta, client, adv_score, mid)


def validate_quote(data_path: str,
                   model_predict_fn,
                   output_path: str = "validation_results.csv") -> None:
    TRUE_GAMMA = 5.0
    TRUE_LAMBDA = 0.8
    PENALTY_BEFORE = 0.05
    PENALTY_AFTER = 0.25

    df = pd.read_csv(data_path)
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['time'])
    df.sort_values('datetime', inplace=True)
    df['Side'] = df['Side'].astype(int)
    df['date'] = df['datetime'].dt.date

    quoter = DynamicQuoter()
    inventory = 0.0
    daily_pnls = []
    unique_dates = sorted(df['date'].unique())

    for day_idx, date in enumerate(unique_dates):
        day_df = df[df['date'] == date].sort_values('datetime')
        day_pnl = 0.0
        mid_returns = deque(maxlen=20)

        penalty_coef = PENALTY_BEFORE if day_idx < 15 else PENALTY_AFTER

        for pos_idx, (_, row) in enumerate(day_df.iterrows()):
            mid = row['M0']

            if len(mid_returns) >= 5:
                sigma = np.std(mid_returns)
            else:
                sigma = 0.01
            sigma = max(sigma, 0.0001)

            alpha = model_predict_fn(
                client_name=row['Name'],
                side=row['Side'],
                volume=row['Volume'],
                trade_price=row['Trade Price'],
                m0=mid,
                spread=row['Spread'],
                tau=30
            )

            if pos_idx > 0:
                prev_mid = day_df.iloc[pos_idx - 1]['M0']
                mid_rising = (mid > prev_mid)
            else:
                mid_rising = False

            t_open = pd.Timestamp(date).replace(hour=9, minute=30)
            t_close = pd.Timestamp(date).replace(hour=16, minute=0)
            eta = (row['datetime'] - t_open) / (t_close - t_open)
            eta = np.clip(eta, 0.0, 1.0)

            delta_b, delta_a = quoter.quote(inventory, sigma, alpha, eta,
                                            client=row['Name'], mid=mid)

            if row['Side'] == 1:
                spread_used = delta_b
                filled_price = mid - delta_b
            else:
                spread_used = delta_a
                filled_price = mid + delta_a

            norm_spread = spread_used / (sigma + 1e-8)
            fill_prob = TRUE_LAMBDA * np.exp(-TRUE_GAMMA * norm_spread)
            fill_prob = np.clip(fill_prob, 0.0, 1.0)
            was_filled = np.random.random() < fill_prob

            if was_filled:
                quoter.update_adverse_score(row['Name'], row['Side'], row['Volume'],
                                            mid_rising, was_filled=True)

            if norm_spread > 0:
                quoter.update_parameters(norm_spread, was_filled)

            if was_filled:
                fill_volume = row['Volume']
                inventory += row['Side'] * fill_volume
                mid_cols = [f'M{t}' for t in [5, 10, 15, 20, 25, 30]]
                avg_future_mid = row[mid_cols].mean()
                trade_pnl = row['Side'] * fill_volume * (avg_future_mid - filled_price)
                day_pnl += trade_pnl

            if len(mid_returns) > 0:
                last_mid = day_df.iloc[pos_idx - 1]['M0'] if pos_idx > 0 else mid
                mid_returns.append(mid / last_mid - 1)
            else:
                mid_returns.append(0.0)

        avg_daily_vol = np.std(mid_returns) if len(mid_returns) > 0 else 0.01
        penalty = penalty_coef * (inventory ** 2) * avg_daily_vol
        day_pnl -= penalty
        daily_pnls.append(day_pnl)

        quoter.update_regime(inventory ** 2, avg_daily_vol, penalty)
        inventory = 0.0

    daily_pnls = pd.Series(daily_pnls)
    total_pnl = daily_pnls.sum()
    if daily_pnls.std() > 0:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    cum = daily_pnls.cumsum()
    mdd = (cum - cum.cummax()).min()

    summary = pd.DataFrame({
        'metric': ['total_pnl', 'sharpe', 'max_drawdown'],
        'value': [total_pnl, sharpe, mdd]
    })
    summary.to_csv(output_path, index=False)
    print(f"Validation complete. Results saved to {output_path}")
    print(f"Total PnL: {total_pnl:.2f}, Sharpe: {sharpe:.3f}, Max DD: {mdd:.2f}")


if __name__ == "__main__":
    print("Task 5 module ready. Use quote() and validate_quote() as per contest spec.")