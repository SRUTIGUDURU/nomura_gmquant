# Nomura GM Quant Challenge — Solutions

Solutions to the Nomura GM Quant Challenge: five Python tasks on market-making, client adverse selection, and quoting under inventory risk, plus a standalone C++ interest-rate curve engine with exact analytical risk.

## Overview

The challenge centers on a liquidity provider (LP) trading against a set of clients, using a trade-level dataset (`trade_data.csv`) with columns including `Name` (client), `Side`, `Volume`, `Trade Price`, `Spread`, and forward mid-prices `M0`–`M30` at multiple horizons. The five tasks build on each other: from measuring how often the LP gets picked off (Task 1), to pricing that risk into spreads (Task 2), to predicting it ahead of time (Task 3), to acting on that prediction (Task 4), to running a full dynamic quoting engine (Task 5). `sol2.cpp` is a separate problem on bootstrapping and risk-managing an interest rate curve. Each task's `.py` file is paired with a `_docu.pdf` write-up containing the methodology, results tables, and discussion; full numbers are reproduced below.

## Repository structure

| File | Description |
|---|---|
| `task1_sol.py` | Computes the % of trades adverse to the LP, by client and horizon |
| `task1_docu.pdf` | Write-up: methodology, full results table, and per-client trend analysis (5 pages) |
| `task2_sol.py` | Expected PnL per client and minimum half-spread δ* for non-negative expected PnL |
| `task2_docu.pdf` | Write-up: derivation of δ*, client profitability classification, and discussion |
| `task3_sol.py` | LightGBM + CatBoost ensemble predicting adverse-selection probability per trade |
| `task3_docu.pdf` | Write-up: feature list (52 features), model design, and train/val/test metrics |
| `task4_sol.py` | Optimal externalization threshold — when to lay off a trade vs. internalize it |
| `task4_docu.pdf` | Write-up: threshold search methodology, global-vs-client-specific comparison, full results tables, PnL-vs-θ plots (2 pages) |
| `task5_sol.py` | Dynamic quoting engine (Avellaneda–Stoikov style) under inventory pressure |
| `task5_docu.pdf` | Write-up: quoting formula derivation, online learning, regime detection, validation PnL/Sharpe/drawdown |
| `sol2.cpp` | Interest rate curve bootstrapping engine with exact analytical risk |
| `sol2_document.docx` | Deep technical write-up: math derivations, design rationale, complexity analysis |

## Results

### Task 1 — Adversity Profile

Percentage of trades that move against the LP, by client and closing horizon τ (seconds):

| Client | τ=5 | τ=10 | τ=15 | τ=20 | τ=25 | τ=30 | Pattern |
|---|---|---|---|---|---|---|---|
| A | 42.3% | 45.1% | 48.6% | 52.0% | 54.2% | 55.8% | Rising — adverse selection |
| B | 38.7% | 37.2% | 35.9% | 34.1% | 33.0% | 32.4% | Falling — mean-reverting, LP-friendly |
| C | 51.2% | 50.5% | 49.8% | 49.1% | 48.5% | 48.0% | Near coin-flip |
| D | 29.5% | 31.8% | 34.0% | 36.2% | 38.5% | 40.1% | Rising — delayed adverse selection |
| E | 63.4% | 62.1% | 60.5% | 58.9% | 57.3% | 55.9% | High but falling — short-lived spike |
| F | 44.0% | 44.2% | 44.1% | 44.3% | 44.2% | 44.0% | Flat — noise/hedging flow |

*The write-up notes these values are illustrative of the patterns observed and may differ slightly from a fresh run against the undisclosed dataset.*

### Task 2 — Client Profitability and Spread Recommendation

| Client | Expected aggregate PnL | Classification | Minimum half-spread δ* |
|---|---|---|---|
| A | −19.4 | Costly | 0.0450 |
| B | +5.2 | Profitable | 0.0000 |
| C | +0.3 | Marginally profitable | 0.0000 |
| D | −14.9 | Costly | 0.0382 |
| E | −19.2 | Costly | 0.0478 |
| F | 0.0 | Breakeven | 0.0000 |

Adversity frequency alone doesn't determine δ*: Client B has 32–39% adversity but stays profitable because losses are small relative to gains, while Client E's adversity falls over time yet still needs the largest spread because its losses are severe in magnitude.

### Task 3 — Adversity Prediction Model

Averaged across all six horizons (52-feature LightGBM + CatBoost ensemble, Platt-scaled):

| Split | Accuracy | Precision | Recall | Log-loss |
|---|---|---|---|---|
| Train | 0.782 | 0.769 | 0.745 | 0.421 |
| Validation | 0.741 | 0.728 | 0.701 | 0.489 |
| Test | 0.738 | 0.721 | 0.695 | 0.495 |

Log-loss of 0.489 against a 0.693 coin-flip baseline, with test metrics close to validation, indicating reasonable calibration and limited overfitting from the chronological/embargoed split.

### Task 4 — Optimal Externalization Threshold

Client-specific thresholds θ* by horizon:

| Client | τ=5 | τ=10 | τ=15 | τ=20 | τ=25 | τ=30 |
|---|---|---|---|---|---|---|
| A | 0.32 | 0.35 | 0.38 | 0.41 | 0.44 | 0.46 |
| B | 0.71 | 0.68 | 0.65 | 0.62 | 0.60 | 0.58 |
| C | 0.52 | 0.51 | 0.50 | 0.50 | 0.49 | 0.49 |
| D | 0.41 | 0.43 | 0.45 | 0.47 | 0.48 | 0.50 |
| E | 0.24 | 0.27 | 0.30 | 0.33 | 0.36 | 0.39 |
| F | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

Costly clients (A, D, E) get low thresholds — externalize early to limit losses; the profitable client (B) gets a high threshold — externalize rarely. At τ=5, client-specific thresholds lift validation PnL from 12,340 (global θ≈0.42) to 18,760 (+52%), and test PnL from 11,210 to 17,530; the average improvement across all horizons is +45%. Total test PnL declines from 17,530 (τ=5) to 14,050 (τ=30) as predictive power decays at longer horizons.

### Task 5 — Dynamic Quoting Under Inventory Pressure

Asymmetric Avellaneda–Stoikov quoting with online-learned fill parameters (γ, λ) and a regime detector that widens spreads when the inventory-penalty coefficient spikes. Validated on the full dataset with a hidden penalty shock (φ: 0.05 → 0.25 after day 15):

- Total net PnL: **24,870**
- Annualized Sharpe: **1.94**
- Max drawdown: **−1,230**

The regime detector correctly widens spreads after the penalty shock without any retraining, and negative spreads near the close are used to actively flatten inventory.

### Curve Engine (`sol2.cpp`)

Bootstraps discount curves from cash and swap instruments via Brent's method, under two interpolation schemes — log-linear and averaged-quadratic on log discount factors — chosen specifically because each admits **exact, local interpolation weights**, which is what makes closed-form risk possible. The accompanying write-up is explicit that the problem statement *forbids* bump-and-revalue (finite-difference) risk, so all rate sensitivities are propagated analytically: interpolation weights map any query date to the relevant node log-discount-factors, and for swap nodes, derivatives are obtained by implicit differentiation through a lower-triangular Jacobian built up node-by-node during the bootstrap. The design is intentionally a strategy pattern (`std::function` interpolators) over a polymorphic instrument hierarchy (`IInstrument` → `CashInstrument`, `SwapInstrument`), so new interpolation schemes or instrument types (FRAs, OIS) can be added without touching the bootstrap, pricing, or risk code.

## Running the code

### Python tasks (1–4 require a dependencies install; Task 5 is dependency-light)

```bash
pip install pandas numpy scipy scikit-learn lightgbm catboost optuna matplotlib
```

Each `taskN_sol.py` expects `trade_data.csv` in the working directory and is runnable standalone:

```bash
python task1_sol.py   # -> task1_results.csv
python task2_sol.py   # -> task2_results.csv
python task3_sol.py   # -> task3_results.csv, feature_order.txt
python task4_sol.py   # -> task4_results.csv, pnl_vs_theta_*.png
```

Task 5 exposes `quote()` and `validate_quote()` for use against a grading harness; run directly to confirm the module loads:

```bash
python task5_sol.py
```

### C++ curve engine

```bash
g++ -std=c++17 -O2 sol2.cpp -o sol2
./sol2   # reads Input.csv, writes Output.csv
```

## Notes

- Tasks 3 and 4 share nearly identical feature engineering; Task 4 adds an embargo period at the train/validation boundary so that a trade's training label (which depends on a future mid-price `Mτ`) can't leak past trades into the validation window.
- All "adverse" / "informed" trade labels are defined symmetrically: a client sells into a rising mid or buys into a falling mid.
- This is contest submission code, not production-hardened — paths to `trade_data.csv` / `Input.csv` are relative and assume the file sits alongside the script.
