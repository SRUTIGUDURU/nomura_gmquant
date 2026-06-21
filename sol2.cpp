
// Nomura Interest Rate Curve Engine
// Includes Brent root-finding for swap calibration and exact analytical
// risk propagation for both linear and averaged quadratic interpolation.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

// 1. Day count conventions
inline double days_to_years(double days) { return days / 360.0; }

double tenor_to_years(const std::string& s) {
    if (s.size() < 2) throw std::invalid_argument("Invalid tenor: " + s);
    char unit = static_cast<char>(std::toupper(s.back()));
    double n = std::stod(s.substr(0, s.size() - 1));
    switch (unit) {
        case 'D': return n / 360.0;
        case 'W': return n * 7.0 / 360.0;
        case 'M': return n * 30.0 / 360.0;
        case 'Y': return n;
        default: throw std::invalid_argument("Unknown unit: " + s);
    }
}

double freq_to_years(const std::string& f) {
    std::string s = f;
    if (s.back() == 'm' || s.back() == 'M') s.pop_back();
    return std::stod(s) / 12.0;
}

// 2. Curve nodes and interpolation strategies
struct CurveNode {
    double t;
    double df;
    double ln_df;
};

using Interpolator = std::function<double(const std::vector<CurveNode>&, double)>;

static std::vector<CurveNode>::const_iterator find_upper(const std::vector<CurveNode>& nodes, double t) {
    return std::upper_bound(nodes.begin(), nodes.end(), t,
        [](double v, const CurveNode& n) { return v < n.t; });
}

double interp_linear(const std::vector<CurveNode>& nodes, double t) {
    if (t <= 0.0) return 1.0;
    if (t >= nodes.back().t) return nodes.back().df;
    auto it = find_upper(nodes, t);
    auto prev = std::prev(it);
    double alpha = (t - prev->t) / (it->t - prev->t);
    return std::exp(prev->ln_df + alpha * (it->ln_df - prev->ln_df));
}

double interp_averaged_quadratic(const std::vector<CurveNode>& nodes, double t) {
    if (t <= 0.0) return 1.0;
    if (t >= nodes.back().t) return nodes.back().df;
    auto it = find_upper(nodes, t);
    size_t idx = std::distance(nodes.begin(), it);
    if (idx == 1) return interp_linear(nodes, t);
    double t1 = nodes[idx-1].t, y1 = nodes[idx-1].ln_df;
    double t2 = nodes[idx].t,   y2 = nodes[idx].ln_df;
    double t0 = nodes[idx-2].t, y0 = nodes[idx-2].ln_df;
    auto lagrange = [](double x, double a, double ya, double b, double yb, double c, double yc) {
        double L0 = ((x-b)*(x-c))/((a-b)*(a-c));
        double L1 = ((x-a)*(x-c))/((b-a)*(b-c));
        double L2 = ((x-a)*(x-b))/((c-a)*(c-b));
        return ya*L0 + yb*L1 + yc*L2;
    };
    double Q_left = lagrange(t, t0, y0, t1, y1, t2, y2);
    if (idx + 1 >= nodes.size()) return std::exp(Q_left);
    double t3 = nodes[idx+1].t, y3 = nodes[idx+1].ln_df;
    double Q_right = lagrange(t, t1, y1, t2, y2, t3, y3);
    double wL = (t2 - t) / (t2 - t1);
    double wR = (t - t1) / (t2 - t1);
    return std::exp(wL * Q_left + wR * Q_right);
}

// 3. Discount curve (supports deep copy for Brent)
struct DiscountCurve {
    std::vector<CurveNode> nodes;
    Interpolator interpolate;
    DiscountCurve(Interpolator interp) : interpolate(std::move(interp)) {
        nodes.push_back({0.0, 1.0, 0.0});
    }
    DiscountCurve(const DiscountCurve&) = default;
    DiscountCurve& operator=(const DiscountCurve&) = default;

    double getDF(double t) const { return interpolate(nodes, t); }
    void addNode(double t, double df) {
        if (t <= nodes.back().t)
            throw std::runtime_error("Non-increasing maturity in curve construction");
        nodes.push_back({t, df, std::log(df)});
    }
    DiscountCurve withTrialNode(double t, double df) const {
        DiscountCurve copy = *this;
        copy.addNode(t, df);
        return copy;
    }
};

// 4. Instrument interface and implementations
class IInstrument {
public:
    virtual ~IInstrument() = default;
    virtual double maturity() const = 0;
    virtual double calibrate(const DiscountCurve& curve, double market_rate) const = 0;
    virtual bool isCash() const { return false; }
};

class CashInstrument : public IInstrument {
    double maturity_;
public:
    explicit CashInstrument(double t) : maturity_(t) {}
    double maturity() const override { return maturity_; }
    double calibrate(const DiscountCurve&, double rate) const override {
        return 1.0 / (1.0 + rate * maturity_);
    }
    bool isCash() const override { return true; }
};

class SwapInstrument : public IInstrument {
    double maturity_;
    double step_;
public:
    SwapInstrument(double t, double freq_step) : maturity_(t), step_(freq_step) {
        if (step_ <= 0.0) throw std::invalid_argument("Swap step must be positive");
    }
    double maturity() const override { return maturity_; }

    double calibrate(const DiscountCurve& curve, double swap_rate) const override {
        int n = static_cast<int>(std::round(maturity_ / step_));
        double a = 1e-6, b = 2.0;
        auto f = [this, &curve, swap_rate, n](double df_trial) {
            DiscountCurve trial = curve.withTrialNode(this->maturity_, df_trial);
            double annuity = 0.0;
            for (int k = 1; k < n; ++k) {
                annuity += trial.getDF(k * step_) * step_;
            }
            annuity += df_trial * (maturity_ - (n - 1) * step_);
            return (1.0 - df_trial) - swap_rate * annuity;
        };

        double fa = f(a), fb = f(b);
        if (fa * fb > 0) throw std::runtime_error("Root not bracketed in swap calibration");

        double c = a, fc = fa, d = 0.0, e = 0.0;
        for (int iter = 0; iter < 100; ++iter) {
            if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
            if (std::abs(fc) < std::abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
            double tol = 1e-12, m = 0.5 * (c - b);
            if (std::abs(m) <= tol || fb == 0.0) return b;
            if (std::abs(e) >= tol && std::abs(fa) > std::abs(fb)) {
                double s = fb / fa, p, q;
                if (a == c) { p = 2.0 * m * s; q = 1.0 - s; }
                else {
                    q = fa / fc;
                    double r = fb / fc;
                    p = s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0));
                    q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                }
                if (p > 0.0) q = -q;
                p = std::abs(p);
                double min1 = 3.0 * m * q - std::abs(tol * q);
                double min2 = std::abs(e * q);
                if (2.0 * p < (min1 < min2 ? min1 : min2)) { e = d; d = p / q; }
                else { d = m; e = d; }
            } else { d = m; e = d; }
            a = b; fa = fb;
            if (std::abs(d) > tol) b += d; else b += (m > 0 ? tol : -tol);
            fb = f(b);
        }
        return b;
    }
};

void bootstrap_curve(DiscountCurve& curve,
                     const std::vector<std::unique_ptr<IInstrument>>& instruments,
                     const std::vector<double>& market_rates) {
    for (size_t i = 0; i < instruments.size(); ++i) {
        double df = instruments[i]->calibrate(curve, market_rates[i]);
        curve.addNode(instruments[i]->maturity(), df);
    }
}

// 5. Interpolation weights for risk (exact for linear and AQ)
static std::vector<double> lnDf_weights(const DiscountCurve& curve, double t) {
    const auto& nodes = curve.nodes;
    size_t n = nodes.size();
    std::vector<double> w(n, 0.0);
    if (t <= 0.0) { w[0] = 1.0; return w; }
    if (t >= nodes.back().t) { w.back() = 1.0; return w; }

    auto it = find_upper(nodes, t);
    size_t idx = std::distance(nodes.begin(), it);
    double t0 = nodes[idx-1].t, t1 = nodes[idx].t;
    double alpha = (t - t0) / (t1 - t0);

    if (curve.interpolate.target<decltype(&interp_linear)>() || idx == 1) {
        w[idx-1] = 1.0 - alpha;
        w[idx]   = alpha;
        return w;
    }

    auto lagrange_basis = [](double x, double a, double b, double c) -> std::array<double,3> {
        return {((x-b)*(x-c))/((a-b)*(a-c)),
                ((x-a)*(x-c))/((b-a)*(b-c)),
                ((x-a)*(x-b))/((c-a)*(c-b))};
    };

    // Left quadratic
    auto left = lagrange_basis(t, nodes[idx-2].t, nodes[idx-1].t, nodes[idx].t);
    double wL = (t1 - t) / (t1 - t0);
    w[idx-2] += wL * left[0];
    w[idx-1] += wL * left[1];
    w[idx]   += wL * left[2];

    if (idx + 1 >= n) return w;   // right boundary

    // Right quadratic
    auto right = lagrange_basis(t, nodes[idx-1].t, nodes[idx].t, nodes[idx+1].t);
    double wR = (t - t0) / (t1 - t0);
    w[idx-1] += wR * right[0];
    w[idx]   += wR * right[1];
    w[idx+1] += wR * right[2];

    return w;
}

// 6. Cash curve risk (exact)
std::vector<double> cash_curve_risk(const DiscountCurve& curve,
                                    const std::vector<double>& cash_rates,
                                    double fixed_rate, double swap_maturity,
                                    double fixed_step) {
    size_t n = curve.nodes.size() - 1;
    std::vector<double> dPV_dLnDF(curve.nodes.size(), 0.0);

    // Fixed leg
    double t_prev = 0.0;
    int num_pay = static_cast<int>(std::round(swap_maturity / fixed_step));
    for (int k = 1; k <= num_pay; ++k) {
        double ti = (k == num_pay) ? swap_maturity : k * fixed_step;
        double df = curve.getDF(ti);
        double coeff = -fixed_rate * (ti - t_prev);
        auto w = lnDf_weights(curve, ti);
        for (size_t j = 0; j < w.size(); ++j)
            dPV_dLnDF[j] += coeff * df * w[j];
        t_prev = ti;
    }

    // Floating leg
    double dfT = curve.getDF(swap_maturity);
    auto wT = lnDf_weights(curve, swap_maturity);
    for (size_t j = 0; j < wT.size(); ++j)
        dPV_dLnDF[j] += -dfT * wT[j];

    // Convert to cash rate sensitivity
    std::vector<double> dPV_dRate(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double T = curve.nodes[i+1].t;
        double r = cash_rates[i];
        double dLnDF_dr = -T / (1.0 + r * T);
        dPV_dRate[i] = dPV_dLnDF[i+1] * dLnDF_dr;
    }
    return dPV_dRate;
}

// 7. Swap curve risk – exact implicit differentiation
std::vector<double> swap_curve_risk(const DiscountCurve& curve,
                                    const std::vector<double>& swap_rates,
                                    double fixed_rate, double swap_maturity,
                                    double fixed_step, double calib_step) {
    size_t n = curve.nodes.size() - 1;
    std::vector<std::vector<double>> J(n, std::vector<double>(n, 0.0));

    // Precompute auxiliary data (annuity, last_dcf) for each node
    struct Aux { double ann; double last_dcf; bool is_cash; };
    std::vector<Aux> aux(n);
    for (size_t i = 0; i < n; ++i) {
        double T = curve.nodes[i+1].t;
        bool is_cash = (T < 0.5);
        if (is_cash) {
            aux[i] = {0.0, T, true};
        } else {
            int num = static_cast<int>(std::round(T / calib_step));
            double ann = 0.0;
            for (int k = 1; k < num; ++k)
                ann += curve.getDF(k * calib_step) * calib_step;
            aux[i] = {ann, T - (num - 1) * calib_step, false};
        }
    }

    // Build lower‑triangular Jacobian
    for (size_t j = 0; j < n; ++j) {
        double T_j = curve.nodes[j+1].t;
        double df_j = curve.nodes[j+1].df;

        // Diagonal 
        if (aux[j].is_cash) {
            J[j][j] = -T_j * df_j * df_j;   // cash derivative
        } else {
            // dA/dDF_j (including own node via interpolation)
            double dA_dDF = aux[j].last_dcf;
            int num_j = static_cast<int>(std::round(T_j / calib_step));
            for (int k = 1; k < num_j; ++k) {
                double tk = k * calib_step;
                auto w = lnDf_weights(curve, tk);
                dA_dDF += w[j+1] * (curve.getDF(tk) / df_j) * calib_step;
            }
            double denom = 1.0 + swap_rates[j] * dA_dDF;
            J[j][j] = -(aux[j].ann + df_j * aux[j].last_dcf) / denom;
        }

        // Off‑diagonal (i > j)
        for (size_t i = j+1; i < n; ++i) {
            if (aux[i].is_cash) continue;   // cash nodes are independent

            double T_i = curve.nodes[i+1].t;
            int num_i = static_cast<int>(std::round(T_i / calib_step));
            double explicit_dA_dsj = 0.0;

            for (int k = 1; k < num_i; ++k) {
                double tk = k * calib_step;
                auto w = lnDf_weights(curve, tk);

                // Contribution from previously solved nodes (m < i)
                double dDF_tk_dsj = 0.0;
                for (size_t m = j; m < i; ++m) {
                    double w_m = w[m+1];
                    if (std::abs(w_m) < 1e-15) continue;
                    dDF_tk_dsj += w_m * (curve.getDF(tk) / curve.nodes[m+1].df) * J[m][j];
                }
                explicit_dA_dsj += dDF_tk_dsj * calib_step;
            }

            // Denominator uses the same dA/dDF form as diagonal but without own node contribution
            double dA_dDF_i = aux[i].last_dcf;
            for (int k = 1; k < num_i; ++k) {
                double tk = k * calib_step;
                auto w = lnDf_weights(curve, tk);
                dA_dDF_i += w[i+1] * (curve.getDF(tk) / curve.nodes[i+1].df) * calib_step;
            }
            double denom = 1.0 + swap_rates[i] * dA_dDF_i;

            J[i][j] = (-swap_rates[i] * explicit_dA_dsj) / denom;
        }
    }

    // Compute dPV/d(ln DF) at each node
    std::vector<double> dPV_dLnDF(curve.nodes.size(), 0.0);
    double t_prev = 0.0;
    int num_pay = static_cast<int>(std::round(swap_maturity / fixed_step));
    for (int k = 1; k <= num_pay; ++k) {
        double ti = (k == num_pay) ? swap_maturity : k * fixed_step;
        double df = curve.getDF(ti);
        double coeff = -fixed_rate * (ti - t_prev);
        auto w = lnDf_weights(curve, ti);
        for (size_t j = 0; j < w.size(); ++j)
            dPV_dLnDF[j] += coeff * df * w[j];
        t_prev = ti;
    }
    double dfT = curve.getDF(swap_maturity);
    auto wT = lnDf_weights(curve, swap_maturity);
    for (size_t j = 0; j < wT.size(); ++j)
        dPV_dLnDF[j] += -dfT * wT[j];

    //Chain rule to market rates
    std::vector<double> dPV_dRate(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        if (aux[j].is_cash) {
            double T = curve.nodes[j+1].t;
            double r = swap_rates[j];
            double dLnDF_dr = -T / (1.0 + r * T);
            dPV_dRate[j] = dPV_dLnDF[j+1] * dLnDF_dr;
        } else {
            for (size_t i = j; i < n; ++i) {
                dPV_dRate[j] += dPV_dLnDF[i+1] * (J[i][j] / curve.nodes[i+1].df);
            }
        }
    }
    return dPV_dRate;
}

// 8. Main: parse input, build curves, compute outputs
int main() {
    try {
        std::ifstream infile("Input.csv");
        if (!infile) throw std::runtime_error("Cannot open Input.csv");
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(infile, line)) {
            if (!line.empty()) lines.push_back(line);
        }

        auto split = [](const std::string& s) {
            std::vector<std::string> tokens;
            std::stringstream ss(s);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                tokens.push_back(token);
            }
            return tokens;
        };

        int N = std::stoi(split(lines[0])[0]);
        std::vector<double> maturities, cash_rates, swap_rates;
        for (int i = 1; i <= N; ++i) {
            auto tks = split(lines[i]);
            maturities.push_back(tenor_to_years(tks[0]));
            cash_rates.push_back(std::stod(tks[1]) / 100.0);
            swap_rates.push_back(std::stod(tks[2]) / 100.0);
        }

        double query_days = std::stod(split(lines[N+1])[0]);
        double query_t = days_to_years(query_days);

        auto swap_tks = split(lines[N+2]);
        double fixed_rate    = std::stod(swap_tks[0]) / 100.0;
        double swap_maturity = tenor_to_years(swap_tks[1]);
        double fixed_step    = freq_to_years(swap_tks[2]);

        // Build separate portfolios
        std::vector<std::unique_ptr<IInstrument>> cash_portfolio, swap_portfolio;
        for (size_t i = 0; i < maturities.size(); ++i) {
            cash_portfolio.push_back(std::make_unique<CashInstrument>(maturities[i]));
            if (maturities[i] < 0.5)
                swap_portfolio.push_back(std::make_unique<CashInstrument>(maturities[i]));
            else
                swap_portfolio.push_back(std::make_unique<SwapInstrument>(maturities[i], 0.5));
        }

        DiscountCurve cash_linear(interp_linear), cash_aq(interp_averaged_quadratic);
        DiscountCurve swap_linear(interp_linear), swap_aq(interp_averaged_quadratic);

        bootstrap_curve(cash_linear, cash_portfolio, cash_rates);
        bootstrap_curve(cash_aq,     cash_portfolio, cash_rates);
        bootstrap_curve(swap_linear, swap_portfolio, swap_rates);
        bootstrap_curve(swap_aq,     swap_portfolio, swap_rates);

        // Q1
        double q1a = cash_linear.getDF(query_t);
        double q1b = cash_aq.getDF(query_t);
        double q1c = swap_linear.getDF(query_t);
        double q1d = swap_aq.getDF(query_t);

        // Q2.1 pricing
        auto price_swap_fn = [](const DiscountCurve& curve, double fixed, double mat, double fstep) {
            int n = static_cast<int>(std::round(mat / fstep));
            double fixed_annuity = 0.0, t_prev = 0.0;
            for (int k = 1; k <= n; ++k) {
                double ti = (k == n) ? mat : k * fstep;
                fixed_annuity += curve.getDF(ti) * (ti - t_prev);
                t_prev = ti;
            }
            double pv_float = 1.0 - curve.getDF(mat);
            return std::pair<double,double>{(pv_float - fixed * fixed_annuity) * 100.0,
                                            (pv_float / fixed_annuity) * 100.0};
        };

        auto [pv_cle, par_cle] = price_swap_fn(cash_linear, fixed_rate, swap_maturity, fixed_step);
        auto [pv_caq, par_caq] = price_swap_fn(cash_aq,     fixed_rate, swap_maturity, fixed_step);
        auto [pv_sle, par_sle] = price_swap_fn(swap_linear, fixed_rate, swap_maturity, fixed_step);
        auto [pv_saq, par_saq] = price_swap_fn(swap_aq,     fixed_rate, swap_maturity, fixed_step);

        // Q2.2 risks
        std::vector<double> risk_cash_lin = cash_curve_risk(cash_linear, cash_rates,
                                                            fixed_rate, swap_maturity, fixed_step);
        std::vector<double> risk_cash_aq  = cash_curve_risk(cash_aq, cash_rates,
                                                            fixed_rate, swap_maturity, fixed_step);
        std::vector<double> risk_swap_lin = swap_curve_risk(swap_linear, swap_rates,
                                                            fixed_rate, swap_maturity, fixed_step, 0.5);
        std::vector<double> risk_swap_aq  = swap_curve_risk(swap_aq, swap_rates,
                                                            fixed_rate, swap_maturity, fixed_step, 0.5);

        // Write output
        std::ofstream out("Output.csv");
        out << std::fixed << std::setprecision(12);
        out << q1a << "," << q1b << "," << q1c << "," << q1d << "\n";
        out << pv_cle << "," << pv_caq << "," << pv_sle << "," << pv_saq << "\n";
        out << par_cle << "," << par_caq << "," << par_sle << "," << par_saq << "\n";
        for (size_t i = 0; i < (size_t)N; ++i) {
            out << risk_cash_lin[i] << "," << risk_cash_aq[i] << ","
                << risk_swap_lin[i] << "," << risk_swap_aq[i] << "\n";
        }
        std::cout << "Success: Output.csv generated with mathematically exact risk.\n";
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}