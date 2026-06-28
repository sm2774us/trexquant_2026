## Q24 · What Is the Sharpe Ratio, What Are Its Limitations, and What Do You Use Instead?

**Open with the intuition (15 seconds):**
> "The Sharpe ratio is the industry's standard risk-adjusted return metric, but it's built
> on the assumption that returns are Normally distributed with IID draws. Quant strategies
> violate both assumptions. Sharpe conflates volatility with risk, rewards strategies that
> sell tail risk (great Sharpe until the disaster), and is inflated by autocorrelated returns.
> I use Sharpe as a screening tool only, and Deflated Sharpe as the production standard."

---

### Sharpe Ratio

$$SR = \frac{\bar{r} - r_f}{\sigma_r}$$

**Annualized** (for daily returns):

$$SR_{\text{ann}} = \frac{\bar{r}_{\text{daily}}}{\sigma_{\text{daily}}} \times \sqrt{252}$$

**Limitations:**

```
LIMITATION                   WHY IT MATTERS IN QUANT
───────────────────────────  ─────────────────────────────────────────────────────────
Assumes Normal returns       Options strategies appear excellent until tail event
Autocorrelated returns       Sharpe artificially inflated (Lo, 2002): adjust by √(1+2ρ)
Symmetric risk treatment     Downside moves are more costly than upside gains
Multiple trial bias           A Sharpe of 2.0 from 100 trials ≠ Sharpe of 2.0 from 1 trial
No regime awareness          Bull-market Sharpe is not Bear-market Sharpe
```

**Lo's autocorrelation adjustment:**

$$SR_{\text{adjusted}} = \frac{SR}{\sqrt{1 + 2\sum_{k=1}^{q}\rho_k}}$$

Where $\rho_k$ = autocorrelation of returns at lag $k$.

**Alternatives I use in practice:**

```
METRIC          FORMULA                         ADVANTAGE
─────────────   ──────────────────────────────  ──────────────────────────────────────
Sortino Ratio   (r̄ - rf) / σ_downside          Penalizes only downside vol
Calmar Ratio    Annualized return / Max DD       Rewards low drawdown strategies
Omega Ratio     E[max(r-L,0)] / E[max(L-r,0)]  Full distribution, not just moments
Deflated SR     DSR = SR*(1 - correction)        Accounts for backtest selection bias
Probabilistic SR P(SR_true > SR_benchmark)       Uncertainty-aware
```

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q25 · How Do You Test for Statistical Significance in a Backtest?

**Open with the intuition (15 seconds):**
> "The core challenge: we only have one history, and we test many signals on it. Standard
> p-value thresholds (5%) are meaningless when we've run 100+ tests. The correct framework
> uses multiple testing correction, minimum t-statistics calibrated to the number of trials,
> and the Deflated Sharpe Ratio to account for the best-of-many-trials selection bias."

---

### Significance Testing Protocol

**Step 1 — t-statistic of mean IC:**

$$t = \frac{\bar{IC} \cdot \sqrt{T}}{\sigma_{IC}}$$

Require $|t| > t_{\min}$ where:

$$t_{\min}^{\text{Bonferroni}} = \Phi^{-1}\!\left(1 - \frac{\alpha}{2N}\right)$$

For $N = 50$ trials, $\alpha = 0.05$: $t_{\min} \approx 3.8$

**Step 2 — Benjamini-Hochberg-Yekutieli (BHY) FDR Control:**
Less conservative than Bonferroni. Controls the expected fraction of false discoveries:

Rank $p$-values: $p_{(1)} \leq p_{(2)} \leq \ldots \leq p_{(N)}$.
Reject $H_{(i)}$ for all $i \leq k$ where:

$$
k = \max\! \lbrace i : p_{(i)} \leq \frac{i}{N} \cdot \frac{q}{\sum_{j=1}^{N}(1/j)} \rbrace
$$

**Step 3 — Minimum Backtest Length:**

$$T_{\min} = \frac{(Z_{1-\alpha/N})^2 \cdot \sigma_r^2}{\hat{\mu}_r^2}$$

Where $\hat{\mu}_r$ = claimed mean return, $\sigma_r$ = return volatility.

> "I enforce a hard rule: any signal tested after the 20th trial in a research campaign
> must have $|t| > 3.5$ to be considered for inclusion, regardless of how compelling
> the economic story is. When I joined my prior fund, they had a backtest library of
> 600+ signals, all tested against the same historical data. Every 'significant' signal
> at $|t| > 2.0$ was re-evaluated under BHY correction — only 30% survived."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q26 · Solve This Probability / Combinatorics Problem

**Open with the intuition (15 seconds):**
> "Trexquant asks probability and combinatorics questions to test your mathematical foundations.
> I'll walk through a representative example and then show how these techniques apply directly
> to alpha research — because a coin-flip that can't be explained in context doesn't impress."

---

### Classic Problem: Expected Number of Distinct Values in N Draws with Replacement

**Problem:** You draw $n$ balls with replacement from an urn of $k$ distinct balls.
What is the expected number of distinct balls you've seen after $n$ draws?

**Solution:**

Let $D_i$ = indicator that ball $i$ is drawn at least once.

$$P(D_i = 1) = 1 - \left(1 - \frac{1}{k}\right)^n$$

By linearity of expectation:

$$\mathbb{E}\!\left[\sum_{i=1}^k D_i\right] = k\left[1 - \left(1 - \frac{1}{k}\right)^n\right]$$

For $k = 52$ (cards), $n = 52$: $\mathbb{E} = 52(1 - (51/52)^{52}) \approx 52(1 - e^{-1}) \approx 32.9$ unique cards.

---

### Finance Application — Birthday Problem in Signal Mining

**Problem:** You have a universe of $N = 500$ stocks and test whether signal IC > 0 for a randomly
selected stock. By chance alone, how many stocks will appear significant at $p < 0.05$?

$$\mathbb{E}[\text{False Discoveries}] = N \cdot \alpha = 500 \times 0.05 = 25$$

**This is why multiple testing correction is non-negotiable in a large universe.**

### Conditional Probability — Permutations Problem

**How many ways can 10 stocks be ranked in the top-3 positions of a 500-stock universe?**

$$P(500, 3) = \frac{500!}{497!} = 500 \times 499 \times 498 = 124,251,000$$

**How many choose-3 subsets (order independent)?**

$$\binom{500}{3} = \frac{500!}{3! \cdot 497!} = \frac{500 \times 499 \times 498}{6} = 20,708,500$$

> "I apply combinatorics reasoning to alpha signal design: if I have 200 candidate features
> and want combinations of size 3, I'm exploring $\binom{200}{3} = 1,313,400$ triplets. Testing
> all of them at $p < 0.05$ means ~65,000 false discoveries by chance. This is the mathematical
> foundation of why feature interaction mining requires rigorous correction."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q27 · What Is the Deflated Sharpe Ratio and Why Does It Matter?

**Open with the intuition (15 seconds):**
> "The Deflated Sharpe Ratio — Bailey and López de Prado, 2014 — answers: what is the probability
> that a strategy's true Sharpe exceeds a benchmark, after accounting for the fact that we selected
> this strategy from a large set of trials? It's the only Sharpe-type metric that explicitly
> penalizes for the best-of-many-backtest selection effect."

---

### PSR and DSR Definitions

**Probabilistic Sharpe Ratio (PSR):**

$$\widehat{PSR}(SR^{\*}) = \Phi\!\left(\frac{(\hat{SR} - SR^{\*})\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}\right)$$

Where:
- $SR^{\*}$ = benchmark Sharpe ratio (typically 0 or the risk-free Sharpe)
- $\hat{\gamma}_3$ = skewness of returns
- $\hat{\gamma}_4$ = excess kurtosis of returns
- $T$ = number of observations

**Deflated Sharpe Ratio (DSR):**

The benchmark $SR^{\*}$ is set to the expected maximum Sharpe from $N$ IID trials:

$$SR^{\*}_{\max} \approx \left(1 - \gamma\right)\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)$$

Where $\gamma \approx 0.5772$ (Euler-Mascheroni constant).

$$DSR = \widehat{PSR}(SR^{\*}_{\max})$$

A DSR > 0.95 (i.e., PSR > 95% against the max-of-N benchmark) is required for promotion to live trading.

---

### DSR Interpretation Table

```
DSR VALUE    INTERPRETATION
───────────  ───────────────────────────────────────────────────────
> 0.95       Genuine alpha — proceed to live
0.80–0.95    Weak evidence — extend backtest window, more OOS data
0.50–0.80    Likely luck — do not trade live
< 0.50       No evidence of skill after trial correction
```

> "I implemented DSR calculation in Python as a standard output of every backtest run.
> Before DSR, our team promoted signals with IS Sharpe > 2.0. After adopting DSR, we rejected
> 45% of those signals. The ones that survived had dramatically better live-to-backtest performance
> ratios — approximately 0.75× vs 0.40× for the rejected signals."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q28 · Explain Bayesian vs Frequentist Inference — Which Do You Use in Practice?

**Open with the intuition (15 seconds):**
> "Frequentists ask: given the null hypothesis is true, how surprising is my data? Bayesians ask:
> given my data, what do I believe about the parameter? In quant research, Bayesian inference
> is conceptually superior for two reasons: (1) we always have strong prior beliefs about what
> signals should look like (IC ~ 0 for most random features), and (2) we want to update beliefs
> as new data arrives — a natural Bayesian framework."

---

### Mathematical Comparison

**Frequentist:**

$$p\text{-value} = P(\text{data this extreme} \mid H_0) \quad \text{— NOT } P(H_0 \mid \text{data})$$

**Bayesian:**

$$P(\theta \mid \text{data}) = \frac{P(\text{data} \mid \theta) \cdot P(\theta)}{P(\text{data})} \propto \mathcal{L}(\theta) \cdot \pi(\theta)$$

---

### Bayesian Application in Alpha Research

**Empirical Bayes shrinkage of IC estimates:**

Prior: $IC \sim \mathcal{N}(0, \tau^2)$ (most signals have near-zero IC)

Likelihood: $\hat{IC} \mid IC \sim \mathcal{N}(IC, \sigma^2_{IC})$

**Posterior (shrinkage formula):**

$$\mathbb{E}[IC \mid \hat{IC}] = \frac{\tau^2}{\tau^2 + \sigma^2_{IC}} \cdot \hat{IC}$$

The observed IC is shrunk toward zero by the factor $\lambda = \sigma^2_{IC}/(\tau^2 + \sigma^2_{IC})$.

**Practical use:** Shrink all IC estimates before computing signal combination weights.
Prevents the optimizer from over-weighting signals with high observed IC due to lucky estimation.

```
EXAMPLE:
  Signal with observed IC = 0.08, σ_IC = 0.07, prior τ = 0.04
  Posterior IC = (0.04²)/(0.04² + 0.07²) × 0.08 = 0.016/0.0065 × 0.08 ≈ 0.021
  The true IC after shrinkage is 0.021 — much more honest than 0.08
```

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

# 📈 TIME SERIES & ECONOMETRICS

---

## Q29 · What Is Stationarity and Why Does It Matter for Financial Time Series?

**Open with the intuition (15 seconds):**
> "Stationarity means the statistical properties of a time series — mean, variance, covariance
> structure — don't change over time. Almost no financial price series is stationary. Stock prices
> trend upward, volatility clusters. The reason it matters: most time series models (ARIMA, regression
> on returns) assume stationarity. Applying them to non-stationary price data produces spurious results."

---

### Types of Stationarity

**Strict stationarity:** Full joint distribution $F(X_t, X_{t+1}, \ldots, X_{t+k})$ invariant to time shifts.

**Weak (covariance) stationarity:**
$$\mathbb{E}[X_t] = \mu \; \forall t \qquad \text{Var}(X_t) = \sigma^2 < \infty \; \forall t \qquad \text{Cov}(X_t, X_{t+k}) = \gamma(k) \text{ depends only on } k$$

---

### Tests for Stationarity

```
TEST                    NULL HYPOTHESIS       ALTERNATIVE        USE CASE
──────────────────────  ────────────────────  ─────────────────  ─────────────────────────────
ADF (Augmented DF)      Unit root (non-stat)  Stationary         Default test for prices/returns
KPSS                    Stationary            Unit root          Confirm ADF (use both)
Phillips-Perron         Unit root             Stationary         Robust to serial correlation
Zivot-Andrews           Unit root with break  Stationary         When structural break suspected
```

**ADF test statistic:**

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{j=1}^p \delta_j \Delta Y_{t-j} + \epsilon_t$$

Reject unit root if $\hat{\gamma}$ is sufficiently negative (below critical values at 1%, 5%, 10%).

---

### Achieving Stationarity in Practice

```
SERIES TYPE             TRANSFORMATION             RESULT
──────────────────────  ─────────────────────────  ──────────────────────────────
Log price               First difference → log ret  I(1) → I(0): stationary
Interest rates          Level (short rates near-stat) Usually I(1); difference
Volatility (VIX)        Level or log                Near-stationary
Factor exposures        Rolling z-score             Forces approximate stationarity
```

> "I always test stationarity before constructing time series features. For price-based signals,
> I work exclusively in return space (first differences of log price). For macro factors like
> the yield curve slope, I use first differences or z-scores. Failure to enforce stationarity
> in the feature set is the number-one cause of backtest inflation I've seen in candidate research."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q30 · Explain ARIMA — How Would You Fit It and What Are Its Limitations?

**Open with the intuition (15 seconds):**
> "ARIMA is a parametric model for stationary time series that captures autoregressive structure
> (current value depends on past values), integration (differencing to achieve stationarity),
> and moving average structure (current value depends on past forecast errors). For equity returns
> with IC ≈ 0.03, ARIMA rarely outperforms a moving average baseline. Its real value in quant
> research is as a baseline and for understanding serial correlation structure."

---

### ARIMA(p,d,q) Specification

**AR(p) component:**
$$X_t = c + \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \epsilon_t$$

**MA(q) component:**
$$X_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \cdots + \theta_q \epsilon_{t-q}$$

**ARIMA(p,d,q):** Apply $d$ differences to achieve stationarity, then fit ARMA(p,q).

$$\Delta^d X_t = c + \sum_{i=1}^p \phi_i \Delta^d X_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

---

### Identification via ACF/PACF

```
PATTERN                          SUGGESTED MODEL
────────────────────────────────  ─────────────────────────────────
ACF decays, PACF cuts off at p    AR(p)
ACF cuts off at q, PACF decays    MA(q)
Both decay exponentially          ARMA(p,q) — use AIC/BIC to select
All near-zero                     White noise (no model needed)
ACF all positive, slow decay      Non-stationary — difference first
```

**Model selection:** Minimize AIC: $\text{AIC} = 2k - 2\ln(\hat{L})$ where $k$ = number of parameters.

---

### Limitations in Finance

```
LIMITATION                    IMPLICATION
───────────────────────────   ─────────────────────────────────────────────
Assumes linearity              Misses non-linear regime changes
Stationary residuals assumed   Financial volatility clusters (→ GARCH instead)
Constant parameters            Structural breaks not captured
Limited predictive horizon     Accuracy decays rapidly beyond 1–2 steps
Cannot capture seasonality     SARIMA needed (and still limited in finance)
```

> "I use ARIMA for two purposes: (1) forecasting macro factor z-scores 1-week ahead as an input
> feature in the ML model, where ARIMA(1,0,1) typically suffices, and (2) as a residual test
> after fitting a factor model — if ARIMA finds structure in the residuals, my factor model
> is missing something important."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q31 · What Is Cointegration and How Do You Use It to Build a Pairs Trading Signal?

**Open with the intuition (15 seconds):**
> "Two non-stationary series are cointegrated if there exists a linear combination of them that
> is stationary. Economically: two stocks that have a long-run equilibrium relationship — same
> sector, shared cost structure — may drift apart short-term but return to equilibrium. The
> spread between them is mean-reverting and tradeable. This is the statistical foundation of
> pairs trading."

---

### Cointegration Definition

Series $X_t, Y_t \sim I(1)$ (integrated of order 1). They are cointegrated if:

$$Z_t = Y_t - \beta X_t \sim I(0)$$

where $\beta$ is the cointegrating coefficient and $Z_t$ is the stationary spread.

**Engle-Granger two-step test:**

Step 1: Regress $Y_t = \alpha + \beta X_t + Z_t$, obtain $\hat{Z}_t$

Step 2: ADF test on $\hat{Z}_t$ — if stationary, the pair is cointegrated.

**Johansen test:** For multivariate case (baskets of assets), tests the rank of the cointegration matrix.

---

### Pairs Trading Signal Construction

**Step 1 — Identify cointegrated pairs:**
Screen all pairs in a sector using Engle-Granger. Filter by ADF $p < 0.05$.

**Step 2 — Compute hedge ratio $\beta$ via OLS or Kalman filter (time-varying $\beta$):**

$$\hat{\beta}_t = \hat{\beta}_{t-1} + \text{Kalman update from new observation}$$

**Step 3 — Spread z-score (entry signal):**

$$z_t = \frac{\hat{Z}_t - \mu_{\hat{Z}}}{\sigma_{\hat{Z}}}$$

**Step 4 — Entry/exit rules:**

```
z-SCORE THRESHOLD    POSITION
───────────────────  ─────────────────────────────────────────────────
z > +2σ              Short Y, Long X (spread will revert down)
z < −2σ              Long Y, Short X (spread will revert up)
|z| < 0.5σ           Exit all positions (spread mean-reverted)
|z| > 3σ             Stop-loss: pair may be breaking down structurally
```

> "I built a pairs book on 150 cointegrated energy sector pairs. The key operational challenge
> was borrow availability: the short leg needs to be borrowable at reasonable cost. I ran a
> pre-trade borrow check daily. Also: Kalman filter on the hedge ratio was critical —
> static OLS hedge ratios break down during sector rotations, and without updating $\beta$
> daily, the z-score calculation becomes stale and generates false entry signals."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q32 · How Do You Handle Non-Stationarity and Regime Changes in Factor Models?

**Open with the intuition (15 seconds):**
> "Factor model parameters estimated over a 10-year history include multiple different market
> regimes: quantitative easing, rising rates, COVID crash, tech bubble. A static parameter
> estimate averages across all of them and is optimal for none of them. The solution is either
> time-varying parameters via Kalman filter, regime-conditional estimation via HMM, or robust
> estimation that downweights historical observations exponentially."

---

### Techniques for Non-Stationary Factor Models

**1. Exponentially Weighted OLS (EWOLS):**

$$\hat{\boldsymbol{\beta}}_t = \underset{\boldsymbol{\beta}}{\arg\min} \sum_{\tau=1}^t \lambda^{t-\tau}(r_\tau - \boldsymbol{\beta}^\top \mathbf{f}_\tau)^2$$

Where $\lambda \in (0,1)$ is the decay factor. Effective lookback $\approx 1/(1-\lambda)$ days.

**2. Kalman Filter (time-varying $\boldsymbol{\beta}$):**

$$\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
$$r_t = \boldsymbol{\beta}_t^\top \mathbf{f}_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, R)$$

Estimate state $\boldsymbol{\beta}_t$ via Kalman recursion (predict + update).

**3. Regime-Conditional Estimation:**
Fit separate OLS in each HMM regime. Switch model as regime posterior shifts.

```
REGIME      β_market     β_value    β_momentum
──────────  ───────────  ─────────  ────────────────────
Bull        0.95         0.3        0.8 (momentum strong)
Bear        1.15         0.6        −0.2 (momentum reverses)
Crisis      1.40         0.1        −0.5 (all sell)
```

> "I use EWOLS with half-life = 63 trading days for signal beta estimation in our equity book.
> The 63-day half-life was chosen via cross-validation: shorter half-lives were too noisy,
> longer ones were too slow to adapt to the 2022 rate regime change. The calibration check:
> plot the rolling CUSUM of model residuals — structural breaks show as persistent drift."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q33 · What Is Autocorrelation in Returns and How Do You Test for It?

**Open with the intuition (15 seconds):**
> "Autocorrelation in returns — serial correlation — means today's return predicts tomorrow's.
> Positive autocorrelation is the basis for momentum strategies. Negative autocorrelation is
> the basis for mean-reversion strategies. Testing for it correctly requires accounting for
> the fat tails and volatility clustering of financial returns."

---

### Definition and Tests

**Autocorrelation function (ACF):**

$$\rho_k = \frac{\text{Cov}(r_t, r_{t-k})}{\text{Var}(r_t)} = \frac{\sum_{t=k+1}^T (r_t - \bar{r})(r_{t-k} - \bar{r})}{\sum_{t=1}^T (r_t - \bar{r})^2}$$

**Ljung-Box Q-test for joint significance:**

$$Q(m) = T(T+2)\sum_{k=1}^{m}\frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m) \text{ under } H_0: \rho_1 = \ldots = \rho_m = 0$$

Reject white noise if $Q(m) > \chi^2_\alpha(m)$.

**Durbin-Watson statistic (for regression residuals):**

$$DW = \frac{\sum_{t=2}^T(e_t - e_{t-1})^2}{\sum_{t=1}^T e_t^2} \approx 2(1 - \hat{\rho}_1)$$

DW ≈ 2 → no autocorrelation; DW < 1.5 → positive serial correlation.

---

### Financial Interpretation

```
AUTOCORRELATION TYPE    HORIZON      SIGNAL TYPE              STRATEGY
───────────────────────  ──────────   ──────────────────────   ────────────────────────────
Positive (daily)         Intraday     Microstructure momentum  Market-making, intraday trend
Positive (weekly/month)  Medium-term  Price momentum            Cross-sectional momentum
Negative (daily)         1–2 days     Mean-reversion            Short-term stat arb
Negative (intraday)      < 1 hour     Bid-ask bounce            HFT market-making
```

> "On the Russell 1000 universe, I find significant negative 1-day autocorrelation in individual
> stock returns (DW ≈ 1.3, Ljung-Box p < 0.001 for lag 1). This supports a 1-day mean-reversion
> signal. But the autocorrelation in the market portfolio (SPX) is near zero — the individual
> stock autocorrelation diversifies away when aggregated. This implies the signal must be
> implemented cross-sectionally, not as a market-level bet."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q34 · Explain the GARCH Model and When You Would Use It

**Open with the intuition (15 seconds):**
> "GARCH models volatility clustering: the empirical observation that large return days tend to
> be followed by more large return days, and calm days follow calm days. ARCH/GARCH models this
> by making today's variance a function of past squared returns and past variances. In alpha
> research, I use GARCH to (1) normalize returns for vol-adjusted signals, (2) forecast
> short-term volatility for risk management, and (3) estimate conditional VaR."

---

### GARCH(1,1) Specification

**Return equation:**
$$r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$

**Variance equation:**
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

Constraints: $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$, $\alpha + \beta < 1$ (stationarity)

**Unconditional variance:**
$$\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}$$

**GARCH persistence:** $\alpha + \beta$ close to 1 means volatility shocks are very persistent (typical in equity markets: $\alpha + \beta \approx 0.98$).

---

### GARCH Extensions Relevant to Trading

```
MODEL         EXTENSION                        USE CASE
────────────  ───────────────────────────────  ──────────────────────────────────────────
EGARCH        Asymmetric leverage effect        Equity markets: bad news → more vol
GJR-GARCH     Threshold for negative shocks     Same as EGARCH, computationally simpler
TGARCH        Threshold-based asymmetry         Robust alternative to EGARCH
GARCH-M       Volatility in mean equation       Time-varying risk premium modeling
DCC-GARCH     Dynamic conditional correlation   Multi-asset risk models
Student-t err Fat-tailed innovations            Better fit for equity/crypto returns
```

**Parameter estimation:** Maximum Likelihood Estimation (MLE) via BFGS or SLSQP optimization on log-likelihood:

$$\mathcal{L} = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=1}^T\left[\ln(\sigma_t^2) + \frac{r_t^2}{\sigma_t^2}\right]$$

> "I use GARCH(1,1) with Student-t innovations for daily equity return volatility forecasting.
> The Student-t degree-of-freedom parameter $\nu$ typically fits around 5–7 for individual stocks,
> confirming fat tails. I feed the 1-day-ahead GARCH variance forecast as a feature into the
> LightGBM model — it captures the 'calm before the storm' and 'fear clustering' patterns that
> improve next-day return prediction."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

# 🏗️ PORTFOLIO CONSTRUCTION & RISK

---

## Q35 · How Do You Construct a Market-Neutral Portfolio from a Set of Alpha Signals?

**Open with the intuition (15 seconds):**
> "Market-neutral means the portfolio has zero net exposure to the market factor — a beta of zero.
> You're long and short in equal dollar amounts (dollar-neutral) and with equal market-beta-weighted
> amounts (beta-neutral). Market-neutral isolates the idiosyncratic alpha from market risk, which
> is exactly what Trexquant's systematic strategy aims for: pure signal P&L, not market beta."

---

### Dollar-Neutral and Beta-Neutral Construction

**Combined signal score:**
$$\alpha_i = \sum_k w_k f_{k,i}$$

**Optimization (Quadratic Programming):**

$$\max_{\mathbf{h}} \; \boldsymbol{\alpha}^\top \mathbf{h} - \frac{\lambda}{2} \mathbf{h}^\top \mathbf{\Sigma} \mathbf{h}$$

Subject to:
$$\mathbf{1}^\top \mathbf{h} = 0 \quad \text{(dollar-neutral)}$$
$$\boldsymbol{\beta}^\top \mathbf{h} = 0 \quad \text{(beta-neutral)}$$
$$|h_i| \leq h_{\max} \quad \forall i \quad \text{(position limit)}$$

**Sector neutralization:** Add constraints $\mathbf{1}_s^\top \mathbf{h} = 0$ for each sector $s$.
This ensures no sector bets confound the signal P&L.

---

### Portfolio Construction Layers

```
LAYER           CONSTRAINT                                PURPOSE
──────────────  ──────────────────────────────────────────  ──────────────────────────────
1. Dollar       Σ long = Σ short                          Remove market dollar exposure
2. Beta         Σ βᵢhᵢ = 0                                Remove market factor exposure
3. Sector       Σ hᵢ within sector = 0                   Remove sector rotation bets
4. Position     |hᵢ| ≤ position limit (e.g., 2% GMV)     Single-name concentration risk
5. Liquidity    |hᵢ| ≤ κ × ADVᵢ                          Execution feasibility
6. Factor       Σ factor_exposure × hᵢ = 0 per factor   Remove known risk premia (Barra)
```

> "At my prior fund, I implemented factor-neutral construction using the Barra USE4 risk model.
> I neutralized 20+ factors simultaneously via constrained QP solved with CVXPY. The P&L
> attribution showed 85% of realized volatility coming from idiosyncratic exposure post-neutralization —
> exactly what we wanted. The quarterly Barra attribution confirmed that less than 8% of returns
> were explained by style factors."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q36 · What Is Mean-Variance Optimization and Why Is It Fragile in Practice?

**Open with the intuition (15 seconds):**
> "Markowitz MVO maximizes expected return for a given level of portfolio variance. The math is
> elegant. The problem is that the solution is extremely sensitive to the expected return vector —
> which in finance is estimated with enormous uncertainty. A small estimation error in expected
> returns produces wildly different portfolio weights. MVO is an error amplifier."

---

### MVO Formulation

$$
\max_{\mathbf{w}} \; \boldsymbol{\mu}^\top \mathbf{w} - \frac{\lambda}{2} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}
$$

**Closed-form (unconstrained):**

$$
\mathbf{w}^{\*} = \frac{1}{\lambda} \mathbf{\Sigma}^{-1} \boldsymbol{\mu}
$$

**The efficiency frontier** traces out the minimum-variance portfolios for each target return level.

---

### Why MVO Fails in Practice

```
PROBLEM                       ROOT CAUSE                   FIX
────────────────────────────  ───────────────────────────  ──────────────────────────────────────
Expected return uncertainty   μ estimated with σ(μ) ≈ μ   Resampling, Black-Litterman
Covariance matrix instability Noisy Σ̂ (estimation error)  Ledoit-Wolf, RMT filtering
Extreme corner solutions       Unconstrained MVO            Box constraints, L1/L2 penalty on w
Concentration in few assets   Optimizer exploits noise      Maximum position size constraints
Turnover spikes                Solution changes monthly      Turnover penalty in objective
```

**Ledoit-Wolf shrinkage (production fix for Σ):**

$$
\hat{\Sigma}^{\text{LW}} = \alpha \hat{\Sigma}_{\text{sample}} + (1-\alpha)\hat{\mu}_{var} \mathbf{I}
$$

Where $\alpha$ is chosen analytically to minimize expected squared Frobenius loss.

**Black-Litterman (production fix for μ):**
Combine market-implied equilibrium returns $\boldsymbol{\Pi} = \lambda \mathbf{\Sigma} \mathbf{w}^{\text{mkt}}$ with model views:
$$\hat{\boldsymbol{\mu}} = \left[(\tau\mathbf{\Sigma})^{-1} + \mathbf{P}^\top\mathbf{\Omega}^{-1}\mathbf{P}\right]^{-1}\left[(\tau\mathbf{\Sigma})^{-1}\boldsymbol{\Pi} + \mathbf{P}^\top\mathbf{\Omega}^{-1}\mathbf{q}\right]$$

> "I never use raw MVO in production. My workflow: (1) Black-Litterman expected returns from
> the alpha signal IC-weighted views, (2) Ledoit-Wolf shrunk covariance, (3) constrained QP
> with position limits, sector neutrality, and a turnover penalty of $\delta\|\mathbf{w}_t - \mathbf{w}_{t-1}\|^2$.
> The turnover penalty alone reduced annual transaction costs by 35% vs unconstrained MVO."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q37 · How Do You Set Position Sizing and Risk Limits on a Live Book?

**Open with the intuition (15 seconds):**
> "Position sizing is where signal confidence meets risk management. The Kelly criterion gives
> the theoretically optimal bet size to maximize long-run geometric return. Fractional Kelly
> (half-Kelly is common) provides a practical, more conservative sizing that trades off growth
> rate for reduced drawdown variance. In a systematic fund, position limits, gross/net exposure
> limits, and stop-loss rules operate as hard constraints above any sizing model."

---

### Kelly Criterion

For a binary bet with win probability $p$, win payout $b$, lose amount 1:

$$f^{\*} = \frac{pb - (1-p)}{b} = \frac{p(b+1) - 1}{b}$$

**Continuous case (normal returns $r \sim \mathcal{N}(\mu, \sigma^2)$):**

$$f^{\*} = \frac{\mu}{\sigma^2}$$

**In practice: use half-Kelly:**

$$f = \frac{1}{2} \cdot \frac{\mu}{\sigma^2}$$

This reduces the theoretical variance of outcomes by 75% at the cost of ~25% growth rate.

---

### Risk Limit Framework

```
LIMIT TYPE          TYPICAL LEVEL          TRIGGER
──────────────────  ─────────────────────  ──────────────────────────────────────────────
Gross exposure      2–3× AUM               Leverage limit; prime broker collateral
Net exposure        ±10% of AUM            Market neutrality; drift monitored daily
Single-name max     1.5–2% of gross        Concentration limit per position
Sector net max      ±5% of gross           Sector neutrality tolerance band
Daily loss limit    −1.5% of AUM           Intraday circuit breaker; freeze new orders
Monthly loss limit  −5% of AUM             Risk committee review trigger
Factor exposure     ±0.2 σ per factor      Barra factor limit; rebalance required
Max drawdown        −15% of HWM            Strategy review / potential suspension
```

> "In my prior role I implemented a real-time risk limit monitoring system in Python:
> positions were streamed from the OMS, Barra exposures recalculated every 15 minutes,
> and an automated alert fired when any limit was within 80% of its bound.
> The daily loss limit was the most critical — when triggered, all pending orders were
> cancelled and the book was allowed to run at current positions only, no new alpha bets.
> This rule alone prevented three potential runaway drawdown events in 18 months."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q38 · Walk Me Through How You Would Combine Multiple Signals into a Single Portfolio

**Open with the intuition (15 seconds):**
> "Signal combination is a three-step problem: estimate the expected contribution of each signal
> (IC), estimate how signals co-vary (their correlation structure), and solve for optimal weights
> subject to practical constraints. The naive approach — weight by IC — ignores redundancy.
> The correct approach accounts for the full IC-weighted covariance structure."

---

### Signal Combination Framework

**Step 1 — IC vector estimation (Empirical Bayes shrinkage):**

$$\boldsymbol{\mu}_{\text{EB}} = \frac{\tau^2}{\tau^2 + \sigma^2_{\text{IC}}} \cdot \hat{\boldsymbol{\mu}}_{IC}$$

**Step 2 — Signal return covariance estimation (Ledoit-Wolf):**

$$\hat{\mathbf{\Sigma}}^{\text{LW}}$$

**Step 3 — Optimal signal weights:**

$$\mathbf{w}^{\*} = \hat{\mathbf{\Sigma}}^{-1\text{LW}} \cdot \boldsymbol{\mu}_{\text{EB}}$$

Normalize: $\tilde{\mathbf{w}} = \mathbf{w}^{\*} / \|\mathbf{w}^{\*}\|_1$ (sum-to-one constraint)

**Step 4 — Orthogonality check (Gram-Schmidt):**
If any pair of signals has $|\rho| > 0.7$, orthogonalize via QR decomposition before combination.

---

### Practical Combination Architecture

```
APPROACH              FORMULA                  WHEN TO USE
──────────────────    ─────────────────────    ──────────────────────────────────────
Equal weight          wᵢ = 1/N                Early stage; no IC history
IC-weighted           wᵢ ∝ ICᵢ                Single-period IC reliable
Mean-variance         w = Σ⁻¹μ               Full covariance available; N < 50
Principal component   w = first k PCs          High-dim, correlated signal library
Ridge-regularized     w = (Σ+λI)⁻¹μ           Σ estimation noisy; N > 50
Stacking (ML)         ML model of signals      Sufficient history; non-linear combos
```

> "I ran a signal combination exercise on 45 equity signals. Equal-weight produced OOS IR of 1.2.
> IC-weighting gave 1.4. MV-optimal with Ledoit-Wolf gave 1.7. ML stacking with LightGBM using
> the 45 signals as features gave 1.9 — but only because we had 8 years of daily training data.
> For a new signal library with < 2 years of history, ML stacking overfits badly; IC-weighting
> is more robust in that regime."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

# 💻 CODING & ALGORITHM DESIGN

---

## Q39 · Design an Algorithm to Play Hangman with >50% Accuracy (The Trexquant Take-Home)

**Open with the intuition (15 seconds):**
> "This is the Trexquant signature take-home challenge. The problem looks like a game but it's
> actually a Bayesian inference problem: given the revealed letters and word length, what is the
> posterior distribution over all possible words? The optimal strategy guesses the letter that
> maximizes information gain — the one that eliminates the most candidates from the remaining
> word distribution. I'll walk through the full solution."

---

### Problem Statement

- Dictionary: ~200k words (some synthetic/fake like `aaaa...`, `bbbb...`)
- Game: guess one letter at a time; 6 misses = loss
- Objective: achieve accuracy > 50% on 1000 OOS words
- Single submission opportunity — you must be confident

---

### Algorithm Design

**Data Preprocessing:**
```python
# Step 1: Clean dictionary — filter fake words
def is_valid_word(w):
    # Fake words have extreme character repetition
    return max(w.count(c) for c in set(w)) / len(w) < 0.6

valid_words = [w for w in dictionary if is_valid_word(w)]
```

**Core Algorithm — Bayesian Letter Frequency with Constraints:**

```
STATE AT EACH GUESS:
  known: pattern like "_A_T__N" (length=7, pos 1=A, pos 3=T, pos 6=N)
  missed: set of already-guessed wrong letters
  alive: set of candidate words matching the pattern

CANDIDATE FILTER:
  For each word w in valid_words:
    1. len(w) == len(pattern)
    2. All revealed letters match their positions
    3. None of the missed letters appear in w

LETTER SCORING (choose letter that maximizes hits in candidates):
  For each unguessed letter c:
    score(c) = |{w ∈ alive : c ∈ w}| / |alive|

  Guess argmax_c score(c)
```

**Enhanced Scoring — Positional Frequency:**

Instead of just letter presence, score by position-conditional frequency:

$$\text{score}(c) = \sum_{i \notin \text{revealed}} P(c \text{ at position } i \mid \text{alive})$$

This biases toward letters in the unknown positions, not just the whole word.

---

### Performance Targets

```
STRATEGY                        EXPECTED ACCURACY   MISSES/GAME
──────────────────────────────  ──────────────────  ────────────
Random letter order             ~16%                ~6
ETAOIN frequency (unigram)      ~35%                ~4.5
Pattern-conditioned frequency   ~52%                ~3.8  ← meets target
Positional bigram model         ~58%                ~3.3
Transformer language model      ~65%+               ~2.8
```

**Python Implementation Skeleton:**

```python
from collections import Counter, defaultdict
import re

class HangmanSolver:
    def __init__(self, dictionary):
        self.full_dict = [w.lower() for w in dictionary if self._valid(w)]

    def _valid(self, w):
        return len(w) > 0 and max(Counter(w).values()) / len(w) < 0.55

    def guess(self, pattern, missed_letters, guessed_letters):
        pattern_re = pattern.replace('_', '[^' + ''.join(missed_letters or ['']) + ']')
        alive = [w for w in self.full_dict
                 if len(w) == len(pattern)
                 and re.fullmatch(pattern_re, w)
                 and all(m not in w for m in missed_letters)]

        if not alive:  # fallback to frequency order
            for c in 'etaoinshrdlcumwfgypbvkjxqz':
                if c not in guessed_letters:
                    return c

        # Score by conditional letter frequency in candidates
        freq = Counter(c for w in alive for c in set(w)
                       if c not in guessed_letters)
        return freq.most_common(1)[0][0]
```

> "My implementation achieved 57% accuracy on the OOS test set. The critical insight that
> lifted accuracy from 45% to 57% was filtering fake words first: the dictionary contained
> synthetic strings that would never appear in the test set. Detecting and removing them
> reduced the candidate pool quality noise substantially. The second insight: use the pattern
> regex to filter candidates lazily — don't recompute from scratch every guess, cache by pattern."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

## Q40 · LeetCode Medium: Find All Permutations / Array Subarray Problem

**Open with the intuition (15 seconds):**
> "Trexquant LeetCode questions in live interviews are typically medium difficulty: array
> manipulation, sliding windows, or probability-adjacent recursion. The interviewer is not
> testing LeetCode memorization — they are testing your ability to think through complexity,
> recognize patterns, and communicate clearly while coding. Always narrate your approach
> before writing a single line."

---

### Problem 1: Find All Permutations of a String (LC 46/567 style)

**Problem:** Given a string `s`, find all unique permutations.

**Approach:** Backtracking with visited array

```python
def permute_unique(s: str) -> list[str]:
    result = []
    chars = sorted(s)           # Sort to group duplicates

    def backtrack(path, remaining):
        if not remaining:
            result.append(''.join(path))
            return
        seen = set()
        for i, c in enumerate(remaining):
            if c in seen:       # Skip duplicate branches
                continue
            seen.add(c)
            backtrack(path + [c], remaining[:i] + remaining[i+1:])

    backtrack([], list(chars))
    return result

# Time: O(n! × n), Space: O(n × n!)
```

---

### Problem 2: Maximum Sum Subarray of Length K (Sliding Window)

**Problem:** Given array `nums` and integer `k`, find maximum sum of any contiguous subarray of length `k`.

```python
def max_sum_subarray(nums: list[int], k: int) -> int:
    if len(nums) < k:
        return 0
    # Initialize window
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide window
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]   # Add new, remove old
        max_sum = max(max_sum, window_sum)

    return max_sum

# Time: O(n), Space: O(1) — this is the key insight
# Naive O(n×k) approach with nested loops is WRONG for large inputs
```

---

### Problem 3: Probability Variant (Trexquant-Specific)

**Problem:** You roll a fair 6-sided die repeatedly. What is the expected number of rolls
to see each face at least once? (Coupon Collector's Problem)

**Solution:**

$$\mathbb{E}[\text{rolls}] = n \sum_{k=1}^{n} \frac{1}{k} = n \cdot H_n$$

For $n = 6$: $\mathbb{E} = 6(1 + 1/2 + 1/3 + 1/4 + 1/5 + 1/6) = 6 \times 2.45 \approx 14.7$ rolls.

**Python verification:**

```python
import random, statistics

def simulate_coupon(n=6, trials=100_000):
    results = []
    for _ in range(trials):
        seen, rolls = set(), 0
        while len(seen) < n:
            seen.add(random.randint(1, n))
            rolls += 1
        results.append(rolls)
    return statistics.mean(results)

# Expected: ~14.7, simulation confirms it
```

> "My approach in every live coding question: (1) restate the problem in my own words to confirm
> understanding, (2) identify the data structure (array → sliding window? Tree → DFS/BFS?),
> (3) state time/space complexity before writing code, (4) write clean code with meaningful
> variable names, (5) walk through a small example at the end. This process catches most
> edge cases and shows the interviewer that I think systematically — which is what they're
> actually hiring for."

[🔝 Back to Top](./TECHNICAL_INTERVIEW.md#table-of-contents)

---
---

# Quick-Reference Equation Sheet

| Symbol | Definition |
|--------|------------|
| $IC$ | Information Coefficient: $\text{Corr}(f_{i,t}, r_{i,t+h})$ |
| $ICIR$ | $\bar{IC} / \sigma_{IC}$ |
| $IR$ | Information Ratio: $\bar{r}_{\text{active}} / \sigma_{\text{active}}$ |
| $SR$ | Sharpe Ratio: $(\bar{r} - r_f) / \sigma_r$ |
| $DSR$ | Deflated Sharpe: $\widehat{PSR}(SR^{\*}_{\max})$ adjusted for $N$ trials |
| $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ | $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| $\hat{\boldsymbol{\beta}}_{\text{Ridge}}$ | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| $\mathbf{w}^{\*}_{\text{Mkt-Neutral}}$ | $\mathbf{\Sigma}^{-1}\boldsymbol{\alpha}$ s.t. $\mathbf{1}^\top\mathbf{w}=0$, $\boldsymbol{\beta}^\top\mathbf{w}=0$ |
| $f^{\*}_{\text{Kelly}}$ | $\mu / \sigma^2$ (continuous) |
| $`\sigma_{t^{2}_{\text{GARCH}}}`$ | $\omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ |
| $Q_{\text{Ljung-Box}}$ | $T(T+2)\sum_{k=1}^m \hat{\rho}_k^2/(T-k) \sim \chi^2(m)$ |
| $Z_t$ (cointegration) | $Y_t - \beta X_t \sim I(0)$ |
| $\hat{\rho}_{XY}$ | $\text{Cov}(X,Y)/(\sigma_X\sigma_Y)$ |
| $h_{1/2}$ (signal decay) | $\ln 2 / \lambda$ where $IC(h) = IC_0 e^{-\lambda h}$ |
| $\mathbb{E}[\text{Coupon Collector}]$ | $n \cdot H_n = n\sum_{k=1}^n 1/k$ |

---

> **Final Interview Mindset for Trexquant:**
> The firm runs **thousands of systematic signals** — they are not hiring for elegance,
> they are hiring for signal density. Every answer should end with a quantitative result:
> "IC of X," "OOS Sharpe of Y," "reduced drawdown by Z%." Opinions without numbers
> are noise. Experience without production evidence is academia. At Trexquant, the
> proof is always in the live P&L.

---

*Document compiled from Glassdoor (2023–2026), Wall Street Oasis, and confirmed Trexquant interview reports.*
*Questions weighted by recency: 2024–2026 reports given higher priority.*
*Prepared for: Trexquant Investment · Quantitative Researcher — Experienced Hire · Stamford CT / New York NY*