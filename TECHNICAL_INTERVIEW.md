# Trexquant Investment — Quantitative Researcher Interview Playbook
### Experienced Hire · Equities, Futures, Commodities & Event-Driven · Stamford CT / New York NY
#### 40 Questions · Alpha Research · ML/Prediction · Statistics · Time Series · Portfolio · Coding

> **Delivery philosophy:** Every answer below follows *intuition → math → production experience* structure.
> Trexquant's senior researchers will probe two levels deeper on any technique you name.
> The firm runs **thousands of statistical algorithms** across equities, futures, and global markets with
> heavy ML stacking. Never name a model you can't calibrate. Never name an alpha you haven't backtested.
> Confirmed signal from Glassdoor/WSO (2023–2026): probability/Markov chains, linear regression derivations,
> ML questions, correlation/covariance, alpha ideation, time series, permutations, and LeetCode medium.
> Superday format: 3 back-to-back 1-hour sessions + CEO final round. Be ready for anything.

---
---

[↩️ Back to README.md](./README.md)

---
---

## 📚 Technical

---

## ⏱️ Question Budget

```
DOMAIN                          QUESTIONS   LIKELY SESSION         TREXQUANT'S LENS
──────────────────────────────  ──────────  ─────────────────────  ──────────────────────────────────────────
ALPHA RESEARCH & SIGNAL GEN     Q01 – Q10   Round 1 (60 min)       Core competency. Alpha ideas on the spot.
ML / PREDICTION MODELS          Q11 – Q20   Round 2 (60 min)       Model architecture, overfitting, features
STATISTICS & PROBABILITY        Q21 – Q28   Round 1/2 overlap      Markov chains, combinatorics, distributions
TIME SERIES & ECONOMETRICS      Q29 – Q34   Round 2                ARIMA, cointegration, stationarity
PORTFOLIO CONSTRUCTION & RISK   Q35 – Q38   Round 3 / CEO          Market-neutral, Sharpe, risk limits
CODING / ALGORITHM DESIGN       Q39 – Q40   All rounds             Hangman-style take-home + live LeetCode
```

> **Priority rule:** Alpha ideation and ML are Trexquant's lifeblood — the CEO has been known to pivot
> from resume review straight to *"give me a novel alpha idea for equities right now."*
> Have Q01–Q10 at fingertips. Statistics (Q21–Q28) surfaces in every round as a filter.

---

## Table of Contents

### 🔬 ALPHA RESEARCH & SIGNAL GENERATION
- [Q01 · What is Alpha? Define it Rigorously and Give a Live Example](#q01--what-is-alpha-define-it-rigorously-and-give-a-live-example)
- [Q02 · Walk Me Through How You Would Design a New Alpha Signal from Scratch](#q02--walk-me-through-how-you-would-design-a-new-alpha-signal-from-scratch)
- [Q03 · How Do You Distinguish True Alpha from Backtest Overfitting?](#q03--how-do-you-distinguish-true-alpha-from-backtest-overfitting)
- [Q04 · What Is the Information Coefficient and How Do You Use It?](#q04--what-is-the-information-coefficient-and-how-do-you-use-it)
- [Q05 · How Do You Measure Signal Decay and What Does It Tell You?](#q05--how-do-you-measure-signal-decay-and-what-does-it-tell-you)
- [Q06 · How Would You Build a Momentum Signal for Cross-Sectional Equity Trading?](#q06--how-would-you-build-a-momentum-signal-for-cross-sectional-equity-trading)
- [Q07 · What Is the Alpha Combination Problem and How Would You Solve It?](#q07--what-is-the-alpha-combination-problem-and-how-would-you-solve-it)
- [Q08 · How Do You Handle Alternative Data in Alpha Research?](#q08--how-do-you-handle-alternative-data-in-alpha-research)
- [Q09 · How Do You Detect and Adjust for Alpha Crowding?](#q09--how-do-you-detect-and-adjust-for-alpha-crowding)
- [Q10 · Propose an Alpha Idea for Futures Markets Right Now](#q10--propose-an-alpha-idea-for-futures-markets-right-now)

### 🤖 MACHINE LEARNING & PREDICTION MODELS
- [Q11 · Derive Linear Regression from First Principles in Matrix Form](#q11--derive-linear-regression-from-first-principles-in-matrix-form)
- [Q12 · How Do You Prevent Overfitting in an ML Model for Financial Data?](#q12--how-do-you-prevent-overfitting-in-an-ml-model-for-financial-data)
- [Q13 · Compare Lasso vs Ridge Regularization — When Do You Use Each?](#q13--compare-lasso-vs-ridge-regularization--when-do-you-use-each)
- [Q14 · How Does Gradient Boosting Work and Why Is It Effective for Alpha Research?](#q14--how-does-gradient-boosting-work-and-why-is-it-effective-for-alpha-research)
- [Q15 · Explain SHAP Values and How You Use Them in Practice](#q15--explain-shap-values-and-how-you-use-them-in-practice)
- [Q16 · How Do You Build an ML Pipeline for Predicting Next-Day Stock Returns?](#q16--how-do-you-build-an-ml-pipeline-for-predicting-next-day-stock-returns)
- [Q17 · What Is a Random Forest and How Does It Reduce Variance?](#q17--what-is-a-random-forest-and-how-does-it-reduce-variance)
- [Q18 · How Would You Use a Neural Network for Time Series Alpha Research?](#q18--how-would-you-use-a-neural-network-for-time-series-alpha-research)
- [Q19 · How Do You Evaluate ML Model Performance in a Financial Context?](#q19--how-do-you-evaluate-ml-model-performance-in-a-financial-context)
- [Q20 · What Is Feature Importance and How Do You Avoid the Correlation Trap?](#q20--what-is-feature-importance-and-how-do-you-avoid-the-correlation-trap)

### 📊 STATISTICS & PROBABILITY
- [Q21 · Derive the Formula for Covariance and Correlation from Scratch](#q21--derive-the-formula-for-covariance-and-correlation-from-scratch)
- [Q22 · What Is a Markov Chain and How Would You Apply It to Markets?](#q22--what-is-a-markov-chain-and-how-would-you-apply-it-to-markets)
- [Q23 · Walk Me Through the Central Limit Theorem and Its Limitations in Finance](#q23--walk-me-through-the-central-limit-theorem-and-its-limitations-in-finance)
- [Q24 · What Is the Sharpe Ratio, What Are Its Limitations, and What Do You Use Instead?](#q24--what-is-the-sharpe-ratio-what-are-its-limitations-and-what-do-you-use-instead)
- [Q25 · How Do You Test for Statistical Significance in a Backtest?](#q25--how-do-you-test-for-statistical-significance-in-a-backtest)
- [Q26 · Solve This Probability / Combinatorics Problem](#q26--solve-this-probability--combinatorics-problem)
- [Q27 · What Is the Deflated Sharpe Ratio and Why Does It Matter?](#q27--what-is-the-deflated-sharpe-ratio-and-why-does-it-matter)
- [Q28 · Explain Bayesian vs Frequentist Inference — Which Do You Use in Practice?](#q28--explain-bayesian-vs-frequentist-inference--which-do-you-use-in-practice)

### 📈 TIME SERIES & ECONOMETRICS
- [Q29 · What Is Stationarity and Why Does It Matter for Financial Time Series?](#q29--what-is-stationarity-and-why-does-it-matter-for-financial-time-series)
- [Q30 · Explain ARIMA — How Would You Fit It and What Are Its Limitations?](#q30--explain-arima--how-would-you-fit-it-and-what-are-its-limitations)
- [Q31 · What Is Cointegration and How Do You Use It to Build a Pairs Trading Signal?](#q31--what-is-cointegration-and-how-do-you-use-it-to-build-a-pairs-trading-signal)
- [Q32 · How Do You Handle Non-Stationarity and Regime Changes in Factor Models?](#q32--how-do-you-handle-non-stationarity-and-regime-changes-in-factor-models)
- [Q33 · What Is Autocorrelation in Returns and How Do You Test for It?](#q33--what-is-autocorrelation-in-returns-and-how-do-you-test-for-it)
- [Q34 · Explain the GARCH Model and When You Would Use It](#q34--explain-the-garch-model-and-when-you-would-use-it)

### 🏗️ PORTFOLIO CONSTRUCTION & RISK
- [Q35 · How Do You Construct a Market-Neutral Portfolio from a Set of Alpha Signals?](#q35--how-do-you-construct-a-market-neutral-portfolio-from-a-set-of-alpha-signals)
- [Q36 · What Is Mean-Variance Optimization and Why Is It Fragile in Practice?](#q36--what-is-mean-variance-optimization-and-why-is-it-fragile-in-practice)
- [Q37 · How Do You Set Position Sizing and Risk Limits on a Live Book?](#q37--how-do-you-set-position-sizing-and-risk-limits-on-a-live-book)
- [Q38 · Walk Me Through How You Would Combine Multiple Signals into a Single Portfolio](#q38--walk-me-through-how-you-would-combine-multiple-signals-into-a-single-portfolio)

### 💻 CODING & ALGORITHM DESIGN
- [Q39 · Design an Algorithm to Play Hangman with >50% Accuracy (The Trexquant Take-Home)](#q39--design-an-algorithm-to-play-hangman-with-50-accuracy-the-trexquant-take-home)
- [Q40 · LeetCode Medium: Find All Permutations / Array Subarray Problem](#q40--leetcode-medium-find-all-permutations--array-subarray-problem)

- **[Quick-Reference Equation Sheet](#quick-reference-equation-sheet)**

[🔝 Back to Top](#-table-of-contents)

---
---

# 🔬 ALPHA RESEARCH & SIGNAL GENERATION

---

## Q01 · What is Alpha? Define it Rigorously and Give a Live Example

**Open with the intuition (15 seconds):**
> "Alpha is excess return unexplained by systematic risk factors — it's the part of P&L that persists after
> you've orthogonalized against market, sector, style, and any other known risk premia. The moment alpha
> becomes explained by a known factor, it's no longer alpha: it's a beta you weren't charging for. At Trexquant's
> scale, running thousands of signals, the real question is always: is this signal orthogonal to everything
> else in the library, or am I just re-discovering momentum in disguise?"

---

### Formal Definition

The return decomposition in a factor model:

$$r_i = \alpha_i + \sum_{k=1}^{K} \beta_{ik} f_k + \epsilon_i$$

Where:
- $r_i$ = total return on asset $i$
- $\alpha_i$ = **idiosyncratic excess return** (alpha): the intercept after factor-adjusting
- $\beta_{ik}$ = exposure of asset $i$ to factor $k$
- $f_k$ = return to factor $k$ (market, value, momentum, size, etc.)
- $\epsilon_i$ = residual: zero-mean, ideally white noise

**Alpha in the systematic context:**

$$\alpha_i = \mathbb{E}[r_i] - r_f - \sum_{k=1}^{K} \beta_{ik} \left(\mathbb{E}[f_k] - r_f\right)$$

---

### The Trexquant Lens — Alpha Taxonomy

```
ALPHA TYPE          DESCRIPTION                              DECAY SPEED   TREXQUANT FIT
──────────────────  ────────────────────────────────────────  ────────────  ──────────────────────────
Price-based         Momentum, mean-reversion, vol signals     Days–weeks    HIGH — thousands of signals
Fundamental         Earnings surprise, analyst revision        Weeks–months  MEDIUM — alt data layer
Microstructure      Order flow imbalance, bid-ask spread       Intraday      HIGH — short horizon ML
Event-driven        Earnings, M&A, index rebalance            Days          HIGH — equities & futures
Cross-sectional     Rank-based signals across universe         Daily         CORE — Trexquant's bread
Cross-asset         FX vol vs equity vol regime signal         Days          MEDIUM — futures book
```

**Illustrate with a live example:**
> "A signal I built at my prior firm was a cross-sectional earnings revision momentum factor.
> Idea: sell-side analysts systematically under-react to positive earnings surprises, continuing
> to revise estimates upward over the subsequent 30–60 days. I computed a 3-month z-score of
> EPS consensus revision velocity across the Russell 1000 universe, formed decile portfolios,
> went long top decile / short bottom decile, dollar-neutral within GICS sector, and achieved
> an information ratio of 0.83 on out-of-sample data from 2018–2023 before transaction costs.
> After Barra risk adjustment, the alpha was robust across market regimes."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q02 · Walk Me Through How You Would Design a New Alpha Signal from Scratch

**Open with the intuition (15 seconds):**
> "Alpha research has a defined scientific method: hypothesis → feature engineering → walk-forward
> validation → orthogonality test → production monitoring. Where most researchers fail is they
> skip the hypothesis step and mine data directly. Data mining without an economic rationale produces
> signals that are historical artifacts, not forward-looking edges."

---

### The Alpha Development Lifecycle

```
STAGE       ACTION                              KEY QUESTION
──────────  ──────────────────────────────────  ─────────────────────────────────────────────
1. IDEA     Economic/behavioral hypothesis       WHY would this signal predict returns?
2. DATA     Source, clean, align to universe     Is this data point-in-time? Survivorship?
3. FEATURE  Transform raw data to ML features    Rank, z-score, winsorize, neutralize?
4. BACKTEST Walk-forward cross-validation        IS-vs-OOS spread acceptable?
5. STATS    T-stat, IC, DSR, turnover            Is this genuine or overfitted?
6. COMBINE  Orthogonalize vs existing library    Marginal IC when added to portfolio?
7. MONITOR  Live performance vs backtest          Decay faster than expected? Kill it.
```

**Illustrate step by step:**

**Step 1 — Hypothesis:** Short-sellers who report positions on SEC 13F-2 (short disclosure) form clusters
of informed bearish positioning. Stocks with recent increases in disclosed short interest relative to
float should underperform over the next 20 days.

**Step 2 — Data:** Pull SEC EDGAR short disclosures (filed within 1 business day of threshold crossing).
Critical: use filing date, not position date. Avoids look-ahead bias.

**Step 3 — Feature:**

$$
\text{ShortMomentum}_i = \frac{\Delta\text{SI}_{i,t} - \mu_{\Delta\text{SI}}}{\sigma_{\Delta\text{SI}}} \quad \text{(cross-sectional z-score)}
$$

Neutralize by sector. Winsorize at 3σ to avoid extreme outliers in thin stocks.

**Step 4 — Validation:** Walk-forward, re-train monthly, 252-day OOS window per fold.

**Step 5 — Stats:** Target $|t| > 2.5$, IC > 0.04, Deflated Sharpe > 1.0.

**Step 6 — Combination:** Run Gram-Schmidt to verify signal is not collinear with existing
price-momentum factor in library. Marginal IR contribution > 0.05 required.

> "In production, I set up a decay monitor: if the rolling 20-day IC drops below 0.02 for
> three consecutive months, the signal is flagged for kill/rebuild. I've killed signals I was
> emotionally attached to — that discipline is what separates researchers from gamblers."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q03 · How Do You Distinguish True Alpha from Backtest Overfitting?

**Open with the intuition (15 seconds):**
> "The fundamental problem in quant research is that with enough parameters, any historical
> dataset can be perfectly explained by a model that has zero predictive power going forward.
> The goal is not a high in-sample Sharpe — any idiot can achieve that. The goal is a
> statistically significant out-of-sample IR that survives multiple testing correction."

---

### The Overfitting Diagnosis Framework

**Rule 1 — Multiple Testing Correction (Harvey-Liu-Zhu, 2016):**

With $N$ trials, the minimum $t$-statistic required for significance at 5% level is NOT 1.96.
Under Bonferroni correction:

$$t_{\min} = \Phi^{-1}\!\left(1 - \frac{\alpha}{2N}\right)$$

For $N = 100$ backtests: $t_{\min} \approx 4.1$, NOT 1.96.
The Bailey-López de Prado **Deflated Sharpe Ratio (DSR)** penalizes for the number of trials:

$$\widehat{SR}^* = \hat{SR} \cdot \left(1 - \gamma \cdot \hat{SR} \cdot \frac{\sqrt{V[\hat{SR}]}}{\sqrt{T}}\right)$$

**Rule 2 — Walk-Forward vs Random CV:**

In financial time series, standard k-fold cross-validation is INVALID because it allows
future information to leak into the training set. Correct approach: **Purged Walk-Forward CV**
or **Combinatorial Purged Cross-Validation (CPCV)**.

```
WALK-FORWARD STRUCTURE

  FOLD    TRAIN WINDOW       GAP    TEST WINDOW    NOTE
  ─────   ────────────────   ───    ────────────   ─────────────────────────────
  1       T0 → T0+2yr        6wk    T0+2yr+1yr     gap prevents leakage
  2       T0 → T0+3yr        6wk    T0+3yr+1yr     expanding window
  3       T0 → T0+4yr        6wk    T0+4yr+1yr
  4       T0 → T0+5yr        6wk    T0+5yr+1yr
  OOS avg = mean of test window Sharpe across all folds
```

**Rule 3 — IS/OOS Sharpe Ratio Spread:**

```
IS SR / OOS SR    VERDICT
────────────────  ─────────────────────────────────────────
1.0 – 1.5×        ACCEPTABLE — expected efficiency loss
1.5 – 2.5×        CAUTION — possible mild overfitting
> 2.5×            REJECT — likely overfitted. Kill signal.
```

> "In my prior role, I mandated CPCV with 10 paths and a 6-week embargo gap between train/test.
> Any signal with IS/OOS Sharpe ratio > 2.5 was rejected regardless of IS performance.
> This filter alone eliminated 40% of the candidate signals that passed naive backtesting."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q04 · What Is the Information Coefficient and How Do You Use It?

**Open with the intuition (15 seconds):**
> "The IC is the Spearman rank correlation between my signal and the next-period realized return.
> It's the single most important diagnostic metric in cross-sectional alpha research. IC tells me
> whether my signal has any predictive power at all, before I touch optimization or portfolio construction."

---

### Definitions

**Pearson IC:**

$$IC_t = \text{Corr}(f_{i,t},\; r_{i,t+h}) = \frac{\sum_i (f_{i,t} - \bar{f}_t)(r_{i,t+h} - \bar{r}_{t+h})}{\sqrt{\sum_i (f_{i,t} - \bar{f}_t)^2 \cdot \sum_i (r_{i,t+h} - \bar{r}_{t+h})^2}}$$

**Rank IC (Spearman) — preferred in practice due to robustness to outliers:**

$$\text{Rank IC}_t = \text{Corr}(\text{rank}(f_{i,t}),\; \text{rank}(r_{i,t+h}))$$

**Information Ratio of the IC series:**

$$ICIR = \frac{\bar{IC}}{\sigma_{IC}}$$

---

### IC Benchmarks for Trexquant's Equity Universe

```
METRIC           WEAK         ACCEPTABLE    STRONG        EXCEPTIONAL
────────────     ─────────    ──────────    ──────────    ──────────────
Mean |IC|        < 0.02       0.02–0.05     0.05–0.10     > 0.10
ICIR             < 0.5        0.5–1.0       1.0–2.0       > 2.0
Fraction IC>0    < 52%        52–55%        55–60%        > 60%
```

**The Fundamental Law of Active Management:**

$$IR \approx IC \cdot \sqrt{N}$$

Where $N$ = number of independent bets per period. This tells you: a signal with IC = 0.03
applied to 1000 stocks daily gives $IR \approx 0.03 \times \sqrt{1000} \approx 0.95$ — tradeable.

> "I monitor IC on a rolling 63-day basis for every signal in production. If the rolling IC
> drops below zero for 10 consecutive trading days, the signal is flagged. I also decompose
> IC by sector and market-cap bucket: sometimes a signal that looks dead aggregate-level is
> actually still live in large-cap tech but dead in small-cap materials. That granularity tells
> you how to recalibrate the signal scope rather than killing it outright."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q05 · How Do You Measure Signal Decay and What Does It Tell You?

**Open with the intuition (15 seconds):**
> "Every alpha has a half-life. A momentum signal built on 12-month returns might have edge at
> 20-day holding horizon but be completely dead at 1-day holding. Decay analysis tells you the
> optimal rebalancing frequency and, critically, whether your signal is competing with HFT or
> with slower fundamental investors."

---

### Signal Decay Measurement

Compute IC at multiple forward horizons $h = 1, 2, 5, 10, 20, 60$ days:

$$IC(h) = \text{Corr}(f_{i,t},\; r_{i,t \to t+h})$$

Plot $IC(h)$ vs $h$. Fit an exponential decay:

$$IC(h) = IC_0 \cdot e^{-\lambda h}$$

The **half-life** is:

$$h_{1/2} = \frac{\ln 2}{\lambda}$$

**Optimal rebalancing frequency:**
- Rebalance when the cost of *not* rebalancing (decayed IC) exceeds transaction cost.
- In practice: rebalance at $h^* \approx h_{1/2}$.

```
DECAY CURVE EXAMPLE

  IC(h)
  0.08 |●
  0.06 |  ●
  0.04 |     ●
  0.03 |        ●
  0.02 |           ●   ●
  0.01 |                  ●   ●
  0.00 +────────────────────────────→ h (days)
        1  2  5  10  20  40  60

  Half-life ≈ 7 days → rebalance weekly
```

> "At my prior fund, I found that our OFI (order flow imbalance) signal had a half-life of
> 2 hours intraday, while our earnings revision signal had a half-life of 18 days. We
> traded them on completely different infrastructure: the OFI signal was executed algorithmically
> at open/close; the revision signal was rebalanced weekly. Mixing them on a daily rebalance
> schedule would have crushed the OFI alpha with turnover costs."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q06 · How Would You Build a Momentum Signal for Cross-Sectional Equity Trading?

**Open with the intuition (15 seconds):**
> "Cross-sectional momentum is the empirical observation — Jegadeesh and Titman, 1993 — that
> stocks that outperformed over the past 3–12 months continue to outperform over the next
> 1–3 months, and vice versa. At Trexquant's scale, the naive 12-1 momentum is well-known and
> heavily crowded. The edge is in the refinements: sector-neutralization, skip-month exclusion,
> volatility scaling, and residual momentum after Barra factor adjustment."

---

### Standard Cross-Sectional Momentum Construction

**Raw momentum:**

$$\text{MOM}_{i,t} = \frac{P_{i,t-21}}{P_{i,t-252}} - 1 \quad \text{(12-month return, skip last month)}$$

**Cross-sectional z-score (neutralized within sector):**

$$z_{i,t} = \frac{\text{MOM}_{i,t} - \mu_{\text{sector},t}}{\sigma_{\text{sector},t}}$$

**Volatility scaling (Barroso-Santa-Clara, 2015):**

$$\text{MOM}^{\text{scaled}}_{i,t} = \frac{z_{i,t}}{\hat{\sigma}_{i,t}}$$

Where $\hat{\sigma}_{i,t}$ = realized volatility over past 21 days. This normalizes position sizing
so that high-volatility stocks don't dominate the risk budget.

**Residual momentum (orthogonal to market and sector factors):**

$$\text{RMOM}_{i,t} = z_{i,t} - \hat{\beta}_{i,t}^{\text{MKT}} \cdot f_{\text{MKT},t} - \hat{\beta}_{i,t}^{\text{SEC}} \cdot f_{\text{SEC},t}$$

This strip out the factor-explained portion and trades only the idiosyncratic momentum component.

---

### Momentum Crash Protection

```
MOMENTUM CRASH REGIME INDICATORS
  Signal: VIX > 30 AND MKT return past 12m < −15%
  Action: Scale exposure to 30% of target weight
  Reason: Momentum unwinds violently after market crashes
          as beaten-down stocks rebound faster (short squeeze)
```

> "I built a momentum signal for a Russell 3000 universe. The key finding: raw 12-1 momentum
> had IC ≈ 0.035. After sector-neutralization and vol-scaling, IC rose to 0.048. After
> residualizing against Fama-French 5 factors (RMOM), IC reached 0.062 with substantially
> lower drawdowns during the 2020 COVID crash. The crash protection regime filter eliminated
> the Feb–May 2020 drawdown entirely."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q07 · What Is the Alpha Combination Problem and How Would You Solve It?

**Open with the intuition (15 seconds):**
> "The combination problem is: given N alpha signals, each with its own IC, noise, and mutual
> correlations, what is the optimal weight vector to assign to each signal to maximize the
> aggregate portfolio IR? The naive answer — weight by IC — ignores signal correlations.
> The correct answer uses the covariance matrix of signal returns, not just individual ICs."

---

### Optimal Signal Combination

Define signal matrix $\mathbf{F} \in \mathbb{R}^{T \times N}$ (rows=time, cols=signals).
Define forward returns $\mathbf{r} \in \mathbb{R}^T$.

IC vector: $\boldsymbol{\mu} = \mathbf{F}^\top \mathbf{r} / T$

Signal covariance: $\mathbf{\Sigma} = \mathbf{F}^\top \mathbf{F} / T$

**Optimal weights (analogous to Markowitz for signals):**

$$\mathbf{w}^* = \mathbf{\Sigma}^{-1} \boldsymbol{\mu}$$

This maximizes the Information Ratio of the combined signal.

**Practical issue:** $\mathbf{\Sigma}$ is estimated with noise. Solutions:

```
COVARIANCE ESTIMATION METHOD    WHEN TO USE
──────────────────────────────  ──────────────────────────────────────────
Ledoit-Wolf shrinkage           N signals, moderate T — default choice
Random Matrix Theory cutoff     Large N (>50 signals) — remove noise eigenvectors
Factor model for Σ              When signals have known factor structure
Equal-weight fallback           When T/N < 3 — regularization beats estimation
```

**Gram-Schmidt Orthogonalization (Trexquant-relevant):**
When signals are nearly collinear, orthogonalize via QR decomposition:

$$\mathbf{F} = \mathbf{Q}\mathbf{R}$$

Trade $\mathbf{Q}$ columns (orthogonal factors) directly — eliminates redundancy.

> "In production I maintained a signal library of ~80 factors. I ran Ledoit-Wolf shrinkage
> on the 252-day rolling signal-return covariance matrix, then solved for optimal weights
> monthly. Signals with weight < 0.01 were quarantined — not deleted, but excluded from
> the live allocation until their IC recovered."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q08 · How Do You Handle Alternative Data in Alpha Research?

**Open with the intuition (15 seconds):**
> "Alternative data is powerful exactly because it's not in consensus models — but it's also
> where the worst data quality and look-ahead bias live. My protocol for any alt data source:
> (1) point-in-time verification, (2) coverage stability analysis, (3) vendor revision history
> audit, and only then (4) signal construction. Skipping step 1 guarantees phantom alpha."

---

### Alternative Data Evaluation Framework

```
STEP    QUESTION                            HOW TO VERIFY
──────  ──────────────────────────────────  ──────────────────────────────────────────
1       Is the data truly point-in-time?    Request vendor's as-of history (not restated)
2       What fraction of universe covered?  Coverage < 30% → size/selection bias risk
3       Is there survivorship bias?         Does coverage shrink for bankrupt firms?
4       Is the revision frequency known?    NLP signals revised retroactively post-training?
5       What is the data latency?           Satellite imagery: 2-day lag matters for fast signals
6       Is it permissioned for trading?     MNPI risk — legal review mandatory
```

**Common Alt Data Sources in Equity Research:**

```
DATA TYPE                   SIGNAL IDEA                         EDGE HORIZON
──────────────────────────  ──────────────────────────────────  ────────────
Credit card transactions    Revenue surprise pre-announcement    30–60 days
Satellite parking lot data  Retail foot traffic vs expectation   30–60 days
Job postings               Hiring momentum → capex proxy         60–90 days
App download data          User growth velocity                  30 days
News/NLP sentiment         Earnings call tone, 8-K language      1–5 days
Short interest disclosures Informed short positioning            10–30 days
Options order flow         Implied smart-money directional bets  1–10 days
```

> "The most important validation I run on any new alt data source: I regress the signal on
> lagged returns to check for autocorrelation induced by look-ahead. If the signal has
> suspiciously high predictive power in the first week of using it, I assume contamination —
> not genius. Every vendor-supplied signal has been backtested to look perfect by the vendor."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q09 · How Do You Detect and Adjust for Alpha Crowding?

**Open with the intuition (15 seconds):**
> "At Trexquant's scale with thousands of signals, some of them are going to discover the same
> phenomenon. Externally, factor crowding is the bigger risk: when many systematic funds run the
> same momentum or value signal, the alpha degrades and the tail risk during unwinds is catastrophic.
> August 2007 quant quake is the textbook example. I have a live crowding dashboard."

---

### Crowding Detection Methods

**Method 1 — Pairwise Signal Correlation:**
If two signals have rolling 63-day rank-correlation > 0.70, they are likely capturing the same
phenomenon. Weight-penalize the newer signal by the overlap fraction.

**Method 2 — Factor Crowding via 13F Positioning:**
Track hedge fund aggregate positioning (from quarterly 13F filings, compiled by sources like FactSet).

$$\text{HF Overlap}_i = \frac{\text{\\# top-10 HFs holding stock } i}{\text{Total HFs in universe}}$$

$$\text{Crowding Signal}_t = z\!\left(\text{HF Overlap}_{i,t}\right) \text{ (cross-sectional z-score)}$$

High crowding → reduce exposure → reversal risk on negative catalyst.

**Method 3 — Short-Side Crowding via Borrow Rate:**
```
BORROW COST       SIGNAL INTERPRETATION
──────────────    ────────────────────────────────────────
< 25 bps          General collateral — not crowded
25–100 bps        Moderately crowded short
100–500 bps       Highly crowded — reversal risk elevated
> 500 bps         Extreme crowd — expect violent squeeze
```

**Method 4 — Factor Return Drawdown vs Expected IC:**
If a factor is experiencing significantly worse live performance than historical IC would predict,
and correlation with other known factors has risen sharply, assume crowding-driven degradation.

> "After the 2022 factor unwind in growth/momentum, I built a regime detector:
> when the 20-day realized correlation between our momentum signal and our quality signal
> exceeded 0.85 (they should be ~0.2 in normal regimes), I interpreted it as systematic
> funds de-risking uniformly — everyone selling everything. The correct response was to
> reduce gross exposure by 50%, not to add."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q10 · Propose an Alpha Idea for Futures Markets Right Now

**Open with the intuition (15 seconds):**
> "I'll give you a concrete one: cross-asset volatility spillover from equity vol (VIX) to commodity
> futures. The economic hypothesis is that when equity implied volatility spikes sharply, institutional
> investors liquidate commodity futures positions to meet margin calls and risk limits — creating a
> short-term predictable price dip in commodities that mean-reverts within 3–5 days once vol settles."

---

### Cross-Asset Vol Spillover Alpha — Specification

**Signal construction:**

```math
\text{VOL\_SPIKE}_t = \frac{\Delta \text{VIX}_t - \mu_{\Delta\text{VIX}}}{\sigma_{\Delta\text{VIX}}} \quad \text{(1-day z-score of VIX change)}
```

```math
\text{ALPHA}_t = -\text{VOL\_SPIKE}_t \cdot \mathbf{1}\left[\text{VOL\_SPIKE}_t > 1.5\sigma\right]
```

**Go long commodities (crude oil, gold, copper) when $\text{ALPHA}_t$ is positive** (i.e., after a VIX spike)
with position scaled by VIX spike magnitude.

```
REGIME              VIX CHANGE     COMMODITY ACTION    EXPECTED HORIZON
─────────────────   ─────────────  ──────────────────  ────────────────
Normal              |ΔVIX| < 1σ    No signal           N/A
Moderate spike      1σ – 2σ        Scale in longs      3–5 days
Extreme spike       > 2σ           Max long            2–3 days
VIX normalization   ΔVIX < −1σ     Exit longs          1 day
```

**Validation criteria:**
- IC target: > 0.04 at 3-day horizon
- Sharpe target: > 0.8 on walk-forward OOS
- Test universe: CL, GC, HG, ZC, ZS futures, 2015–2025
- Transaction cost model: round-trip ~2–4 bps per futures contract

> "I've explored a version of this on crude oil futures. The signal is strongest in the 72
> hours after a VIX spike > 2σ. The mean-reversion in oil was 1.3% on average, which at
> 1-2× leverage gives a clean edge before transaction costs. The risk: sometimes vol spikes
> are accompanied by fundamental negative news for oil (demand destruction), so I layer in
> a filter: only take the long side when the WTI spot price itself did NOT fall more than 2σ
> on the same day as the VIX spike."

[🔝 Back to Top](#-table-of-contents)

---
---

# 🤖 MACHINE LEARNING & PREDICTION MODELS

---

## Q11 · Derive Linear Regression from First Principles in Matrix Form

**Open with the intuition (15 seconds):**
> "Linear regression is the backbone of every factor model, every signal validation, and every
> ML baseline. OLS gives you the Best Linear Unbiased Estimator under the Gauss-Markov assumptions.
> The closed-form matrix solution is elegant but fragile when features are collinear or when
> assumptions are violated — which is almost always in financial data."

---

### OLS Derivation — Matrix Form

**Setup:** $N$ observations, $K$ features.
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

**Loss function (Residual Sum of Squares):**

$$\mathcal{L}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^\top (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

**Expand:**

$$\mathcal{L} = \mathbf{y}^\top\mathbf{y} - 2\boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{y} + \boldsymbol{\beta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta}$$

**Take gradient and set to zero:**

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^\top\mathbf{y} + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

**Normal equations:**

$$\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}$$

**Closed-form solution (when $\mathbf{X}^\top\mathbf{X}$ is invertible):**

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

**Variance of estimator:**

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X})^{-1}$$

---

### Why $(\mathbf{X}^\top\mathbf{X})^{-1}$ Fails in Finance

```
PROBLEM                  CAUSE                          SOLUTION
──────────────────────   ─────────────────────────────  ──────────────────────────────
Perfect multicollinearity  Correlated factors (e.g.,    Ridge: (X'X + λI)⁻¹X'y
                           value + quality signals)
Near-collinearity          High pairwise IC in library  PCA or Lasso to select
Fat tails in ε             Returns ≠ Gaussian            Quantile regression or robust regression
Non-stationary X           Trending factor exposures    First-difference or WLS
```

> "In practice I never use OLS for live signal regression. I use Ridge with cross-validated λ:
> it shrinks the coefficient estimates toward zero, which dramatically reduces the overfitting
> problem when I have 50+ correlated features. The cost is a slight bias increase, but the
> variance reduction more than compensates when T/K < 20."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q12 · How Do You Prevent Overfitting in an ML Model for Financial Data?

**Open with the intuition (15 seconds):**
> "Financial time series has three properties that make overfitting uniquely dangerous: low signal-to-noise
> ratio (IC ≈ 0.03–0.08), non-stationarity, and serial dependence. Standard CV techniques are invalid.
> The anti-overfitting toolkit for quant finance has three pillars: data hygiene, correct cross-validation,
> and parsimony priors on model complexity."

---

### Anti-Overfitting Toolkit

**1. Correct Time Series Cross-Validation:**

Use **Purged Walk-Forward** or **CPCV** — never standard k-fold:
```
PURGED WALK-FORWARD CV

  FOLD   TRAIN           GAP      TEST
  ────   ─────────────   ──────   ──────────
  1      [T0, T1]        21 days  [T1+21d, T2]
  2      [T0, T2]        21 days  [T2+21d, T3]
  ...
  K      [T0, TK-1]      21 days  [TK-1+21d, TK]

  Gap prevents labels computed from overlapping windows
  from leaking between train and test.
```

**2. Regularization:**
- L2 (Ridge): penalizes $\|\boldsymbol{\beta}\|^2$ — shrinks all coefficients
- L1 (Lasso): penalizes $\|\boldsymbol{\beta}\|_1$ — zeros out irrelevant features
- ElasticNet: $\alpha \|\boldsymbol{\beta}\|_1 + (1-\alpha)\|\boldsymbol{\beta}\|^2$

**3. Feature Count Control:**

```
RULE OF THUMB: T / K ratio (observations to features)

  T/K < 10    Extremely dangerous. Regularize heavily.
  T/K 10–30   Regularize moderately. Use Ridge/Lasso.
  T/K > 30    Standard ML acceptable. Still use CV.
```

**4. Feature Importance Stability:**
Check that SHAP value rankings are stable across CV folds. If the top-3 features rotate
completely across folds, the model has no stable signal.

**5. Deflated Sharpe Ratio:**
After $M$ trials, require DSR > 1.0 before declaring signal genuine.

> "My workflow: I set a hard cap of 30 features per model, enforce CPCV with 6 folds and a
> 21-day gap, and require that the OOS Sharpe is > 0.5 AND the IS/OOS ratio < 2.0 before
> a signal is promoted. I've rejected signals with IS Sharpe of 4.2 because the IS/OOS ratio
> was 6×. Those signals consistently underperformed in live trading."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q13 · Compare Lasso vs Ridge Regularization — When Do You Use Each?

**Open with the intuition (15 seconds):**
> "Ridge and Lasso solve the same overfitting problem with different assumptions about the feature
> space. Ridge assumes all features are informative to some small degree. Lasso assumes only a
> sparse subset of features are truly informative and the rest should be zeroed out. In a quant
> library of 200 candidate features, Lasso is almost always the right starting point."

---

### Mathematical Comparison

**Ridge (L2) — minimizes:**
$$\mathcal{L}_{\text{Ridge}} = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$$

**Closed-form solution:**
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

**Lasso (L1) — minimizes:**
$$\mathcal{L}_{\text{Lasso}} = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1$$

No closed form — requires coordinate descent or LARS algorithm.

```
PROPERTY                RIDGE               LASSO
──────────────────────  ─────────────────   ──────────────────────────────
Coefficients at λ→∞    → 0 smoothly        → 0 with exact sparsity
Feature selection       No (shrinks all)    Yes (sets subsets to zero)
Correlated features     Shares weight       Selects one arbitrarily
Computational cost      O(K³) closed form   Iterative (coordinate descent)
When to use             All features valid  Many irrelevant features
Finance use case        Factor model        Signal library selection
```

**Elastic Net — best of both:**
$$\mathcal{L}_{\text{EN}} = \|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|^2$$

Handles correlated features (Ridge component) while achieving sparsity (Lasso component).

> "For a 200-signal library I ran ElasticNet with α=0.7 (70% L1 character). The model
> selected 35 signals out of 200. I then ran Ridge on just those 35 to get stable coefficient
> estimates without Lasso's arbitrary choice between correlated signals. This two-stage approach
> gave me better OOS IR than either Lasso or Ridge alone."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q14 · How Does Gradient Boosting Work and Why Is It Effective for Alpha Research?

**Open with the intuition (15 seconds):**
> "Gradient boosting builds an ensemble of weak learners — typically shallow decision trees —
> where each new tree fits the residual errors of the ensemble so far. It's sequentially additive,
> not independently parallel like Random Forest. In financial data, gradient boosting (LightGBM
> specifically) handles non-linear feature interactions, mixed data types, and sparse alt-data
> features better than any parametric model — which is why it dominates Kaggle-style alpha challenges."

---

### Gradient Boosting Algorithm

Initialize: $F_0(\mathbf{x}) = \arg\min_\gamma \sum_i L(y_i, \gamma)$

For $m = 1, 2, \ldots, M$:

**Step 1 — Compute pseudo-residuals:**
$$r_{im} = -\left[\frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F = F_{m-1}}$$

**Step 2 — Fit a base learner** $h_m(\mathbf{x})$ to residuals $\{r_{im}\}$

**Step 3 — Line search for step size:**
$$\gamma_m = \arg\min_\gamma \sum_i L(y_i,\; F_{m-1}(\mathbf{x}_i) + \gamma h_m(\mathbf{x}_i))$$

**Step 4 — Update model:**
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \cdot \gamma_m h_m(\mathbf{x})$$

Where $\nu \in (0,1]$ is the learning rate (shrinkage parameter).

---

### Why LightGBM Dominates for Alpha Research

```
ADVANTAGE                    DETAIL
───────────────────────────  ──────────────────────────────────────────────────
Histogram-based splits        O(N) vs O(N log N) — critical for large universes
Leaf-wise tree growth         Deeper asymmetric trees → captures non-linearities
Native missing value handling  Alt data often has gaps — no imputation needed
DART dropout variant          Prevents overfitting in boosting iterations
GPU acceleration              Train on 10yr daily data in minutes, not hours
Category features native       Sector/country encoding without one-hot expansion
```

> "I use LightGBM as the primary model for next-5-day cross-sectional return prediction.
> Features: 60+ signals across momentum, mean-reversion, quality, alt-data, and microstructure.
> I run CPCV with 8 folds, tune learning rate and num_leaves jointly via Optuna with a 20-trial
> budget. The model consistently achieved OOS IC of 0.05–0.07 on large-cap US equities.
> I use SHAP waterfall plots in every model review to verify that the feature importance
> structure is economically sensible — if a synthetic noise feature ranks in the top 10,
> the model goes back to training."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q15 · Explain SHAP Values and How You Use Them in Practice

**Open with the intuition (15 seconds):**
> "SHAP — SHapley Additive exPlanations — answers the question: for this specific prediction on
> this specific asset on this specific date, how much did each feature contribute? It's game theory
> applied to ML interpretability. For a quant researcher, SHAP is how you convince a CIO that a
> black-box model is actually doing something economically sensible and not just pattern-matching noise."

---

### SHAP Value Definition

For a prediction $f(\mathbf{x})$, the Shapley value for feature $j$ is:

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} \left[f_{S\cup\{j\}}(\mathbf{x}) - f_S(\mathbf{x})\right]$$

Where $\mathcal{F}$ is the full feature set and $S$ is a subset excluding feature $j$.

**Additive property (key):**

$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{K} \phi_j(\mathbf{x})$$

The prediction decomposes exactly into a sum of feature contributions plus baseline.

---

### Practical SHAP Usage in Alpha Research

```
USE CASE                    SHAP TOOL                   WHAT TO LOOK FOR
──────────────────────────  ──────────────────────────  ────────────────────────────────────────
Feature importance          Mean |SHAP| across universe  Stable ranking across CV folds?
Feature interaction         SHAP interaction values      Does momentum × sector matter?
Single prediction debug     Waterfall plot               Which features drove today's signal?
Feature drift monitoring    Rolling mean SHAP over time  Are features' contributions stable?
Regime analysis             Group by VIX regime          Does feature importance change in stress?
```

```
SHAP WATERFALL EXAMPLE (single stock, next-5d return prediction)

  Base value:  +0.001  (mean return of universe)
  + MOM_12m:   +0.008  (strong trailing momentum)
  + EARN_REV:  +0.005  (recent positive estimate revision)
  − VOL_5d:    −0.003  (elevated recent realized vol)
  + QUALITY:   +0.002  (strong ROE, low leverage)
  − CROWD:     −0.004  (high hedge fund overlap → reversal risk)
  ─────────────────────────────────────────────
  Final pred:  +0.009  (9 bps expected 5-day alpha)
```

> "I run a SHAP review in every monthly model validation meeting. The top-5 features by mean
> |SHAP| must include at least 3 features with clear economic rationale. If a data field I
> can't explain — like the hash of a ticker symbol — shows up in the top features, that's a
> data leakage flag. I've caught three leakage incidents in production using this method."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q16 · How Do You Build an ML Pipeline for Predicting Next-Day Stock Returns?

**Open with the intuition (15 seconds):**
> "A production ML pipeline for daily return prediction has nine distinct steps, each with a
> common failure mode. Getting features right matters more than model architecture — garbage in,
> garbage out. The feature engineering step is where 80% of the alpha lives."

---

### End-to-End ML Pipeline

```
STEP    COMPONENT                   KEY IMPLEMENTATION NOTE
──────  ──────────────────────────  ─────────────────────────────────────────────────────
1       Universe Construction       Liquid stocks (ADV > $5M), survivorship-free, sector map
2       Return Label Computation    Forward 1-day raw return, winsorize at 5σ
3       Feature Engineering         60+ signals: MOM, REV, VOL, QUAL, ALT, MICRO
4       Point-in-Time Alignment     Every feature lagged 1 day; alt-data lagged by filing delay
5       Cross-Sectional Normalization Z-score within GICS sector per day, winsorize at 3σ
6       Train/Test Split            CPCV: 8-fold walk-forward with 21-day gap/embargo
7       Model Training              LightGBM with Optuna hyperparameter optimization
8       Prediction                  Cross-sectional rank of predicted returns
9       Portfolio Construction      Long top quintile, short bottom quintile, dollar-neutral
```

**Feature Engineering Categories:**

```
CATEGORY        EXAMPLES                              HORIZON
─────────────   ────────────────────────────────────  ─────────
Price signals   MOM_1m, MOM_3m, MOM_12m, RVOL, beta  1–60 days
Quality         ROE, D/E, earnings stability           Slow
Revisions       EPS revision 1m, 3m; PO changes       10–30 days
Microstructure  OFI_1d, bid-ask spread, short int     1–5 days
Alt data        CC transactions, job postings, NLP     30–90 days
Cross-asset     FX carry, rates vol, credit spread     5–20 days
```

> "In the feature importance check: I verify that features computed from future data
> cannot appear in the training set. I run a 'time-travel' test: train on the full
> history, then check if any feature has a correlation > 0.3 with next-day return on
> the test set that is *larger* than the 5-day lagged correlation. If yes, look-ahead
> bias is present. I run this test every time a new data provider is onboarded."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q17 · What Is a Random Forest and How Does It Reduce Variance?

**Open with the intuition (15 seconds):**
> "A random forest is an ensemble of decorrelated decision trees. The key insight is that
> averaging independent noisy estimators reduces variance without increasing bias. Two sources
> of randomness are injected deliberately: bootstrap sampling of observations (bagging) and
> random feature subsampling at each split. The decorrelation is what makes it more powerful
> than simply averaging trees trained on the same data."

---

### Bias-Variance Decomposition

For a single tree: $\text{MSE} = \text{Bias}^2 + \text{Variance}$. Deep trees → low bias, high variance.

For an ensemble of $B$ trees, each with variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}(\bar{f}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$: $\text{Var}(\bar{f}) \to \rho\sigma^2$.

**The irreducible floor is determined by $\rho$, not $B$.**
This is why random feature subsampling (reducing $\rho$) matters more than adding more trees.

---

### Random Forest vs Gradient Boosting in Finance

```
PROPERTY              RANDOM FOREST         GRADIENT BOOSTING (LightGBM)
────────────────────  ────────────────────  ──────────────────────────────────
Parallelizable?       Yes (trees are indep) No (sequential)
Handles noise?        Good (averaging)      More sensitive to noise
Overfitting risk?     Low                  Higher (needs tuning)
Non-linear capture?   Moderate              Higher
Speed (large N)?      Moderate              Fast (histogram)
OOS IR in practice?   0.03–0.05 IC          0.05–0.08 IC (finance tasks)
```

> "I use Random Forest as my baseline model and LightGBM as my production model.
> If LightGBM doesn't beat RF by >15% in OOS IC, I default to RF — it's more robust
> to parameter sensitivity and easier to explain to portfolio managers. The governance
> argument matters: regulators and risk committees want to understand what the model does."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q18 · How Would You Use a Neural Network for Time Series Alpha Research?

**Open with the intuition (15 seconds):**
> "Neural networks in quant research are powerful but dangerous. The ratio of parameters to
> training observations in finance is typically 100–1000×, which means without aggressive
> regularization you are fitting noise. The use cases where NNs genuinely add value over
> gradient boosting: (1) unstructured data like text or satellite images, (2) very long
> sequence dependencies where LSTM/Transformer captures structure, (3) multi-task learning
> across asset classes."

---

### Practical NN Architectures for Finance

```
USE CASE                     ARCHITECTURE         KEY HYPERPARAMETER
───────────────────────────  ──────────────────   ────────────────────────────────────
Cross-sectional prediction   MLP (3-5 layers)     Dropout 0.3, batch norm, tanh
Sequential pattern           LSTM / TCN           Sequence len 63, hidden 128
Text/NLP alpha               Transformer (BERT)   Fine-tune on 8-K / earnings calls
Multi-asset prediction       Shared encoder +     Task-specific heads per asset class
                             task heads
```

**Critical regularization for financial NNs:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_1\|\mathbf{W}\|_1 + \lambda_2\|\mathbf{W}\|^2 + \mathcal{L}_{\text{dropout}}$$

**Walk-forward training protocol for RNNs:**

Avoid standard early stopping on a random val set — it creates look-ahead.
Use **temporal validation**: the last 10% of time is validation, not a random 10%.

> "I built an LSTM for earnings call sentiment extraction. Input: 3-year history of
> earnings transcripts (8-K filings), tokenized with FinBERT. Output: 5-day post-earnings
> return direction. The model achieved 58% directional accuracy OOS on large-cap
> US equities, vs 52% for a simple sentiment polarity baseline. That 6% lift translated
> to IC ≈ 0.05, comparable to our best traditional signals but orthogonal to them."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q19 · How Do You Evaluate ML Model Performance in a Financial Context?

**Open with the intuition (15 seconds):**
> "Standard ML metrics — RMSE, AUC, accuracy — are poor proxies for financial value. A model
> can have low RMSE but garbage IC if it's very accurate on the wrong part of the distribution.
> The relevant metric hierarchy is: IC first, then turnover-adjusted IR, then realized live
> Sharpe. Everything else is noise."

---

### Financial ML Evaluation Stack

```
METRIC              DEFINITION                              TARGET
──────────────────  ──────────────────────────────────────  ─────────────────────
Rank IC             Spearman corr(prediction, forward ret)  > 0.04
ICIR                Mean IC / Std IC                        > 0.8
Turnover            Daily 1-way turnover %                  < 20% (cost-aware)
Sharpe (OOS)        Mean / Std of daily portfolio returns   > 1.0 annualized
Max Drawdown        Peak-to-trough of OOS equity curve      < −15%
Hit Rate            % days with positive portfolio return   > 54%
IS/OOS SR ratio     IS Sharpe / OOS Sharpe                  < 2.0
Deflated SR (DSR)   DSR after M-trial penalty               > 1.0
```

**Quintile backtest — the gold standard visualization:**

```
QUINTILE    MEAN FORWARD RETURN    DESIRED PATTERN
─────────   ────────────────────   ──────────────────────────────────────
Q5 (Long)   highest                Monotonically decreasing from Q5 to Q1
Q4
Q3
Q2
Q1 (Short)  lowest
```

If the quintile spread is non-monotonic (Q3 > Q4), the model has noise in the middle of
the distribution — likely still tradeable but with higher tracking error.

> "Beyond these metrics, I always run a 'cost-sensitive IR': I deduct realistic transaction
> costs (market impact + bid-ask + commissions) from the signal's gross alpha at each
> turnover level. If net-of-cost IC drops below zero, the signal is not tradeable at scale
> regardless of how good it looks gross."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q20 · What Is Feature Importance and How Do You Avoid the Correlation Trap?

**Open with the intuition (15 seconds):**
> "The correlation trap is the most common mistake in feature importance analysis: when two
> features are highly correlated, tree-based importance metrics assign arbitrarily low importance
> to one and high to the other based on which split was encountered first. This makes the
> importance metric unstable and misleading. SHAP solves this properly; gain-based importance does not."

---

### Feature Importance Methods Comparison

```
METHOD                      CORRELATION HANDLING    STABILITY    USE CASE
──────────────────────────  ──────────────────────  ───────────  ──────────────────────
Split count (Gini)          POOR — arbitrary split  Low          Never use alone
Gain-based (LightGBM)       POOR — assigns to 1st   Moderate     Directional signal only
Permutation importance      BETTER — tests each     High         Good for validation
SHAP values                 BEST — game-theoretic   Highest      Production standard
MDA (Mean Decrease Acc.)    GOOD — randomizes feat  High         Tree-specific
```

**To detect and handle the correlation trap:**

1. Compute pairwise Pearson correlation matrix of all features.
2. For any pair with $|\rho| > 0.70$: inspect their SHAP values.
3. If SHAP assigns near-zero to one member of the pair: run the model without the high-SHAP member
   and see if the low-SHAP member's contribution rises. If yes: both carry equivalent information.
4. Select the one with better point-in-time availability and lower data risk.

> "I had two signals: 6-month momentum and 9-month momentum, correlated at 0.87.
> LightGBM gain-importance showed 6m MOM at rank #2, 9m MOM at rank #15.
> SHAP showed them at almost identical mean absolute values. I dropped 9m MOM from the
> feature set — it added zero marginal information and slowed training. The model's OOS IC
> was unchanged to 3 decimal places."

[🔝 Back to Top](#-table-of-contents)

---
---

# 📊 STATISTICS & PROBABILITY

---

## Q21 · Derive the Formula for Covariance and Correlation from Scratch

**Open with the intuition (15 seconds):**
> "Covariance measures the degree to which two variables move together in an absolute sense.
> Correlation is covariance normalized by the product of standard deviations — it's the
> unit-free version. In financial models, we almost always want correlation, not covariance,
> because covariance conflates co-movement with individual volatility magnitudes."

---

### Derivations

**Population covariance:**

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mu_X \mu_Y$$

**Sample covariance (unbiased, Bessel correction):**

$$\hat{\sigma}_{XY} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Why $n-1$ not $n$?** We've used one degree of freedom to estimate the sample mean $\bar{x}$, $\bar{y}$. Dividing by $n$ produces a biased (downward) estimate of the population covariance. The $n-1$ denominator gives an unbiased estimator.

**Population correlation (Pearson):**

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} \in [-1, +1]$$

**Sample correlation:**

$$\hat{\rho}_{XY} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i(x_i-\bar{x})^2 \cdot \sum_i(y_i-\bar{y})^2}}$$

**Spearman rank correlation:** Replace $x_i, y_i$ with their ranks $R(x_i), R(y_i)$ before applying Pearson formula. Robust to outliers — critical for financial returns.

---

### Matrix Form (for Signal Library)

For a matrix of $N$ signals $\mathbf{F} \in \mathbb{R}^{T \times N}$, the sample covariance matrix:

$$\mathbf{\Sigma} = \frac{1}{T-1}(\mathbf{F} - \bar{\mathbf{F}})^\top(\mathbf{F} - \bar{\mathbf{F}})$$

Correlation matrix: $\mathbf{R}_{ij} = \mathbf{\Sigma}_{ij} / \sqrt{\mathbf{\Sigma}_{ii}\mathbf{\Sigma}_{jj}}$

> "I use the covariance matrix in two places in production: (1) optimal signal combination
> weights via $\hat{\boldsymbol{w}} = \mathbf{\Sigma}^{-1}\boldsymbol{\mu}$, and (2) portfolio risk management
> via the covariance of factor exposures. In both cases I use Ledoit-Wolf shrinkage because
> raw sample covariance matrices are notoriously noisy with a large number of assets."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q22 · What Is a Markov Chain and How Would You Apply It to Markets?

**Open with the intuition (15 seconds):**
> "A Markov chain is a stochastic process where the probability of transitioning to the next
> state depends only on the current state, not on the history of past states. This memoryless
> property — the Markov property — simplifies analysis enormously. In markets, Hidden Markov
> Models extend this: the observed returns are emissions from a hidden latent state (bull, bear, crisis)
> that we can infer but not observe directly."

---

### Formal Definition

A discrete-time Markov chain on state space $\mathcal{S} = \{s_1, \ldots, s_K\}$:

$$P(X_{t+1} = s_j \mid X_t = s_i, X_{t-1} = s_{i-1}, \ldots) = P(X_{t+1} = s_j \mid X_t = s_i) = p_{ij}$$

**Transition matrix:**

$$\mathbf{P} = \begin{pmatrix} p_{11} & p_{12} & \cdots & p_{1K} \\ p_{21} & p_{22} & \cdots & p_{2K} \\ \vdots & & \ddots & \vdots \\ p_{K1} & p_{K2} & \cdots & p_{KK} \end{pmatrix}, \quad \sum_j p_{ij} = 1 \; \forall i$$

**Stationary distribution $\pi$:** satisfies $\pi^\top \mathbf{P} = \pi^\top$

The $n$-step transition probability: $\mathbf{P}^n \to \mathbf{1}\pi^\top$ as $n \to \infty$ (for ergodic chains).

---

### Market Application — Hidden Markov Model (HMM)

```
HIDDEN STATES       OBSERVABLE EMISSIONS (RETURNS)
─────────────────   ────────────────────────────────────────────
State 1: Bull       μ₁ > 0, σ₁ low   (trend, low volatility)
State 2: Bear       μ₂ < 0, σ₂ mid   (drawdown)
State 3: Crisis     μ₃ << 0, σ₃ high (fat tails, jumps)

Transition matrix example:
          Bull    Bear    Crisis
Bull    [ 0.95   0.04    0.01  ]
Bear    [ 0.10   0.85    0.05  ]
Crisis  [ 0.05   0.25    0.70  ]
```

**Viterbi algorithm:** finds the most likely state sequence given observed returns.
**Baum-Welch (EM):** estimates $\mathbf{P}$ and emission parameters from data.

**Practical use:**
- When posterior probability of Crisis state exceeds 0.6: reduce gross exposure by 50%.
- Regime label as a conditioning variable in cross-sectional signal weighting.

> "I fit a 3-state Gaussian HMM on daily returns of SPX using scikit-learn's GaussianHMM.
> The states naturally aligned with calm/trending/crisis regimes. Using the state posterior
> as a feature in our LightGBM alpha model improved OOS IC by 0.008 on its own — equivalent
> to adding 5 traditional price signals. The intuition: momentum signals perform very differently
> in bull vs crisis regimes, and the HMM tells the model which regime it's in."

[🔝 Back to Top](#-table-of-contents)

---
---

## Q23 · Walk Me Through the Central Limit Theorem and Its Limitations in Finance

**Open with the intuition (15 seconds):**
> "The CLT is the backbone of classical statistics: averages of IID random variables converge
> to a Normal distribution regardless of the underlying distribution. The problem in finance:
> returns are NOT IID. They have fat tails (kurtosis >> 3), serial correlation, and volatility
> clustering. The CLT still applies asymptotically, but requires much larger samples than
> normal approximations assume."

---

### CLT — Formal Statement

Let $X_1, X_2, \ldots, X_n$ be IID with $\mathbb{E}[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$.

$$\sqrt{n}\left(\bar{X}_n - \mu\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad \text{as } n \to \infty$$

**Equivalently:** $\bar{X}_n \approx \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$ for large $n$.

---

### CLT Violations in Financial Returns

```
VIOLATION               WHAT IT MEANS                   CONSEQUENCE
──────────────────────  ──────────────────────────────  ──────────────────────────────────
Fat tails (kurtosis>3)  Extreme returns more likely     t-tests underestimate p-values
Serial correlation      Returns NOT independent         Standard errors underestimated
Volatility clustering   $\sigma_t$ time-varying         CLT convergence much slower
Non-stationarity        $\mu_t$ changes over time       OLS estimates unstable
Jumps (crash returns)   Discontinuous distribution      Bootstrap required, not CLT
```

**Fix 1 — Newey-West HAC Standard Errors:**
$$\text{Var}_{\text{HAC}}(\hat{\beta}) = (\mathbf{X}^\top\mathbf{X})^{-1} \hat{\mathbf{\Omega}}_{\text{NW}} (\mathbf{X}^\top\mathbf{X})^{-1}$$

Accounts for serial correlation in residuals; essential for finance t-statistics.

**Fix 2 — Bootstrap:**
Resample from historical return blocks (block bootstrap, preserves autocorrelation) rather
than assuming CLT-based Normal approximation.

> "When I report t-statistics on IC, I always use Newey-West HAC errors with lag = $\lfloor 4(T/100)^{2/9}\rfloor$.
> For a 252-observation sample this gives lag ≈ 3. Standard OLS t-stats on IC overstate significance
> by ~25% when there's autocorrelation in the IC series, which is almost always present in momentum signals."

[🔝 Back to Top](#-table-of-contents)

---
---

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

[🔝 Back to Top](#-table-of-contents)

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

$$k = \max\!\left\{i : p_{(i)} \leq \frac{i}{N} \cdot \frac{q}{\sum_{j=1}^{N}(1/j)}\right\}$$

**Step 3 — Minimum Backtest Length:**

$$T_{\min} = \frac{(Z_{1-\alpha/N})^2 \cdot \sigma_r^2}{\hat{\mu}_r^2}$$

Where $\hat{\mu}_r$ = claimed mean return, $\sigma_r$ = return volatility.

> "I enforce a hard rule: any signal tested after the 20th trial in a research campaign
> must have $|t| > 3.5$ to be considered for inclusion, regardless of how compelling
> the economic story is. When I joined my prior fund, they had a backtest library of
> 600+ signals, all tested against the same historical data. Every 'significant' signal
> at $|t| > 2.0$ was re-evaluated under BHY correction — only 30% survived."

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

$$\widehat{PSR}(SR^*) = \Phi\!\left(\frac{(\hat{SR} - SR^*)\sqrt{T-1}}{\sqrt{1 - \hat{\gamma}_3 \hat{SR} + \frac{\hat{\gamma}_4-1}{4}\hat{SR}^2}}\right)$$

Where:
- $SR^*$ = benchmark Sharpe ratio (typically 0 or the risk-free Sharpe)
- $\hat{\gamma}_3$ = skewness of returns
- $\hat{\gamma}_4$ = excess kurtosis of returns
- $T$ = number of observations

**Deflated Sharpe Ratio (DSR):**

The benchmark $SR^*$ is set to the expected maximum Sharpe from $N$ IID trials:

$$SR^*_{\max} \approx \left(1 - \gamma\right)\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)$$

Where $\gamma \approx 0.5772$ (Euler-Mascheroni constant).

$$DSR = \widehat{PSR}(SR^*_{\max})$$

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

$$\max_{\mathbf{w}} \; \boldsymbol{\mu}^\top \mathbf{w} - \frac{\lambda}{2} \mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}$$

**Closed-form (unconstrained):**
$$\mathbf{w}^* = \frac{1}{\lambda} \mathbf{\Sigma}^{-1} \boldsymbol{\mu}$$

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
$$\hat{\Sigma}^{\text{LW}} = \alpha \hat{\Sigma}_{\text{sample}} + (1-\alpha)\hat{\mu}_{var} \mathbf{I}$$

Where $\alpha$ is chosen analytically to minimize expected squared Frobenius loss.

**Black-Litterman (production fix for μ):**
Combine market-implied equilibrium returns $\boldsymbol{\Pi} = \lambda \mathbf{\Sigma} \mathbf{w}^{\text{mkt}}$ with model views:
$$\hat{\boldsymbol{\mu}} = \left[(\tau\mathbf{\Sigma})^{-1} + \mathbf{P}^\top\mathbf{\Omega}^{-1}\mathbf{P}\right]^{-1}\left[(\tau\mathbf{\Sigma})^{-1}\boldsymbol{\Pi} + \mathbf{P}^\top\mathbf{\Omega}^{-1}\mathbf{q}\right]$$

> "I never use raw MVO in production. My workflow: (1) Black-Litterman expected returns from
> the alpha signal IC-weighted views, (2) Ledoit-Wolf shrunk covariance, (3) constrained QP
> with position limits, sector neutrality, and a turnover penalty of $\delta\|\mathbf{w}_t - \mathbf{w}_{t-1}\|^2$.
> The turnover penalty alone reduced annual transaction costs by 35% vs unconstrained MVO."

[🔝 Back to Top](#-table-of-contents)

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

$$f^* = \frac{pb - (1-p)}{b} = \frac{p(b+1) - 1}{b}$$

**Continuous case (normal returns $r \sim \mathcal{N}(\mu, \sigma^2)$):**

$$f^* = \frac{\mu}{\sigma^2}$$

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

[🔝 Back to Top](#-table-of-contents)

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

$$\mathbf{w}^* = \hat{\mathbf{\Sigma}}^{-1\text{LW}} \cdot \boldsymbol{\mu}_{\text{EB}}$$

Normalize: $\tilde{\mathbf{w}} = \mathbf{w}^* / \|\mathbf{w}^*\|_1$ (sum-to-one constraint)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

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

[🔝 Back to Top](#-table-of-contents)

---
---

# Quick-Reference Equation Sheet

| Symbol | Definition |
|--------|------------|
| $IC$ | Information Coefficient: $\text{Corr}(f_{i,t}, r_{i,t+h})$ |
| $ICIR$ | $\bar{IC} / \sigma_{IC}$ |
| $IR$ | Information Ratio: $\bar{r}_{\text{active}} / \sigma_{\text{active}}$ |
| $SR$ | Sharpe Ratio: $(\bar{r} - r_f) / \sigma_r$ |
| $DSR$ | Deflated Sharpe: $\widehat{PSR}(SR^*_{\max})$ adjusted for $N$ trials |
| $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ | $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| $\hat{\boldsymbol{\beta}}_{\text{Ridge}}$ | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| $\mathbf{w}^*_{\text{Mkt-Neutral}}$ | $\mathbf{\Sigma}^{-1}\boldsymbol{\alpha}$ s.t. $\mathbf{1}^\top\mathbf{w}=0$, $\boldsymbol{\beta}^\top\mathbf{w}=0$ |
| $f^*_{\text{Kelly}}$ | $\mu / \sigma^2$ (continuous) |
| $\sigma_t^2_{\text{GARCH}}$ | $\omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$ |
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
