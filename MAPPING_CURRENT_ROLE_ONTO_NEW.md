# 🏦 Balyasny Experience → Trexquant JD Requirements Mapping
> **Candidate:** Shaikat Majumdar · Senior Quantitative Researcher, Balyasny Asset Management LP
> **Target Role:** Quantitative Researcher — Experienced Hire · Trexquant Investment LP
> **Purpose:** Demonstrate that every Trexquant responsibility is work already executed in production at Balyasny — with specificity, conviction, and depth.

---
---

---
---

## 📑 Table of Contents

- **[🏗️ Context: How Balyasny Systematic Is Structured](#️-context-how-balyasny-systematic-is-structured)**
- **[✅ JD Requirement → Balyasny Proof Point](#-jd-requirement--balyasny-proof-point)**
  - [📌 Responsibility 1 - **Machine Learning & Asset Prediction**](#-responsibility-1)
  - [📌 Responsibility 2 - **Data Parsing & Alpha Signal Generation**](#-responsibility-2)
  - [📌 Responsibility 3 - **Academic Research & Implementation**](#-responsibility-3)
  - [📌 Responsibility 4 - **Model Innovation & Scalability**](#-responsibility-4)
  - [📌 Responsibility 5 - **Team Collaboration & Strategy Backtesting**](#-responsibility-5)
- **[📋 Requirements Checklist](#-requirements-checklist)**
- **[💬 Closing Statement for the Trexquant Interview Panel](#-closing-statement-for-the-trexquant-interview-panel)**

[🔝 Back to Top](#-table-of-contents)

---
---

## 🏗️ Context: How Balyasny Systematic Is Structured

Before mapping to JD requirements, it is worth establishing what "Balyasny Systematic" actually is — because it is not a single-pod operation. It is a **large, central systematic research and trading engine of 100+ people** with two primary mandates:

```
┌─────────────────────────────────────────────────────────┐
│                  BALY SYSTEMATIC                        │
│                                                         │
│  ┌───────────────────┐   ┌──────────────────────────┐   │
│  │  PROPRIETARY BOOKS│   │   EXECUTION SERVICES     │   │
│  │                   │   │                          │   │
│  │  1. Quant/Cash EQ │   │  • Algos & Smart Order   │   │
│  │  2. Macro Books   │   │  • Transaction Cost Model│   │
│  │  3. Replication   │   │    (Central Risk Book)   │   │
│  │   (Alpha-Capture  │   │    (EQ + Macro)          │   │
│  │    of Disc. PMs)  │   │                          │   │
│  └───────────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

I sit on the **Macro Proprietary Book** — developing systematic investment models across Equities, Fixed Income, FX, Commodities, and Derivatives. But I operate with full awareness of the platform's broader architecture: how my signals interact with the Central Risk Book, how they may be consumed by the Replication layer, and how Execution Services TCM benchmarks constrain what counts as "real" alpha after costs.

This is not a narrow pod. It is a research environment that **demands signals be defensible at every layer** — statistical, economic, and execution.

[🔝 Back to Top](#-table-of-contents)

---

## ✅ JD Requirement → Balyasny Proof Point

[🔝 Back to Top](#-table-of-contents)

---

### 📌 Responsibility 1
**"Design, implement, and optimize various machine learning models aimed at predicting liquid assets using a wide set of financial data and a vast library of trading signals."**

#### 🔵 What I did at Balyasny

At Balyasny, I designed and implemented a **multi-model ensemble architecture** for predicting short-to-medium horizon returns across liquid global macro futures and FX pairs. The architecture was not a single model — it was a structured hierarchy of ML models, each targeting a distinct predictive regime.

**Concretely:**

- **Gradient Boosting (XGBoost/LightGBM):** I trained cross-sectional GBM models on a feature matrix spanning carry differentials, trend momentum, realized volatility ratios, cross-asset correlation breaks, and alternative data proxies (sentiment-adjusted positioning from COT reports). The models were trained on rolling 5-year windows with quarterly refit cycles, and I used Bayesian hyperparameter optimization (Optuna) to prevent grid-search overfitting during the tuning phase.

- **Random Forests for Feature Importance Auditing:** Before committing a new feature to the live signal library, I ran RF-based permutation importance and SHAP decomposition to confirm the feature was contributing genuine marginal predictive lift — not proxying for an existing signal in the library. This prevented signal redundancy at the source.

- **NLP on Central Bank Communications:** I applied transformer-based NLP (fine-tuned FinBERT variants) to FOMC statements, ECB press conferences, and BoJ Summary of Opinions to extract forward-guidance sentiment scores. These scores fed directly into rate-sensitive FX and Fixed Income signal components, improving next-week directional accuracy on USD pairs by a statistically significant margin (validated out-of-sample over a 12-month holdout period).

- **Hidden Markov Models (HMM) for Regime-Conditional Optimization:** Rather than running a single global ML model, I used a 3-state HMM (Risk-On / Risk-Off / Transition) trained on VIX, cross-asset correlation, and credit spread dynamics to route signal weights dynamically. In Risk-Off regimes, trend models received higher allocation; in Risk-On, carry and value signals dominated. This conditional weighting was the single largest contributor to live Sharpe stability across the 2025 vol regime shifts.

**Optimization layer:** I implemented signal-level and portfolio-level optimization iteratively — not as a one-time calibration. Every quarter, I ran a full out-of-sample attribution decomposition (signal IC, IC-IR, decay half-life) and updated model weights accordingly. Models that showed IC decay below the 40th percentile of their historical distribution were either recalibrated or retired.

[🔝 Back to Top](#-table-of-contents)

---

### 📌 Responsibility 2
**"Parse and analyze large datasets to identify actionable alpha signals and develop strategies for systematic trading."**

#### 🔵 What I did at Balyasny

Large-scale dataset analysis was not a support function at Balyasny — it was the **primary research activity**. The Macro Systematic book operates across thousands of instruments and dozens of data vendors simultaneously. My workflow:

**Dataset Coverage I Worked With Directly:**

| Data Category | Source Type | Signal Extracted |
|---|---|---|
| **Price / OHLCV** | Exchange feeds, vendor consolidated | Trend, mean-reversion, breakout signals |
| **Order Flow** | Consolidated tape, dark pool prints | Adverse selection / informed flow detection |
| **Volatility Surface** | Options chains (equity + rates) | IV/RV spread signals, vol risk premium |
| **COT / Positioning** | CFTC Commitments of Traders | Crowding proxies, contrarian positioning |
| **Macro Fundamentals** | Bloomberg, FactSet | Carry differentials, value z-scores |
| **Cross-Asset Microstructure** | Correlated asset tick data | Lead-lag arbitrage, correlation regime signals |
| **Alternative / NLP** | CB transcripts, earnings call audio | Sentiment-adjusted directional signals |

**Example — Cross-Asset Microstructure Signal:**

I identified a lead-lag relationship between moves in the front-month WTI Crude futures contract and lagged responses in Energy sector equities (XLE, individual E&P names). The raw relationship was well-known and arbitraged away at the second-to-second level. The alpha resided in **regime-conditional persistence**: during low-liquidity intraday windows (first 15 minutes, last 30 minutes of RTH), the lead-lag reversion window extended to 3–8 minutes rather than the typical sub-minute. I extracted this as a features vector (lag-conditioned cross-asset momentum + microstructure noise ratio) and validated it using CPCV with 6-fold combinatorial splits. The signal had a live IC of ~0.05 — uncorrelated to the existing library — and was approved for capital allocation within the Macro book.

**Example — Replication Layer Interaction:**

One of Balyasny Systematic's mandates is alpha-capture replication of discretionary PMs — systematically extracting the *implicit rules* from observed PM behavior. I contributed to this by reverse-engineering position change patterns from a macro discretionary PM's book, identifying that their sizing behavior exhibited a strong regime-conditional signal: they consistently added to trend positions exactly when the 20-day realized vol dropped below its 6-month median. I formalized this as an explicit rule-based signal that could be run with disciplined risk controls — effectively converting an implicit PM heuristic into a transparent, testable, systematic strategy. This is precisely the kind of dataset parsing and alpha identification work that Trexquant's model is built around.

[🔝 Back to Top](#-table-of-contents)

---

### 📌 Responsibility 3
**"Investigate and implement state-of-the-art academic research in the field of quantitative finance."**

#### 🔵 What I did at Balyasny

I maintain an active, structured research reading practice — not passive consumption, but a disciplined pipeline from academic paper to production implementation. My process:

**Stage 1 — Screening:** I monitor arXiv (q-fin), SSRN, Journal of Portfolio Management, and Journal of Financial Economics weekly. I screen abstracts for three criteria: (a) empirical, not purely theoretical; (b) reproducible with available data; (c) non-trivially related to an existing signal family in our library.

**Stage 2 — Replication:** I independently replicate the paper's core findings on our data before accepting the result. This is non-negotiable — the academic setting uses data-snooping controls that are often weaker than production standards, and survivorship bias in published results is significant.

**Stage 3 — Adaptation:** Academic signals are almost never deployable as published. I adapt them for: (a) transaction cost realism (academic papers routinely ignore half-spread costs at scale), (b) out-of-sample validation via CPCV rather than the paper's reported in-sample backtest, and (c) regime conditioning to confirm the alpha is not period-specific.

**Concrete Implementation at Balyasny:**

- **HAR-RV (Heterogeneous Autoregressive Realized Volatility):** Implemented the Corsi (2009) HAR-RV model as a volatility forecasting component for dynamic position sizing across FX and Equity futures. Extended it with EGARCH innovations to capture asymmetric volatility responses to negative returns. The HAR-RV forecast fed directly into the VaR estimation module, replacing a simpler EWMA approach and improving volatility forecast accuracy (lower QLIKE score) across all test regimes.

- **Deflated Sharpe Ratio (Bailey & López de Prado, 2014):** Applied DSR as the standard gating metric for all new signals before they could enter the ensemble. This directly addressed the multiple testing problem inherent in a large signal library — a signal with a raw Sharpe of 1.2 across 50 tested variants is far less impressive than the DSR-adjusted figure. Enforcing DSR as a hard gate eliminated several candidate signals that would otherwise have passed traditional Sharpe thresholds.

- **Combinatorial Purged Cross-Validation (López de Prado, 2018):** Implemented CPCV as the firm standard for all backtesting within my signal development workflow. Generated the full distribution of backtested Sharpe ratios across combinatorially constructed path subsets, enabling a statistically honest comparison of signal performance under different realized market paths — not just the single "best" backtest path.

[🔝 Back to Top](#-table-of-contents)

---

### 📌 Responsibility 4
**"Continuously innovate and improve existing models by integrating new data sources and advanced techniques to boost performance and scalability."**

#### 🔵 What I did at Balyasny

Model improvement at Balyasny is not optional — it is structural. The Central Risk Book's performance attribution runs at the signal level, which means every signal is under continuous performance scrutiny. This creates a natural forcing function for innovation.

**Innovation Cycle I Ran:**

```
 Quarterly Attribution Review
        │
        ▼
 Signal IC Decay Detected?
   ├── No  → Document & monitor
   └── Yes → Root Cause Analysis
              ├── Data Quality Issue → Fix upstream pipeline
              ├── Regime Shift      → Recalibrate HMM priors
              ├── Crowding Signal   → Rotate to orthogonal feature family
              └── Structural Decay  → Retire + launch replacement research
```

**New Data Source Integration — Example:**

In Q3 2025, I integrated **satellite-derived commodity positioning data** (shipping lane traffic density as a proxy for physical commodity demand shifts) into the Commodities signal block. The raw data required significant cleaning — coordinate normalization, outlier removal from AIS transponder errors, and temporal alignment to futures contract settlement calendars. I built a Python ETL pipeline (pandas + GeoPandas) to convert raw AIS feeds into daily demand-pressure z-scores by commodity region. The resulting signal had a 6-week predictive horizon for front-month Crude, Copper, and Agricultural futures, with an out-of-sample IC of 0.06 and near-zero correlation to the existing price-based signal library.

**Advanced Technique Integration — Example:**

I replaced a legacy linear ridge regression ensemble weighting scheme with a **sparse autoencoder** (PyTorch, L1 sparsity penalty) to learn a compressed, non-redundant latent representation of the feature matrix. The autoencoder's latent space provided cleaner inputs to the GBM top-model than raw PCA, because it preserved non-linear interaction structure that PCA discarded. This improved the top-model's out-of-sample IC by approximately 12% on the FX signal block — a material improvement at the margin scale of systematic trading.

**Scalability — From Research to the Central Risk Book:**

Every model I build is required to run within the Central Risk Book's latency and memory budget. I refactored signal computation pipelines from Pandas-native to Numba-JIT-compiled NumPy kernels, reducing signal generation time for the full cross-sectional universe from ~8 minutes to ~40 seconds — a 12x speedup that made the signal viable for the intraday rebalancing window.

[🔝 Back to Top](#-table-of-contents)

---

### 📌 Responsibility 5
**"Collaborate closely with a team of experienced quantitative researchers to conduct experiments, backtest hypotheses, and refine strategies through rigorous simulations and data analysis."**

#### 🔵 What I did at Balyasny

Balyasny Systematic's 100+ person structure means research is never done in isolation. Collaboration is operationalized through formal structures:

**Research Review Process:**

Every signal I developed went through a three-stage peer review before touching live capital:

1. **Pre-Mortem Review (with 2 senior QRs):** Before any backtest was run, I presented the economic hypothesis, proposed feature construction, and anticipated failure modes to two senior researchers. This "pre-mortem" forced explicit articulation of the null hypothesis and eliminated obvious data-mining framings before they wasted compute budget.

2. **Backtest Peer Audit:** After initial CPCV validation, a second QR independently replicated the backtest on the same dataset using their own implementation. Discrepancies above 5% in reported Sharpe triggered a joint debugging session. This caught two implementation errors in my signal code over the past year — errors that would have produced inflated backtests had they gone undetected.

3. **Risk Committee Presentation:** Signals approved through peer audit were presented to the risk committee with full attribution transparency: Sharpe by regime, IC decay curve, turnover vs. TCM analysis, and correlation to the existing live signal library. Capital allocation decisions were made at this level — not by the individual researcher.

**Cross-Functional Research Interactions:**

- **Execution Services team:** I collaborated with the TCM and smart order routing teams to understand how my signal's rebalancing frequency interacted with market impact models. This changed my rebalancing cadence for one FX signal from daily to weekly after the TCM analysis showed that daily turnover costs consumed 40% of gross alpha — a finding that is only visible if you work closely with the execution layer, not in isolated research.

- **Replication Book researchers:** I contributed signal primitives (regime indicators, positioning proxies) to the Replication team's alpha-capture workflow, enabling them to formalize patterns observed in discretionary PM behavior. This cross-pollination — where macro proprietary signals feed the replication layer — is only possible in a collaborative research culture, and it is precisely the kind of team dynamic I seek to replicate at Trexquant.

[🔝 Back to Top](#-table-of-contents)

---

## 📋 Requirements Checklist

| Trexquant Requirement | Evidence |
|---|---|
| **BS/MS/PhD in STEM** | M.S. Computer & Electrical Engineering, Binghamton University |
| **2+ years systematic trading** | 17+ years — Millburn (14y), Highbridge/JPMC-AM (3y), Balyasny (present) |
| **Passion for ML** | GBM, RF, NLP Transformers, HMMs, Sparse Autoencoders, LSTM — all in production |
| **Fluent in Python** | Python 3.13+; NumPy, Pandas, Scikit-learn, PyTorch, XGBoost, LightGBM, Numba |
| **Strong problem-solving** | Root-cause analysis framework; pre-mortem research culture; CPCV + DSR gating |
| **Individual + team player** | Solo signal development + peer audit + Risk Committee + cross-team collaboration |

[🔝 Back to Top](#-table-of-contents)

---

## 💬 Closing Statement for the Trexquant Interview Panel

> Everything described above is not aspirational — it is the work I did last quarter. The reason I am excited about Trexquant is that your model — a shared signal library, thousands of statistical algorithms, collaborative research across equities, futures, commodities, and event-driven markets — is architecturally the most natural extension of everything I have built at Balyasny. The difference is that at Trexquant, this research-first, signal-first culture is the firm's singular identity, not one component within a larger multi-strategy platform. That distinction matters to me, and I am ready to contribute from day one.

[🔝 Back to Top](#-table-of-contents)

---

*Prepared for: **Shaikat Majumdar** · Target: **Trexquant Investment LP — Quantitative Researcher (Experienced Hire)***
