# Linear Regression Deep-Dive: Adding a Dead Feature, Its Geometry, and the F-Test

### Shared Setup & Notation (used throughout Q01–Q03)

> True DGP:
> * $y = X\beta + \epsilon$
> * $X$ is $n\times k$ full rank
> * $\epsilon \sim N(0,\sigma^2 I_n)$, independent of $X$
>
> This is the **correctly specified** $k$-feature model — no omitted variables, right functional form.
>
> We then form the **augmented** design:
>
> $$
> X_+ = \begin{bmatrix} X & x_{k+1} \end{bmatrix}
> $$
>
> ( $n\times(k+1)$, full rank, $x_{k+1} \notin \text{col}(X)$ ),
>
> where $x_{k+1}$ is a feature you already know **does not work** — i.e. its *true* population coefficient is $\beta_{k+1}=0$.
> The augmented model is therefore *also* correctly specified (it nests the truth with one redundant parameter pinned at zero).
>
> Define:
>
> $$
> \begin{flalign}
> && P_X &= X(X^\top X)^{-1}X^\top,\quad M_X = I-P_X \qquad\text{(orthogonal projector onto col(X), its annihilator)} & \\
> && \tilde x_{k+1} &= M_X x_{k+1} \qquad\text{(component of the new feature orthogonal to everything already in the model)} & \\
> && \hat y_k &= P_{X}y & \\
> && \hat\epsilon_k &= M_Xy \qquad\qquad \hat y_{k+1}=P_{X_+}y & \\
> && \hat\epsilon_{k+1} &= M_{X_+}y &
> \end{flalign}
> $$
>

This single setup is reused without restatement in all three answers below — it is exactly the kind of shared
scaffolding you're expected to set up *once*, out loud, in the first 30 seconds of a Citadel/Cubist quant interview,
before touching any of the three sub-questions.

---

## Q01 · You Add a Feature You Already Know Doesn't Work — What Happens?

**Open with the intuition (15 seconds):**
> "Nothing happens to bias — the true model is nested inside the bigger one, so every coefficient stays unbiased,
> including the dead feature's own coefficient, whose expectation is exactly zero. What changes is purely
> *second-moment*: in-sample fit mechanically improves by construction — R² cannot decrease — while every standard
> error widens and out-of-sample performance gets strictly worse in expectation. You're not introducing omitted-
> variable bias; you're paying a pure variance tax for zero expected benefit. In a signal library with thousands
> of candidate features, this is the default failure mode, not the exception — which is exactly why naive in-sample
> R² is a dangerous gate and CPCV / walk-forward penalization is mandatory."

---

### 🧮 First-Principles Derivation

**Step 1 — Unbiasedness is preserved (this is the asymmetric twin of omitted-variable bias).**

Because $\beta_{k+1}=0$ truly, $X_+$ contains the true model as a special case. OLS on the augmented design gives:

$$
\begin{aligned}
\hat\beta_+ &= (X_+^\top X_+)^{-1}X_+^\top y \\
\mathbb{E}[\hat\beta_+ \mid X_+] &= (X_+^\top X_+)^{-1}X_+^\top \mathbb{E}[y\mid X_+] \\
&= (X_+^\top X_+)^{-1}X_+^\top X_+\beta_+ \\
&= \beta_+
\end{aligned}
$$

where $\beta_+ = (\beta^\top, 0)^\top$. So $\mathbb{E}[\hat\beta_j]=\beta_j$ for $j\le k$ **and** $\mathbb{E}[\hat\beta_{k+1}]=0$.
Contrast this with the *reverse* mistake (omitting a relevant regressor $x_{k+1}$ that is correlated with $X$):

$$
\begin{flalign}
&& \mathbb{E}[\hat\beta^{short}] &= \beta + \delta \cdot \beta_{k+1} & \\
&& \delta &= (X^\top X)^{-1} \cdot X^\top \cdot x_{k+1} &
\end{flalign}
$$

Omission biases the kept coefficients ( unless $\delta=0$ ); **inclusion of a useless feature never does.** This asymmetry is the whole point of the question.

**Step 2 — In-sample fit weakly improves, by construction, regardless of truth.**

OLS minimizes $\|y-X_+\gamma\|^2$ over a strictly larger feasible set than $\|y-X\beta\|^2$ ( set $\gamma_{k+1}=0$ to recover the smaller problem ), so:

$$
RSS_{k+1} \le RSS_k \quad\Longrightarrow\quad R^2_{k+1}\ge R^2_k \quad\text{always, with equality iff } \tilde x_{k+1}^\top y = 0
$$

With continuous data this is a probability-zero event — **$R^2$ will go up even though the feature is genuinely useless.** This is the mechanical artifact that makes naive in-sample $R^2$ unusable as a feature-inclusion gate.

**Step 3 — Quantify the expected drop in residual sum of squares.**

Since the model is correctly specified, $\hat\epsilon_k = M_Xy = M_X\epsilon$ ( the true signal is annihilated by $M_X$ ). Using $\tilde x_{k+1}^\top \hat y_k = 0$ (orthogonality of projections):

$$
\begin{aligned}
\hat\beta_{k+1} &= \frac{\tilde x_{k+1}^\top y}{\tilde x_{k+1}^\top \\
\tilde x_{k+1}} &= \frac{\tilde x_{k+1}^\top \epsilon}{\|\tilde x_{k+1}\|^2} \\
RSS_k-RSS_{k+1} &= \hat\beta_{k+1}^2 \\
\|\tilde x_{k+1}\|^2 &= \frac{(\tilde x_{k+1}^\top \epsilon)^2}{\|\tilde x_{k+1}\|^2}
\end{aligned}
$$

Conditioning on $X_+$, $\tilde x_{k+1}^\top\epsilon \sim N(0,\sigma^2\|\tilde x_{k+1}\|^2)$, so the quantity above is $\sigma^2\chi^2_1$, hence:

|     |
| :-- |
| $`\mathbb{E}[RSS_k - RSS_{k+1}] = \sigma^2 \text{ exactly} `$

**A dead feature soaks up exactly one unit of noise variance, no more, no less, in expectation.** This single
identity is the seed of the entire F-test in Q03.

**Step 4 — The residual-variance estimator $\hat\sigma^2$ stays unbiased — but loses efficiency.**

$RSS_{k+1}=\|M_{X_+}\epsilon\|^2\sim\sigma^2\chi^2_{n-k-1}$, so $\hat\sigma^2_{k+1}=RSS_{k+1}/(n-k-1)$ is unbiased for $\sigma^2$, exactly as $\hat\sigma^2_k$ was. **No bias appears anywhere.** 

But $\hat\sigma^2_{k+1}$ is built from one fewer degree of freedom, so:

$$\text{Var}(\hat\sigma^2_{k+1}) = \frac{2\sigma^4}{n-k-1} > \frac{2\sigma^4}{n-k} = \text{Var}(\hat\sigma^2_k)$$

— a strictly noisier estimate of the same true quantity.

**Step 5 — Every *existing* coefficient's variance weakly increases — the VIF mechanism.**

By the block-inversion (Frisch–Waugh) identity, for any regressor $x_j$ in the augmented set:

$$\text{Var}(\hat\beta_j) = \frac{\sigma^2}{\|M_{X_{(-j)}}\,x_j\|^2}$$

where $X_{(-j)}$ is *all other* regressors. Adding $x_{k+1}$ to that "all other regressors" set can only shrink $\|M_{X_{(-j)}}x_j\|^2$ ( same nested-projection argument as Step 2, with $x_j$ now playing the role of $y$ ):

$$
\text{Var}(\hat\beta_j^{(k+1)\text{-model}}) \;\ge\; \text{Var}(\hat\beta_j^{(k)\text{-model}}) \qquad \text{for every } j\le k
$$

Equivalently, in VIF language — these are two equal ways of writing the *same* quantity, not one multiplied by the other:

$$
\begin{aligned}
\text{Var}(\hat\beta_j) &= \frac{\sigma^2}{SST_j(1-R_j^2)} \\
&= \frac{\sigma^2}{SST_j} \cdot \text{VIF}_j \\
\text{VIF}_j &= \frac{1}{1-R_j^2}
\end{aligned}
$$

where $R_j^2$ is from regressing $x_j$ on every other regressor. Adding $x_{k+1}$ weakly raises $R_j^2$ for every existing $j$ ⇒ VIF rises ⇒ standard errors widen, $t$-stats shrink, confidence intervals balloon.

**$X^\top x_{k+1}=0$ exactly is a sufficient condition for zero damage** — it leaves $\text{Var}(\hat\beta_j)$ unchanged for *every* $j\le k$ simultaneously, because the dead feature is orthogonal to the entire existing design. (It is not narrowly necessary for any single $j$ in isolation — but generically, any nonzero correlation between $x_{k+1}$ and $X$ will strictly inflate at least one of the $k$ variances.)

**Step 6 — Out-of-sample: the bias–variance decomposition flips sign on you.**

Expected out-of-sample prediction error decomposes as $\text{MSE} = \text{Bias}^2+\text{Variance}+\sigma^2$. Step 1 showed Bias² stays at (essentially) zero. Step 5 showed Variance strictly rises whenever $x_{k+1}$ is even slightly correlated with $X$, and even when orthogonal, you still pay a pure estimation-variance cost for the extra
parameter $\hat\beta_{k+1}$ itself, whose sampling noise propagates into every future prediction:

$$
\begin{aligned}
\widehat{y}_{new} &= x_{new}^\top\hat\beta + x_{k+1,new}\hat\beta_{k+1} \\
\text{Var add-on} &= x_{k+1,new}^2\,\text{Var}(\hat\beta_{k+1})
\end{aligned}
$$

Net effect: **in-sample $R^2$ ↑ (mechanically), out-of-sample $R^2$ ↓ (in expectation)** — the textbook signature of overfitting, derived here without invoking the word "overfitting" once.

```
┌──────────────────────────┬─────────────────────┬──────────────────────────┐
│ Quantity                 │ Adding RELEVANT x   │ Adding a DEAD feature    │
│                          │ (omitted var. case) │ (this question)          │
├──────────────────────────┼─────────────────────┼──────────────────────────┤
│ Bias of kept β̂'s         │ Eliminated (was ≠0) │ Was already 0 — no chg   │
│ Bias of new β̂_(k+1)      │ n/a (it's relevant) │ 0 (unbiased)             │
│ In-sample R² / RSS       │ ↑ / ↓ (real signal) │ ↑ / ↓ (pure noise pickup)│
│ Var(β̂_j), j ≤ k          │ may ↑ or ↓          │ weakly ↑ (VIF)           │
│ E[σ̂²]                    │ was biased, now OK  │ stays unbiased           │
│ Var(σ̂²)                  │ improves (less bias)│ worsens (df loss)        │
│ Out-of-sample performance│ improves            │ degrades in expectation  │
└──────────────────────────┴─────────────────────┴──────────────────────────┘
```

---

### 🔎 Follow-Up Questions (interviewer will push here)

**FU1 — "What if $x_{k+1}$ is correlated with the existing features but NOT with $y$?"**

Then $X^\top x_{k+1}\ne 0$ but $\tilde x_{k+1}^\top\epsilon$ is still mean-zero (true coefficient still 0), so Step 1–4 are unchanged. The *only* thing that gets worse is Step 5: multicollinearity is now severe, VIF blows up, and existing $\hat\beta_j$'s become numerically unstable (large standard errors, sign flips across resamples) even though they remain unbiased. This is the realistic case in a signal library — candidate features are rarely exactly orthogonal to your existing 200-factor stack.

**FU2 — "Does Adjusted R² save you?"**

Adjusted $R^2 = 1-\frac{RSS/(n-p)}{TSS/(n-1)}$ rises when adding a regressor **iff $|t_{k+1}|>1$**, i.e. iff $F>1$ — a much weaker bar than the conventional $|t|>2$ significance threshold. So roughly half the time a truly dead feature is added, Adjusted R² will *still* go up by chance ($P(|t|>1)\approx 32\%$ under $H_0$ with large df — not negligible). Adjusted R² is a better gate than raw R², but it is **not** a hypothesis test and should never be used as the sole inclusion criterion.

**FU3 — "What's the multiple-testing version of this?"**

This is the most important real-world extension. If you test $m$ independently-dead candidate features and keep whichever ones cross a $t>2$ (≈95% one-at-a-time) threshold, you expect $\approx 0.05m$ false "discoveries" by pure chance. This is exactly the **Deflated Sharpe Ratio / Benjamini–Hochberg** problem from systematic signal research: naive single-feature significance tests don't control the family-wise or false-discovery rate when you've searched a library of thousands. The fix is multiplicity-corrected thresholds (Bonferroni/BH) or, better, out-of-sample / CPCV validation that doesn't even see the feature-selection step.

**FU4 — "How would Ridge or Lasso change this picture?"**

Ridge shrinks $\hat\beta_{k+1}$ toward 0 proportionally to $\lambda$, directly trading the variance increase from Step 5 for a small, controlled bias — exactly the bias–variance lever Step 6 says you want to pull. Lasso, via its $\ell_1$ penalty, can drive $\hat\beta_{k+1}$ to *exactly* zero with positive probability, which is the regularized analogue of "the F-test failed to reject" — both mechanisms are answering the same question (is this direction worth one degree of freedom?) with different decision rules (continuous shrinkage vs hard threshold).

---

### 🎯 Connect-Back — Systematic Macro Pod, Day-to-Day

This is the single most common silent failure mode in a macro signal pipeline: a researcher backtests a new candidate (e.g., a cross-asset vol-regime overlay) on the same window used to build the existing 30-factor stack, sees Sharpe tick up from 1.55 to 1.58, and ships it. Step 2–3 above say that tick-up is *guaranteed* by construction even if the feature is pure noise — it's not evidence of anything. The pod-level discipline that follows directly from this derivation: (1) never gate signal inclusion on in-sample fit improvement alone; (2) run CPCV / walk-forward OOS Sharpe comparison, where a dead feature's expected contribution is zero or negative once you price in the extra estimation variance; (3) track VIF / correlation-to-existing-stack explicitly before a candidate even reaches backtest, since Step 5 says the *existing* book's coefficient stability degrades the moment you add a correlated-but-useless feature — this shows up as P&L attribution instability on the live book, not just a backtest number.

---

## Q02 · Geometric Interpretation

**Open with the intuition (15 seconds):**
> "OLS is orthogonal projection of $y\in\mathbb{R}^n$ onto the column space of $X$. Adding a feature enlarges that
> subspace from a $k$-dimensional hyperplane to a $(k+1)$-dimensional one that *contains* the old one — and moving
> to a bigger subspace containing your old position can only get you closer to $y$, never further, by the
> Pythagorean theorem. If the new direction happens to be orthogonal to the true signal, you're not capturing
> alpha along it — you're capturing whichever sliver of noise in the residual vector happens to line up with it.
> That's the entire mechanism in one picture."

---

### 🧮 First-Principles Derivation

**Step 1 — OLS as orthogonal projection.**

$\hat y_k = P_X y$ is, by the projection theorem, the *unique* point in $\text{col}(X)$ closest to $y$ in Euclidean distance — this is precisely what "$\min_\beta\|y-X\beta\|^2$" means geometrically. The residual $\hat\epsilon_k = M_Xy$ is, by construction of the orthogonal projector, perpendicular to every vector in $\text{col}(X)$ :

$$
X^\top\hat\epsilon_k = X^\top M_X y = 0
$$

the normal equations *are* the orthogonality condition.

**Step 2 — Nested subspaces: $\text{col}(X)\subset\text{col}(X_+)$.**

Because $X_+ = \begin{bmatrix} X & x_{k+1} \end{bmatrix}$, every linear combination of $X$'s columns is trivially also a linear combination of $X_+$'s columns (just set the new coefficient to 0). So $\text{col}(X)$ is a $k$-dimensional subspace strictly nested inside the $(k+1)$-dimensional $\text{col}(X_+)$.

**Step 3 — Pythagorean proof that $RSS_{k+1}\le RSS_k$ (the geometric version of Q01-Step2), proved rigorously via the orthogonal-direct-sum projector identity — not asserted.**

Because $\tilde x_{k+1}=M_Xx_{k+1}$ is, by construction, orthogonal to every vector in $\text{col}(X)$, the bigger column space splits as an **orthogonal direct sum**: $\text{col}(X_+)=\text{col}(X)\oplus\text{span}\{\tilde x_{k+1}\}$. For any orthogonal direct sum of subspaces $V=V_1\oplus V_2$ (with $V_1\perp V_2$), decomposing any vector uniquely into its $V_1$-component, $V_2$-component, and orthogonal remainder shows immediately that the projector onto $V$ is the **sum** of the individual projectors: $P_V=P_{V_1}+P_{V_2}$. Applying this with $V_1=\text{col}(X)$, $V_2=\text{span}\{\tilde x_{k+1}\}$ :

$$
\begin{flalign}
&& P_{X_+} &= P_X + \frac{\tilde x_{k+1}\tilde x_{k+1}^\top}{\|\tilde x_{k+1}\|^2} & \\
&& \quad\Longrightarrow\quad & \\
&& \hat{y_{k+1}} &= P_{X_+}y & \\
&& &= \hat{y_k} + \underbrace{\frac{\tilde x_{k+1}^\top y}{\|\tilde x_{k+1}\|^2} \cdot \tilde x_{k+1}}_{= \hat\beta_{k+1} \cdot \tilde x_{k+1}} &
\end{flalign}
$$

So $\hat y_{k+1}-\hat y_k=\hat\beta_{k+1} \cdot \tilde x_{k+1}$ **exactly** (this is also exactly the FWL coefficient from Step 4 below), and since $\tilde x_{k+1}\perp\text{col}(X)\ni\hat y_k$, this increment is orthogonal to $\hat y_k$ by construction, not approximately. Combined with $\hat\epsilon_{k+1}\perp\text{col}(X_+)\supseteq\{\hat y_k, \, \hat y_{k+1}-\hat y_k\}$, the Pythagorean theorem on the right triangle $y,\hat y_k,\hat y_{k+1}$ gives:

$$
\|y-\hat y_k\|^2 = \|y-\hat y_{k+1}\|^2 + \|\hat y_{k+1}-\hat y_k\|^2 \quad\Longrightarrow\quad RSS_k = RSS_{k+1} + \underbrace{\hat\beta_{k+1}^2 \cdot \|\tilde x_{k+1}\|^2}_{\ge 0}
$$

This is a **purely geometric** re-derivation of **[Q01-Step3](#q01--you-add-a-feature-you-already-know-doesnt-work--what-happens)** — moving to a bigger subspace can only shorten the distance to $y$ (or leave it unchanged), it can never lengthen it.

**Step 4 — Frisch–Waugh–Lovell (FWL): what the "extra" direction actually is.**

$\tilde x_{k+1} = M_X \cdot x_{k+1}$ is the part of the new feature that is genuinely *new information* — everything about $x_{k+1}$ that was already spanned by $X$ has been projected away. FWL states that regressing $y$ on the full $X_+$ gives the *exact same* $\hat\beta_{k+1}$ as the simple univariate regression of $y$ on $\tilde x_{k+1}$ alone ( equivalently, of $\hat\epsilon_k$ on $\tilde x_{k+1}$, since $\tilde x_{k+1} \perp \text{col}(X)$ kills the $\hat y_k$ part of $y$ ):

$$
\hat\beta_{k+1} = \frac{\tilde x_{k+1}^\top \hat\epsilon_k}{\|\tilde x_{k+1}\|^2}
$$

Geometrically: **the new coefficient is the projection of the *old residual vector* onto the new orthogonalized direction.** This is the cleanest possible statement of "is there anything left in the residual that this new feature explains?"

**Step 5 — Why this is "noise capture" when $\beta_{k+1}=0$ truly.**

Under correct specification, $\hat\epsilon_k=M_X\epsilon$ — the old residual *is* pure noise (rotated into the $(n-k)$-dimensional orthogonal complement of $\text{col}(X)$). Since $\epsilon$ is isotropic Gaussian noise, its projection onto *any* fixed direction inside that $(n-k)$-dim complement — in particular onto $\tilde x_{k+1}$ — has expected squared length exactly $\sigma^2$ (one unit of noise variance per dimension "spent").  The new feature isn't explaining signal; **it is mechanically guaranteed to soak up some noise just by virtue of being a new axis inside a space that is, by assumption, nothing but noise.**

---

### 📐 ASCII Diagram — Nested Subspaces & the Orthogonal Decomposition

```
   R^n  (the full n-dimensional data space)
   ┌───────────────────────────────────────────────────────────┐
   │                                                           │
   │                         y                                 │
   │                        ╱│                                 │
   │                       ╱ │ ε_(k+1) = M_{X+}y               │
   │                      ╱  │ (⟂ col(X₊), lives in            │
   │                     ╱   │  (n-k-1)-dim complement)        │
   │                    ╱    │                                 │
   │             ŷ_(k+1) ────┘                                 │
   │               ╱╲                                          │
   │              ╱  ╲  β̂_(k+1)·x̃_(k+1)                        │
   │             ╱    ╲ ("the extra direction" — 1-dim,        │
   │            ╱      ╲  ⟂ col(X), captured noise if H0 true) │
   │           ╱        ╲                                      │
   │   ┌──────●──────────┴──────────────┐                      │
   │   │     ŷ_k = P_X y                │  col(X)  — k dims    │
   │   │   (closest point in col(X))    │  nested strictly     │
   │   └─────────────────────────────────┘  inside col(X₊)     │
   │            col(X₊) = col(X) ⊕ span{x̃_(k+1)}  — k+1 dims   │
   └───────────────────────────────────────────────────────────┘

   Pythagoras on the big right triangle (y, ŷ_k, ŷ_(k+1)):
       ‖y − ŷ_k‖² = ‖y − ŷ_(k+1)‖² + ‖ŷ_(k+1) − ŷ_k‖²
           RSS_k  =      RSS_(k+1)  +   β̂_(k+1)² ‖x̃_(k+1)‖²
```

---

### 🔎 Follow-Up Questions

**FU1 — "State Frisch–Waugh–Lovell formally and say why it matters operationally."**

For any partition $X_+=[X_1 \, X_2]$, the OLS coefficient on $X_2$ from the *full* regression equals the OLS coefficient from regressing $M_{X_1}y$ on $M_{X_1}X_2$. Operationally: **you can always interpret a multi-factor model coefficient as a simple bivariate regression after "cleaning" both sides against everything else.** In a
macro pod, this is exactly how you read a partial-correlation / marginal-contribution chart for a new signal against an existing 30-factor risk model — you're implicitly running FWL.

**FU2 — "What's the geometric meaning of $R^2$?"**

$R^2 = \cos^2\theta$, where $\theta$ is the angle between the (demeaned) $y$ vector and its projection $\hat y$ onto $\text{col}(X)$. Adding a regressor can only shrink $\theta$ (bring $y$ angularly closer to the subspace) — consistent with Step 3's monotonicity result viewed through trigonometry instead of Pythagoras.

**FU3 — "Does the 'noise capture' argument still hold in high dimensions ($k$ large, $k\to n$)?"**

Yes, and it gets worse: as $k\to n-1$, the orthogonal complement of $\text{col}(X)$ shrinks to nearly nothing, and *any* new direction you add — even pure noise — necessarily aligns more and more with whatever residual variance remains, because there are fewer independent dimensions of "leftover noise" to be orthogonal to. This is the geometric face of the curse of dimensionality / why high-dimensional regression with $k$ comparable to $n$ requires regularization — you eventually run out of room to be orthogonal to noise.

---

### 🎯 Connect-Back — Systematic Macro Pod, Day-to-Day

The "subspace gets bigger, distance never increases" picture is the right mental model whenever you're asked to review a colleague's signal-stacking work: every additional factor in a multi-factor macro risk model strictly shrinks the angle between fitted and actual P&L *in-sample*, full stop, with zero information content required. This is why pod risk reviews lean on **partial correlation against the existing factor stack** (FWL, FU1) rather than raw $R^2$ improvement — you want to know if $\tilde x_{k+1}$, the *cleaned* version of the candidate signal, actually correlates with the *cleaned* residual P&L, not whether the raw $R^2$ ticked up. It is also the reason a senior researcher gets nervous, not reassured, when a junior shows a backtest where adding the 40th factor to an already-dense macro book still "improves" fit — Step 5 above predicts that will happen almost mechanically once the existing factor stack already explains most of the structured signal in the return series.

---

## Q03 · The F-Test, In Context, With Geometric Interpretation

**Open with the intuition (15 seconds):**
> "The F-test asks a precise question: is the extra distance you closed by adding this feature — that little
> orthogonal hop in Q02's diagram — large relative to the noise floor you'd expect from a useless direction? It's
> a ratio of two independent chi-squared pieces living in orthogonal subspaces of the *same* residual vector,
> normalized by their dimension. Under the null it's centered near 1 by construction, because Q01-Step3 already
> told us the expected gain from a dead feature is exactly one unit of $\sigma^2$ — which is precisely what the
> F-test's numerator is built to detect, and precisely what it will, on average, fail to find significant."

---

### 🧮 First-Principles Derivation

**Step 1 — Hypotheses and the nested-model setup.**

$$
H_0:\beta_{k+1}=0 \quad (\text{restricted model} = k\text{-feature model}) \qquad\text{vs}\qquad H_1:\beta_{k+1}\ne 0 \quad (\text{unrestricted} = (k+1)\text{-feature model})$$

Both models are fit by OLS on the *same* $n$ observations; the restricted model is nested inside the unrestricted one ($q=1$ restriction here; the general case tests $q$ joint restrictions, e.g. several features added at once).

**Step 2 — The F-statistic, and its decomposition into orthogonal "lengths."**

$$
\begin{aligned}
F &= \frac{(RSS_R - RSS_U)/q}{RSS_U/(n-p_U)} \\
&= \frac{(RSS_k-RSS_{k+1})/1}{RSS_{k+1}/(n-k-1)}
\end{aligned}
$$

From **[Q02-Step3](#q02--geometric-interpretation)** and **[Q02-Diagram](#q02--geometric-interpretation)**: the numerator's bracketed term is $\|\hat y_{k+1}-\hat y_k\|^2=\hat\beta_{k+1}^2\|\tilde x_{k+1}\|^2$ — the squared length of the **extra fitted vector**, living in the 1-dimensional space $\text{span}\{\tilde x_{k+1}\}$. The denominator's bracketed term is $RSS_{k+1}=\|\hat\epsilon_{k+1}\|^2$ — the squared length of the **final residual**, living in the $(n-k-1)$-dimensional orthogonal complement of $\text{col}(X_+)$. **These two subspaces are orthogonal to each other by construction** (one is exactly $\text{span}\{\tilde x_{k+1}\}$, the other is everything left over after also removing that span), so by Cochran's theorem applied to the Gaussian vector $\epsilon$, the numerator and denominator pieces are **statistically independent**.

**Step 3 — Distribution under $H_0$.**

Under $H_0$, both pieces are pure noise: $(RSS_k-RSS_{k+1})\sim\sigma^2\chi^2_1$ (Q01-Step3) and $RSS_{k+1} \sim \sigma^2\chi^2_{n-k-1}$ (Q01-Step4), independent of each other. A ratio of two independent $\chi^2$ variables, each divided by its own degrees of freedom, is by *definition* an $F$ random variable:

$$
\begin{aligned}
F &= \frac{\chi_1^2/1}{\chi^2_{n-k-1}/(n-k-1)} \\
&\sim F_{1, \, n-k-1}
\end{aligned}
$$

$\mathbb{E}[F]=\frac{n-k-1}{n-k-3}\approx 1$ for reasonably large $n-k$ — **exactly the "centered near 1" intuition from the opener, derived rigorously.** Note this exact finite-sample result needs Gaussian, i.i.d., homoskedastic errors independent of $X$; absent normality, $q\cdot F\to\chi^2_q$ asymptotically by the CLT, which is the basis for robust/asymptotic Wald tests when those assumptions fail.

**Step 4 — The $F=t^2$ identity for single-restriction tests.**

$SE(\hat\beta_{k+1}) = \hat\sigma_{k+1}/\|\tilde x_{k+1}\|$, with $\hat\sigma_{k+1}^2=RSS_{k+1}/(n-k-1)$

Then:

$$
t^2 = \frac{\hat\beta_{k+1}^2}{SE(\hat\beta_{k+1})^2} = \frac{\hat\beta_{k+1}^2\|\tilde x_{k+1}\|^2}{\hat\sigma_{k+1}^2} = \frac{RSS_k-RSS_{k+1}}{RSS_{k+1}/(n-k-1)} = F
$$

**For $q=1$, the $t$-test and $F$-test are the identical test** — the $t$-test is just the signed square root. This is *why* the question only becomes genuinely new (not a relabeling) once $q>1$ — i.e. once you ask whether a *block* of several candidate features adds anything jointly, which a one-at-a-time $t$-test cannot answer (a block can be jointly significant with no single $t$-stat exceeding 2, or vice versa, due to collinearity within the block ).

**Step 5 — $R^2$-based equivalent form (useful when reporting from a regression table without raw $RSS$).**

Since $TSS$ is shared by both models, $RSS=TSS(1-R^2)$, substitute directly:

$$
F = \frac{(R^2_U-R^2_R)/q}{(1-R^2_U)/(n-p_U)}
$$

identical content to Step 2, just rescaled by the common factor $TSS$.

---

### 📐 ASCII Diagram — The F-Test as a Ratio of Orthogonal Squared Lengths

```
                              y
                             ╱│
                            ╱ │
                           ╱  │   ε_(k+1)           RSS_(k+1) = ‖ε_(k+1)‖²
                          ╱   │   lives in the         spread over
                         ╱    │   (n-k-1)-dim         (n-k-1) DIMENSIONS
                        ╱     │   leftover space      → divide by (n-k-1)
                       ╱      │   (⟂ col(X₊))            ↓ "noise floor per dim"
                ŷ_(k+1)───────┘
                  │  ╲
   numerator:     │   ╲   β̂_(k+1)·x̃_(k+1)
   RSS_k−RSS_(k+1)│    ╲  lives in EXACTLY 1 dimension
   = this length² │     ╲ (span{x̃_(k+1)}, ⟂ col(X))
                  │      ╲    → divide by 1
                 ŷ_k ──────●
                     col(X)        ↑ these two boxed
                                     subspaces are mutually
                                     orthogonal ⇒ independent
                                     chi-square pieces (Cochran)

        ┌─────────────────────────────────────────────────────┐
        │         (RSS_k − RSS_(k+1)) / 1                     │
        │  F  =  ─────────────────────────────  ~  F_{1,n-k-1}│
        │           RSS_(k+1) / (n-k-1)                       │
        └─────────────────────────────────────────────────────┘
           "extra length² per new dimension you spent"
        ───────────────────────────────────────────────────────
           "leftover length² per dimension still unexplained"
```

**Reading the picture directly off Q01/Q02:** if $x_{k+1}$ is truly dead, the numerator length² has expectation $\sigma^2$ (one noise unit, Q01-Step3) and the denominator length² **per dimension** also has expectation $\sigma^2$ (Q01-Step4) — so the ratio is expected to sit at $\approx 1$, which is the **non-rejection region**. A real feature instead produces a numerator length² that grows with $n$ (signal accumulates), while the denominator stays pinned near $\sigma^2$ — pushing $F$ to diverge from 1 and into the tail.

---

### 🔎 Follow-Up Questions

**FU1 — "Generalize to testing $q>1$ added features jointly (partial F-test)."**

Same picture, but the numerator subspace is now $q$-dimensional ($\text{span}\{\tilde x_{k+1},\ldots,\tilde x_{k+q}\}$, each orthogonalized against $X$ *and* against each other in sequence), giving $F=\frac{(RSS_k-RSS_{k+q})/q}{RSS_{k+q}/(n-k-q)}\sim F_{q,n-k-q}$. This is exactly the right tool for "does this whole new alternative-data bucket add anything jointly," as opposed to feature-by-feature $t$-tests that are blind to within-bucket collinearity.

**FU2 — "Relate the F-test to the likelihood ratio test."**

Under Gaussian errors, $-2\log\Lambda = n\log(RSS_R/RSS_U)$ is the LRT statistic. Rearranging the definition of $F$ gives the algebraic identity $RSS_R/RSS_U = 1+qF/(n-p_U)$, so substituting yields an **exact** relationship between the two statistics, true for every finite $n$ — not an approximation:

$$
-2\log\Lambda \;=\; n\log\!\left(1+\frac{qF}{n-p_U}\right) \qquad \text{(exact, for all finite } n)
$$

The asymptotics enter only one level down, in the **reference distributions**: as $n\to\infty$ with $q$ fixed, $qF/(n-p_U)\to 0$, so $\log(1+x)\approx x$ gives $-2\log\Lambda\approx qF$, and since $F_{q,n-p_U}\to\chi^2_q/q$ in that limit, both statistics converge to the same $\chi^2_q$ law. So: the *statistics* are tied together exactly at any sample size; it is only the convergence of the *exact* $F_{q,n-p_U}$ distribution to the *asymptotic* $\chi^2_q$ distribution that is a large-$n$ approximation. This is why model-selection criteria built on likelihood (AIC/BIC) and hypothesis-testing criteria (F-test) tend to agree qualitatively but not numerically — AIC's implicit threshold is roughly $F>2$, not the conventional $F$-table $\alpha=0.05$ cutoff.

**FU3 — "What breaks the exact $F_{q,n-p}$ distribution, and what do you do instead?"**

Heteroskedasticity or serial correlation in $\epsilon$ (the realistic case for daily macro/futures returns) breaks both the $\chi^2$ shape *and* the independence-of-numerator-and-denominator argument in Step 2 (Cochran's theorem needs i.i.d. Gaussian errors). The fix is a **heteroskedasticity/autocorrelation-robust Wald test**: replace the classical $(X^\top X)^{-1}\sigma^2$ covariance with a Newey–West or White sandwich estimator and use the asymptotic $\chi^2_q/q$ reference distribution instead of the exact finite-sample $F$.

**FU4 — "Why use a hypothesis test at all instead of just comparing OOS Sharpe directly?"**

Because the F-test, run correctly *in-sample*, is answering a *narrower* and *cheaper* question — "is this specific coefficient distinguishable from the noise floor given this exact design matrix" — which is useful as a fast first-pass filter precisely because it doesn't require burning held-out data. But it shares the multiple-testing vulnerability from Q01-FU3 and assumes a fixed, pre-specified hypothesis — it says nothing about data-mined features chosen *after* looking at many candidates. OOS/CPCV Sharpe comparison is the test that actually answers the economically relevant question and is robust to the search process itself; the F-test is a fast triage tool upstream of it, not a substitute for it.

---

### 🎯 Connect-Back — Systematic Macro Pod, Day-to-Day

This is the formal version of the everyday "is this candidate signal's t-stat real" conversation in a macro factor review. In practice you rarely run a textbook single-feature $F$-test in isolation — you run **partial F-tests on blocks** (FU1) when a researcher proposes adding a cluster of correlated alt-data features together, because a block can hide real joint signal behind individually-insignificant $t$-stats, or hide pure collinearity behind a deceptively large joint $R^2$ gain. And because daily macro/futures P&L residuals are essentially never i.i.d. Gaussian — they're fat-tailed and regime-dependent — every F-test result that matters operationally gets re-run with Newey–West-robust standard errors (FU3) before it's allowed to influence book construction; an F-stat that looks significant under classical assumptions but evaporates under HAC correction is one of the most common "this signal isn't real" red flags reviewed in pod signal-acceptance meetings, right alongside the CPCV / Deflated-Sharpe gate from Q01.
