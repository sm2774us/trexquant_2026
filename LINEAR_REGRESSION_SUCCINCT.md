# Linear Regression Deep-Dive: Adding a Dead Feature, Its Geometry, and the F-Test

## Q1 · A k-feature linear model accurately describes the data-generating process. If I add an extra feature that I already know does not work, what would you expect to see?

**Open with the intuition (15 seconds):**

**"When you inject a pure noise feature into a perfectly specified linear model, you are essentially taxing your degrees of freedom without buying any new structural information. In expectation, the OLS coefficient for this garbage feature will be zero. However, in practice, the model will aggressively overfit to whatever spurious in-sample correlation exists between the noise and the target. This marginally artificially inflates your in-sample $R^2$, strictly degrades your adjusted $R^2$, and inflates the variance of the true estimators if the noise is even slightly collinear with your real features, ultimately destroying out-of-sample predictive power."**

### 🧮 First-Principles Derivation

**Step 1 —**

Define the true Data Generating Process (DGP). The target $\mathbf{y}$ is fully explained by an $n \times k$ matrix of true features $\mathbf{X}$ and spherical errors $\boldsymbol{\epsilon}$:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \qquad \mathbb{E}[\boldsymbol{\epsilon}|\mathbf{X}] = \mathbf{0}, \quad \text{Var}(\boldsymbol{\epsilon}|\mathbf{X}) = \sigma^2 \mathbf{I}
$$

**Step 2 —**

Define the unrestricted (misspecified) model. We augment $\mathbf{X}$ with a pure noise feature $\mathbf{z}$ (an $n \times 1$ vector), giving a new parameter $\gamma$:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\hat{\beta}} + \mathbf{z}\hat{\gamma} + \mathbf{e}
$$

**Step 3 —**

Isolate the effect of the new feature using the Frisch-Waugh-Lovell (FWL) Theorem. To find the OLS estimate $\hat{\gamma}$, we project $\mathbf{z}$ onto the null space of $\mathbf{X}$. Let $\mathbf{M_X} = \mathbf{I} - \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ be the residual maker matrix. We regress the $\mathbf{X}$-residuals of $\mathbf{y}$ onto the $\mathbf{X}$-residuals of $\mathbf{z}$:

$$
\hat{\gamma} = (\mathbf{z}^\top \mathbf{M_X} \mathbf{z})^{-1} \mathbf{z}^\top \mathbf{M_X} \mathbf{y}
$$

**Step 4 —**

Substitute the true DGP into our FWL equation. Since $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, we know that $\mathbf{M_X} \mathbf{X} = \mathbf{0}$. Therefore, $\mathbf{M_X} \mathbf{y} = \mathbf{M_X} \boldsymbol{\epsilon}$:

$$
\hat{\gamma} = (\mathbf{z}^\top \mathbf{M_X} \mathbf{z})^{-1} \mathbf{z}^\top \mathbf{M_X} \boldsymbol{\epsilon}
$$

**Step 5 —**

Take the expectation to prove unbiasedness. Because $\mathbf{z}$ is known noise and exogenous, $\mathbb{E}[\boldsymbol{\epsilon}|\mathbf{X}, \mathbf{z}] = \mathbf{0}$:

$$
\mathbb{E}[\hat{\gamma}|\mathbf{X}, \mathbf{z}] = (\mathbf{z}^\top \mathbf{M_X} \mathbf{z})^{-1} \mathbf{z}^\top \mathbf{M_X} \mathbb{E}[\boldsymbol{\epsilon}|\mathbf{X}, \mathbf{z}] = 0
$$

*Conclusion:* The expected value of the noise coefficient is zero.

**Step 6 —**

Derive the Variance Inflation Factor (VIF). What happens to the estimates of the *true* features, $\boldsymbol{\hat{\beta}}$? The variance of the $j$-th true coefficient in the new model becomes:

$$
\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{\sum (x_{ij} - \bar{x}_j)^2} \times \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other features *including* $\mathbf{z}$. Even if $\mathbf{z}$ is random, any finite-sample collinearity increases $R_j^2$, strictly inflating the variance of your valid estimators.

**Follow-Up Questions You Should Anticipate:**

* *Q: What happens to in-sample vs out-of-sample MSE?*
* **A:** In-sample MSE strictly decreases (or stays exactly flat in the 0-probability event of perfect orthogonality) because OLS minimizes squared error and has an extra parameter to play with. Out-of-sample MSE strictly increases because the variance of the predictions $\text{Var}(\hat{\mathbf{y}})$ increases by $\sigma^2/n$ for every useless parameter added, injecting bias via overfitting.

* *Q: What if the noise feature is perfectly orthogonal to X?*
* **A:** Then $\mathbf{M_X} \mathbf{z} = \mathbf{z}$, and VIF = 1. The variance of $\boldsymbol{\hat{\beta}}$ is unaffected. However, you still lose 1 degree of freedom, widening your confidence intervals.

**The Systematic Macro Pod Context:**
In a macro pod, your $N$ (number of observations) is brutally small—perhaps 300 months of data. If you have a perfectly specified nowcast for Non-Farm Payrolls (NFP) using yield curve spreads and ISM data (k=4), and a junior quant pitches adding a trendy, low-history alternative data feature (like satellite imagery of parking lots), it is disastrous if it's noise. Even if the alt-data is structurally useless, it will soak up your scarce degrees of freedom. In small $N$ regimes, finite-sample spurious correlation is rampant. Adding that feature will blow up the variance of your trusted ISM coefficients, causing your model to aggressively trade false signals during the next macro regime shift.

---

## Q2 · Can you provide a geometric interpretation for your above answer.

**Open with the intuition (15 seconds):**

**"Geometrically, Ordinary Least Squares is simply the act of dropping a perpendicular line from the target variable onto the floor spanned by your features. When you add a useless feature, you are adding a new dimension to that floor—turning a 2D plane into a 3D volume, for instance. Because the new feature is noise, the true target vector has no real inclination toward this new dimension. However, the spherical noise ($\boldsymbol{\epsilon}$) will inevitably have a tiny, random 'tilt' into this newly opened dimension. OLS greedily captures this tiny tilt, pulling the shadow (prediction) slightly away from the truth and toward the noise."**

### 🧮 First-Principles Derivation (Geometric)

**Step 1 —**

Setup the Vectors. Let $\mathbf{y} \in \mathbb{R}^n$ be the target vector. The feature matrix $\mathbf{X}$ spans a $k$-dimensional subspace $\mathcal{S_X}$ within $\mathbb{R}^n$.

**Step 2 —**

The Original Projection. The OLS prediction $\mathbf{\hat{y}}$ is the orthogonal projection of $\mathbf{y}$ onto $\mathcal{S_X}$. The residual $\mathbf{e} = \mathbf{y} - \mathbf{\hat{y}}$ is orthogonal to every vector in $\mathcal{S_X}$.

$$
\mathbf{\hat{y}} = \mathbf{P_X} \mathbf{y}, \qquad \mathbf{e} = \mathbf{M_X} \mathbf{y}
$$

**Step 3 —**

Adding the Useless Feature. Adding $\mathbf{z}$ expands our subspace to $k+1$ dimensions: $\mathcal{S_{X,z}} = \text{span}(\mathbf{X}, \mathbf{z})$.

**Step 4 —**

Orthogonalizing the New Feature. The new dimension introduced isn't just $\mathbf{z}$, it's the part of $\mathbf{z}$ that doesn't already overlap with $\mathbf{X}$. This new orthogonal direction is exactly $\mathbf{M_X} \mathbf{z}$ (the residual of $\mathbf{z}$ regressed on $\mathbf{X}$).

**Step 5 —**

The New Projection. By the properties of orthogonal projections, the new prediction $\mathbf{\tilde{y}}$ in the $k+1$ space is the sum of the old projection and the projection onto the new, orthogonalized vector:

$$
\mathbf{\tilde{y}} = \mathbf{\hat{y}} + \mathbf{P_{M_X z}} \mathbf{y}
$$

Since the true model is $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$, the "true" part $\mathbf{X}\boldsymbol{\beta}$ is fully captured by $\mathbf{\hat{y}}$. Therefore, the new addition is just projecting the *noise* :

$$
\mathbf{\tilde{y}} = \mathbf{\hat{y}} + \mathbf{P_{M_X z}} \boldsymbol{\epsilon}
$$

#### Visual Aid (ASCII Diagram):

```text
               y (Target Vector in R^n)
               *
              /|
             / |
            /  |  e (Original Residual, orthogonal to Plane X)
           /   |
          /    |
         /     |                        z (New Noise Vector)
        *------* y_hat (Original Pred)   /
       /       | \                      /
      /        |  \ e_new              /
     /         |   \                  /
    /          *----* y_tilde        /
   /           (New Pred)           /
  /________________________________/
   Plane X (k-dimensional subspace)

```

**Follow-Up Questions You Should Anticipate:**

* **Q: What is the geometric meaning of $R^2$ in this context?**
* **A:** $R^2$ is $\cos^2(\theta)$, where $\theta$ is the angle between the target vector $\mathbf{y}$ (mean-centered) and the feature subspace. When you add $\mathbf{z}$, the subspace expands, allowing the projection to get slightly closer to $\mathbf{y}$, strictly decreasing the angle $\theta$, thus strictly increasing $R^2$.

* *Q: How does L2 Regularization (Ridge) alter this geometry?*
* **A:** Ridge pulls the projection $\mathbf{\tilde{y}}$ back towards the origin. It prevents the model from fully committing to the $\mathbf{P_{M_X z}} \boldsymbol{\epsilon}$ projection unless the variance explained justifies the penalty norm.

**The Systematic Macro Pod Context:**

When we run PCA on global interest rate curves, we extract Principal Components (Level, Slope, Curvature). These represent our orthogonal subspace $\mathbf{X}$. If a researcher wants to add a "4th PC" (which is mostly microstructure noise), geometrically, they are just offering the regression a new axis to fit the training noise. In macro, where signal-to-noise is razor-thin, protecting the geometric purity of your dominant subspace (the first 3 PCs) by refusing to project onto noisy dimensions is the only way to survive out-of-sample drawdowns.

---

## Q3 · Can you describe the F-test in greater detail, specifically in the context of the previous problem and then give the geometric interpretation?

**Open with the intuition (15 seconds):**

**"The F-test is the statistical bouncer at the club of your linear model. It evaluates whether a new block of features (or a single feature) explains enough variance to justify the degrees of freedom it consumes. In the context of adding our useless feature $\mathbf{z}$, the F-test measures the signal-to-noise ratio of that specific addition. It compares the squared length of the new projection onto $\mathbf{z}$ against the squared length of the remaining residual noise. If the new feature is true noise, this ratio will be small, and the F-test will fail to reject the null hypothesis that $\gamma = 0$."**

### 🧮 First-Principles Derivation

**Step 1 — Define the Hypotheses.**

* Restricted Model ($H_0$): $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$
* Unrestricted Model ($H_A$): $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \mathbf{z}\gamma + \boldsymbol{\epsilon}$
* Null Hypothesis: $\gamma = 0$. (Number of restrictions $q = 1$).

**Step 2 — Define the Sum of Squared Residuals (SSR).**

* $SSR_{restricted} = ||\mathbf{y} - \mathbf{\hat{y}}||^2 = \mathbf{e}^\top \mathbf{e}$
* $SSR_{unrestricted} = ||\mathbf{y} - \mathbf{\tilde{y}}||^2 = \mathbf{e}_{new}^\top \mathbf{e}_{new}$

**Step 3 — Construct the F-statistic.**

The F-stat compares the drop in SSR per added parameter ($q$) to the variance of the unrestricted residuals per remaining degree of freedom ($n - k - 1$):

$$F = \frac{(SSR_{restricted} - SSR_{unrestricted}) / q}{SSR_{unrestricted} / (n - k - q)}$$

**Step 4 — Geometric Substitution.**

From [Q2](#q2--can-you-provide-a-geometric-interpretation-for-your-above-answer), we know the unrestricted prediction $\mathbf{\tilde{y}}$ is the restricted prediction $\mathbf{\hat{y}}$ plus the projection onto the orthogonalized $\mathbf{z}$:

$$
\mathbf{\tilde{y}} = \mathbf{\hat{y}} + \mathbf{P_{M_X z}} \mathbf{y}
$$

By the Pythagorean theorem in $n$-dimensional space, the old residual hypotenuse is split into the new projection and the new residual:

$$||\mathbf{y} - \mathbf{\hat{y}}||^2 = ||\mathbf{P_{M_X z}} \mathbf{y}||^2 + ||\mathbf{y} - \mathbf{\tilde{y}}||^2$$

$$SSR_{restricted} - SSR_{unrestricted} = ||\mathbf{P_{M_X z}} \mathbf{y}||^2$$

**Step 5 —**

The Geometric F-statistic. Substitute this back into the F-formula ($q=1$):

$$F = \frac{||\mathbf{P_{M_X z}} \mathbf{y}||^2 / 1}{||\mathbf{e}_{new}||^2 / (n - k - 1)}$$

*Geometric Interpretation:* The F-statistic is exactly the ratio of the squared length of the target's shadow on the *newly added dimension* divided by the squared length of the *unexplained noise vector*, scaled by their respective dimensions. If $\mathbf{z}$ is noise, the shadow $||\mathbf{P_{M_X z}} \mathbf{y}||^2$ is tiny (just capturing $\boldsymbol{\epsilon}$), making $F$ small, meaning we fail to reject $H_0$.

**Follow-Up Questions You Should Anticipate:**

* *Q: How does the F-test relate to the t-test for a single coefficient?*
* **A:** When testing a single restriction ($q=1$), the F-statistic is mathematically identical to the square of the t-statistic ($F = t^2$). The t-test checks if the coefficient is far from zero scaled by its standard error; the F-test checks if the variance explained by the subspace is large relative to the noise variance.

* *Q: What if we added a block of $m$ useless features instead of just one?*
* **A:** The geometry remains the same, but the new orthogonal subspace is $m$-dimensional. The numerator becomes the squared length of the projection onto this $m$-dimensional subspace divided by $m$. The denominator loses $m$ degrees of freedom instead of 1.

**The Systematic Macro Pod Context:**
We rarely evaluate single factors in isolation because macroeconomic data is highly correlated (e.g., you don't just add "German CPI", you add a block of "European Inflation Gauges"). The F-test is the daily workhorse for *block significance*. If a QR wants to add 5 new APAC supply chain indices to a global equities model, we run an F-test comparing the unrestricted ($K_{old} + 5$) model against the baseline. The geometric reality is that those 5 indices usually span a very small unique volume orthogonal to what we already know. If the shadow of the target on that new 5D volume isn't significantly longer than the ambient noise, the F-test fails, and we reject the PR to merge those signals into production.