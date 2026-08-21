<p align="right" class="gh-only">
  <a href="../index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../assets/atlas-mark-dark.svg">
      <img src="../assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Tabular data

Every block on this page runs. This one sets up the data the rest of them use:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 800
X = rng.normal(size=(n, 8))
z = np.tanh(X[:, 0]) - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3]
y_num = z + 0.3 * rng.normal(size=n)                        # a continuous target
y_bin = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)  # a binary one

X_train, X_test = X[:600], X[600:]
y_train, y_test = y_bin[:600], y_bin[600:]
```

## Classification

```python
from atlas import AtlasClassifier

m = AtlasClassifier().fit(X_train, y_train)
p = m.predict_proba(X_test)[:, 1]
labels = m.predict(X_test)          # cut at m.threshold_, itself chosen leave-one-out
```

Labels come back in the domain you passed in — strings, `{-1, 1}`, anything with two levels.

### A calibrated probability

Every expert is fitted with balanced class weights, because a neighbourhood of ten points is
often one-sided and an unweighted local fit collapses there. The consequence is that each
expert's probability sits on a 50/50 prior, and the combiner's intercept is the only place your
base rate can be recovered:

```python
from atlas import Convex

calibrated = AtlasClassifier(combiner=Convex(class_weight=None)).fit(X_train, y_train)
```

Ranking is identical either way — the shift is monotone — so this matters for the Brier score
and for reading the number *as a probability*, not for AUC.

### Choosing the threshold

```python
from atlas import Fixed, Objective

AtlasClassifier(decision=Objective("f1"))   # a built-in name, or a callable (y, p) -> float
AtlasClassifier(decision=Fixed(0.8))        # when the cut is a business decision
m.threshold_                                # what it settled on
```

The threshold is computed from the combiner's own leave-one-out scores, so it does not move with
`random_state`. That costs a little balanced accuracy against using *this* seed's threshold, and
the trade is deliberate: a decision rule that changes when you change a seed is not a decision
rule.

## Regression

```python
from atlas import AtlasRegressor

r = AtlasRegressor().fit(X[:600], y_num[:600])
r.predict(X[600:])
```

Predictions are invariant to the units of `y`. Each local prediction is also clamped to its own
neighbourhood's range — without that, a narrow rung extrapolates wildly on skewed targets.

The pairwise channel is on by default here and off for classification, and so is the single-index
channel. Both asymmetries are measured rather than assumed: they pay clearly on regression and
not on classification.

## Categorical features

```python
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
df["city"] = pd.Categorical(rng.choice(["riga", "porto", "kyoto"], size=n))
df["brand"] = pd.Categorical(rng.choice(list("abcd"), size=n))

r = AtlasRegressor(cat_features="auto").fit(df, y_num)          # every object/category/bool
c = AtlasClassifier(cat_features=["city", "brand"]).fit(df, y_bin)
c2 = AtlasClassifier(cat_features=[8, 9]).fit(df, y_bin)        # or by position
```

Declared columns become smoothed out-of-fold target statistics, fitted **inside the fold** —
fitting them on all of `X` leaks, invisibly. On columns with only a handful of rows per level,
`cat_count=True` also gives the model a measure of how well supported a level was, which helps
exactly in that sparse case and does nothing once levels are well populated.

Why bother declaring them: every member here is metric-based, so an ordinal code asserts that
category 3 lies between 2 and 4. Why it is not automatic: the encoder is an estimate as well,
and across the benchmark suite declaring everything is a wash on average — clearly better on
some datasets, clearly worse on others, with no rule of thumb separating them. Fit it both ways
and keep the better one; it costs one extra fit.

## Reading the model

```python
m.expert_weights_        # one weight per member, by name
m.library_.names_        # what was built: global, local@k, vote@k, additive@nk, …
m.combiner_, m.stack     # the stages, as configured and as used
print(m.summary())       # a printable digest
```

Expert names are unique by construction — the estimator refuses a library where two members
would be called the same thing, because every one of these views is keyed by name.

## Attribution

```python
a = m.attribution(X_test)

a["shape"]              # (n, p) — per-feature contribution, per row
a["baseline"]           # each row's neighbourhood level; belongs to no feature
a["opaque_share"]       # the fraction from members that cannot be split per feature
```

The identity `shape + surface + baseline + opaque == prediction` holds to machine precision, and
is checked rather than claimed. Use this rather than a global importance vector, which says
nothing about a particular row.

The keys do not depend on which channels you switched on: `surface` is always there, empty when
nothing produced one.

## Blending with a boosted model

This is the model's best use on ordinary tabular data. It is structurally different from a tree
ensemble, so it makes different mistakes — on real regression data a convex blend with a boosted
tree beat that booster alone, while this model by itself was behind.

```python
import numpy as np
from atlas import AtlasRegressor, r2
from atlas.core.library import kfold
from sklearn.ensemble import HistGradientBoostingRegressor as Booster

Xa, ya = X[:600], y_num[:600]
fa, fb = np.zeros(len(ya)), np.zeros(len(ya))
for tr, va in kfold(ya, 5, 0, "regression"):        # out of fold, on the training split only
    fa[va] = AtlasRegressor().fit(Xa[tr], ya[tr]).predict(Xa[va])
    fb[va] = Booster().fit(Xa[tr], ya[tr]).predict(Xa[va])

grid = np.linspace(0, 1, 21)
w = float(grid[np.argmax([r2(ya, u * fa + (1 - u) * fb) for u in grid])])

atlas_model = AtlasRegressor().fit(Xa, ya)
boosted = Booster().fit(Xa, ya)
pred = w * atlas_model.predict(X[600:]) + (1 - w) * boosted.predict(X[600:])
```

The weight goes where it should on its own — near one where this model is much better, near zero
where it is not. On classification the same blend is a wash, so there it buys nothing.

## Speed

It is a lazy learner: the training set *is* the model, and a prediction fits a local regression
per query per rung. Fitting is fast — on the twenty-dataset benchmark it is several times
quicker than a boosted tree — but predicting is not free the way a tree's is, and memory scales
with the data you fitted on.

If a fit is too slow, in the order worth trying: leave `Gradient` off (it is a large fraction of
a fit), drop `Pairwise` on wide data, lower `fit_cap`, and set `approx="auto"` once you are past
a few tens of thousands of rows.
