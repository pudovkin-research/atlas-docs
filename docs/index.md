<div class="hero" align="center">
  <picture class="gh-only">
    <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-mark-dark.svg">
    <img src="assets/atlas-mark.svg" alt="ATLAS" width="132" height="132">
  </picture>
  <span class="hero__mark" role="img" aria-label="ATLAS"></span>
  <h1>atlas-ml</h1>
  <p class="hero__sub">SuperLearner by pudovkin-research</p>
  <p class="hero__thesis">A binary classifier and a regressor built as a library of local models,
  combined by a <em>strictly convex program</em>.</p>
</div>
Python, with a Rust core; numpy and pandas are the only runtime dependencies.

```bash
pip install atlas-ml
```

## Sixty seconds

Every example on this site runs as written. This one makes its own data:

```python
import numpy as np
from atlas import AtlasClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 8))
z = X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3]
y = (rng.random(600) < 1 / (1 + np.exp(-z))).astype(int)
X_train, y_train, X_test, y_test = X[:400], y[:400], X[400:], y[400:]

m = AtlasClassifier().fit(X_train, y_train)
p = m.predict_proba(X_test)[:, 1]
```

That is the whole API. Nothing is searched, so a second fit on the same data returns the same
model — weights, probabilities and decision threshold alike.

```python
m.expert_weights_        # what the combiner leaned on, one weight per member
m.attribution(X_test)    # every prediction split per feature, exactly
print(m.summary())       # members, in-fold metrics, top weights
```

Regression is the same shape:

```python
from atlas import AtlasRegressor

r = AtlasRegressor().fit(X_train, z[:400])
r.predict(X_test)
```

## The four things worth knowing

If you read nothing else on this site:

**One.** Do not standardise your features — the model does it, and rescaling a column changes
its answer by nothing at all. What *does* pay is dropping columns that carry no signal: this is a
metric method, so every useless column dilutes the useful ones. [Preparing your data](data.md)
measures both.

**Two.** The defaults are meant to be used. There is no tuning step, because there is nothing to
tune — every ladder is mixed rather than selected. The only settings most people ever touch are
`cat_features` when the frame has categoricals, and `stack=Blocked(...)` when the rows are in
time order. Both are on the [choosing](choosing.md) page.

**Three.** The number `summary()` prints is *in-fold* and optimistic — it is cut with the same
folds that built the model's inputs. It is a diagnostic. To know how the model will do, hold
data out yourself.

**Four.** It is a lazy learner: the training set is the model. Fitting is fast, predicting is
not free, and memory scales with the data you fitted on.

## Where to go next

| | |
|---|---|
| [Preparing your data](data.md) | **start here** — what to do to the columns before the model sees them |
| [Choosing settings](choosing.md) | and then this — what to set for the data you actually have |
| [Concepts](concepts.md) | what a fit builds, and the properties the design protects |
| [Parameters](parameters.md) | every parameter of every object, generated from the code |
| [API](api.md) | methods, fitted attributes, metrics |
| [Tabular data](tutorials/tabular.md) | classification, regression, categoricals, calibration, attribution |
| [Time series](tutorials/series.md) | the ordered stack, the delay design, the channels for time |
| [Extending it](tutorials/extending.md) | your own channel, combiner or decision rule |

## Should you use it?

**Yes, if** your sample is small — at a few hundred rows it is ahead of gradient boosting on
twenty real datasets, and the advantage decays as n grows. **Yes, if** you need per-sample per-feature
attribution that is exact rather than approximated. **Yes, if** you want a second, structurally
different model to blend with a boosted one; on tabular regression that blend beat CatBoost
alone.

**No, if** you have hundreds of thousands of rows and want the last decimal. By a few thousand
rows the advantage is gone, and a boosted tree will be faster to predict with.
