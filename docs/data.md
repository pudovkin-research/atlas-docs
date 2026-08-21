<p align="right" class="gh-only">
  <a href="index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-mark-dark.svg">
      <img src="assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Preparing your data

Every block on this page runs, and each one is a comparison you can re-run on your own data.
That matters more here than anywhere else on the site: which of these things bites depends on
what your columns look like, and the code below is how you find out.

```python
import numpy as np
from atlas import AtlasRegressor, r2

rng = np.random.default_rng(0)
n = 800
X = rng.normal(size=(n, 6))
z = np.tanh(X[:, 0]) - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3]
y = z + 0.3 * rng.normal(size=n)
tr, te = slice(0, 600), slice(600, None)

def score(Xv, yv=y):
    m = AtlasRegressor(verbose=False).fit(Xv[tr], yv[tr])
    return round(r2(yv[te], m.predict(Xv[te])), 4)
```

## Do you need to scale the features? No.

The model standardises internally, twice and for different reasons: neighbours are measured in a
space whose axes are scaled by the global fit's coefficients, and the channels that read a shape
or a slope get a plainly standardised space of their own. Rescaling or shifting a column before
you hand it over changes nothing at all:

```python
print("reference          ", score(X))
print("one column x 1000  ", score(X * np.where(np.arange(6) == 0, 1000.0, 1.0)))
print("one column + 500   ", score(X + np.where(np.arange(6) == 0, 500.0, 0.0)))
print("every column rescaled", score(X * rng.uniform(0.01, 100, size=6)))
```

Identical, to the last decimal — the differences are floating-point noise. So: do not
standardise, do not min-max, do not centre. It is work that changes nothing.

!!! warning "That is *affine* rescaling only"
    A **monotone but non-affine** transform — a log, a square root, a rank — is a different
    matter entirely, and it does change the model. A tree only cares about the order of a
    column; this model computes distances in it, so the *shape* of the column matters. The next
    two sections are about exactly that.

## The one that costs most: columns that carry nothing

This is the largest single lever on this page, and it is the opposite of the usual advice for
tree ensembles. A tree splits on informative coordinates and ignores the rest. A metric method
computes a distance in **all** of them, so every uninformative column dilutes the ones that
matter:

```python
for extra in (0, 6, 24, 60):
    Xn = np.hstack([X, rng.normal(size=(n, extra))]) if extra else X
    print(f"{6 + extra:3d} features ({extra:2d} of them noise): {score(Xn)}")
```

The fall is steep and it does not level off. If you have a wide table and no idea which columns
matter, spend your effort there before you spend it anywhere else — a correlation filter, a
variance filter, a quick importance pass with any model you like. Passing everything "just in
case" is not free here.

The same logic explains a smaller effect: an exact duplicate column and a constant column both
cost a little, because both take up room in the distance while carrying nothing.

!!! tip "It is the useless columns, not the number of them"
    This is worth being precise about, because the obvious conclusion — "reduce the dimension" —
    is wrong. Measured on real sentence embeddings, where every coordinate carries a little
    signal: **384 embedding dimensions cost nothing at all**, and the model is competitive with a
    linear model there and clearly ahead of a boosted tree. Take 64 of those dimensions and pad
    back to 384 with noise, and the score falls sharply. Same width, opposite outcome.

    So do not run a PCA on a good representation to make it narrower — on embeddings that only
    loses information. Drop columns that carry nothing; keep columns that carry a little.

## Heavy tails and skew in a feature

A lognormal column is a genuine problem for a distance: most of the mass sits in a small region
and a few points sit far away, so the metric spends its resolution on the tail.

```python
Xh = X.copy()
Xh[:, 1] = np.exp(2.0 * X[:, 1])                                  # a heavy-tailed column
Xl = Xh.copy(); Xl[:, 1] = np.log(Xh[:, 1])                       # a log puts it back
Xr = Xh.copy(); Xr[:, 1] = np.argsort(np.argsort(Xh[:, 1])) / n   # a rank transform

print("raw (lognormal) ", score(Xh))
print("log             ", score(Xl))
print("rank / quantile ", score(Xr))
```

The log recovers what the raw column lost. The rank transform does **not**, and that is the
lesson worth taking: a rank makes the marginal distribution pretty and destroys the *shape* of
the relationship inside the column. Prefer a transform that keeps the relationship — a log for
something multiplicative, a square root for a count — over one that only fixes the histogram.

## A skewed target

Regression predictions are invariant to the units of `y`, so scaling the target is as pointless
as scaling a feature. A skewed target is different, and the difference is large:

```python
y_pos = np.exp(0.9 * z + 0.3 * rng.normal(size=n))               # a positive, skewed target

m_raw = AtlasRegressor(verbose=False).fit(X[tr], y_pos[tr])
m_log = AtlasRegressor(verbose=False).fit(X[tr], np.log(y_pos[tr]))

print("raw target                  ", round(r2(y_pos[te], m_raw.predict(X[te])), 4))
print("log target, back-transformed", round(r2(y_pos[te], np.exp(m_log.predict(X[te]))), 4))
```

If your target is a price, a count, a duration or anything else that is positive and
right-skewed, model it on a log scale and transform back. Two cautions: back-transforming a
prediction of `log y` gives you a median rather than a mean, and the attribution you get out is
a split of `log y`, which is a different question from a split of `y`.

## Outliers in a feature

The internal scaling is a plain standardisation, which is not robust: one wild value compresses
everything else into a narrow band.

```python
Xo = X.copy()
Xo[3, 2] = 1e4
print("clean        ", score(X))
print("one wild value", score(Xo))
```

Clip or winsorise obviously impossible values before fitting. Note that this is about outliers
in `X`; contaminated values in the **target** are a different problem, and there the library has
a channel for it — see [choosing settings](choosing.md).

## Missing values

They are refused, at every door, naming the row and column:

```python
Xm = X.copy()
Xm[5, 2] = np.nan
try:
    AtlasRegressor(verbose=False).fit(Xm[tr], y[tr])
except ValueError as e:
    print(str(e)[:110])
```

Refused rather than filled, deliberately. Every member here is metric-based, so one NaN reaches
the distance scan; and which fill to use — a median, a group median, an indicator column, a
model — is a modelling decision that belongs to you, not to a silent default. Fill them however
your problem demands, and consider adding a `was_missing` indicator column: that is often the
informative part.

## Categorical columns

**Do not one-hot them.** One-hot encoding a twenty-level column adds twenty columns to a
distance, which is the dilution problem above in its purest form. That part is not close.

**Whether to *declare* them is a question to measure, not a rule to follow.** Declaring a column
replaces its levels with a smoothed target statistic fitted inside the fold; leaving it alone
means the model sees an integer code and treats it as a number. Measured across the categorical
datasets in the benchmark suite, declaring everything is a wash on average — it helps clearly on
some sets and hurts clearly on others, and no threshold on the number of levels separates the two
cases. Try both on your data; the comparison costs one extra fit.

```python
import pandas as pd

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
df["kind"] = pd.Categorical(rng.choice(list("abcd"), size=n))
m = AtlasRegressor(cat_features="auto", verbose=False).fit(df.iloc[tr], y[tr])
```

An integer code does assert that level 3 sits between 2 and 4, and every member here is metric-
based, so it believes that assertion. The reason declaring is not simply better anyway is that
the encoder is itself an estimate: with few levels, or few rows per level, it adds variance in
exchange for removing an assumption that the additive and local members were already able to
shape around.

## Rows in time order

If your rows are a sequence, the preparation question changes shape completely — the design
matters more than any transform, and a wide dump of every lag you can compute is the dilution
problem again. [Time series](tutorials/series.md) covers it.

## The checklist

1. **Drop columns that carry nothing.** The largest lever on this page, by a wide margin.
2. **Do not standardise.** It changes nothing; the model already does it.
3. **Do fix shape** — a log for a multiplicative column, not a rank.
4. **Model a skewed target on a log scale** and transform back.
5. **Clip impossible values** in `X`; the internal scaling is not robust to them.
6. **Fill missing values yourself**, and consider keeping an indicator.
7. **Never one-hot** for this model; try declaring your categoricals both ways and keep the
   better one.
8. **If the rows are ordered, say so** — that is a bigger decision than any of the above.
