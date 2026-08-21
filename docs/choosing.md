<p align="right" class="gh-only">
  <a href="index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-mark-dark.svg">
      <img src="assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Choosing settings

Start with the defaults. They are not a starting point to be tuned away from — there is no
tuning step in this model, because every ladder is *mixed* by the combiner rather than selected.
This page is about the handful of decisions that are genuinely yours: they follow from what your
data **is**, not from a search over what scores best. For what to do to the columns themselves
before any of this — and the short answer on scaling, which is *don't* — see [preparing your
data](data.md).

## The short version

| your data | what to write |
|---|---|
| a plain table, rows independent | `AtlasClassifier()` / `AtlasRegressor()` |
| a DataFrame with categorical columns | pass the frame; try `cat_features="auto"` both ways |
| rows in time order | `AtlasRegressor(stack=Blocked(n_splits=5, embargo=<one cycle>))` |
| you need calibrated probabilities | `AtlasClassifier(combiner=Convex(class_weight=None))` |
| n is in the tens of thousands | `AtlasRegressor(approx="auto", fit_cap=200)` |

Everything below is why, and what it costs.

---

## Rows independent, or rows in order?

This is the only decision that can silently invalidate a model, so make it first.

If your rows are a **sequence** — anything with a timestamp, a sensor trace, a market series —
say so. A shuffled split lets a row be predicted by its own neighbours in time, and the model
then learns to lean on information it will not have at deployment.

```python
import numpy as np
from atlas import AtlasRegressor, Blocked

rng = np.random.default_rng(0)
t = np.arange(600)
series = np.sin(2 * np.pi * t / 24) + 0.01 * t + 0.3 * rng.normal(size=600)
X = np.column_stack([np.roll(series, k) for k in (1, 2, 3, 24)])[48:]
y = series[48:]

m = AtlasRegressor(stack=Blocked(n_splits=5, embargo=24)).fit(X, y)
```

`embargo` is a gap purged around every fold, in rows. Set it to one cycle of your data — 24 for
hourly data with a daily cycle, 7 for daily data with a weekly one. Measured on a synthetic
series with autocorrelated features *and* autocorrelated errors, a shuffled stack overstates its
own quality by a wide margin and produces a genuinely worse model; `Blocked` errs conservative
on both counts.

Declaring an ordered stack also changes the default library — see the table further down.

`RollingOrigin` is the same idea with only past data in each fold, which is stricter and costs
you the earliest rows.

## How the model checks itself: `auto`, `cv`, `blocked`

The combiner's weights have to be learned from predictions that did **not** see the row they
are predicting. There are three ways to arrange that, and they are not interchangeable.

### `stack="auto"` — the default, and no folds at all

One library is fitted on all the data, and each row's prediction is recomputed *without that
row*, in closed form. For the neighbourhood members that means dropping the row from its own
neighbour search — which for a neighbourhood estimator **is** leaving it out, not an
approximation of it. For the parametric members it is the hat matrix, which is exact.

Use it because it is cheaper (one library instead of `n_splits + 1`), because it is
deterministic — there is no fold assignment to vary with a seed — and because every member has
seen `n-1` rows, which is the model that actually ships.

### `stack="cv"` — one library per fold

The classical arrangement: cut `n_splits` folds, fit a library on each, predict the fold that
was held out. Slower by a factor of `n_splits + 1` and its fold assignment moves with
`random_state`.

You do not usually choose this; a channel chooses it for you. Any channel without an honest
closed-form leave-one-out forces it, and the estimator will take that path automatically. In the
shipped set that is `Gradient` and `SparseAdditive`.

Choose it deliberately when comparing two configurations against each other, so that both are
measured under the same arrangement.

### `Blocked` / `RollingOrigin` — for rows in order

These refuse the closed-form path outright, and that refusal is the point: excluding row `i`
alone is no exclusion when `t-1` and `t+1` are nearly the same row. They cut contiguous folds
with an embargo band purged around each.

!!! note "Why the default is not cross-validation"
    Moving folds out of the expert matrix made the model deterministic across seeds and left
    held-out quality essentially unchanged, while removing a factor of `n_splits` from the cost.
    Folds did not disappear entirely: `fold_report_` is still cut over them, so that one report
    still moves with `random_state`.

## Categorical columns

Pass a DataFrame rather than one-hot encoding anything yourself — that part matters a great deal,
and [preparing your data](data.md) shows why. Whether to *declare* the columns is a smaller
question and an empirical one: it helps on some datasets and hurts on others, so fit it both ways
and keep the better.

```python
import numpy as np, pandas as pd
from atlas import AtlasRegressor

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.normal(size=400), "kind": rng.choice(list("abcd"), size=400)})
df["kind"] = df["kind"].astype("category")
y = df["x"].to_numpy() + (df["kind"].cat.codes.to_numpy() % 2) * 1.5

m = AtlasRegressor(cat_features="auto").fit(df, y)
```

`cat_features="auto"` takes every `category`, `object` and `bool` column. Pass a list of names
or indices to be explicit — that is also how you declare some columns and not others. Declared
columns are encoded by target statistics fitted **inside the fold**, which is what keeps the
encoding honest; that encoder is an estimate too, which is why declaring is not free.

## How much data you have

Sample size is the single strongest predictor of whether this model is the right choice.

| n | what to expect | what to set |
|---|---|---|
| a few hundred | this is where the model is strongest against boosted trees | defaults |
| a few thousand | roughly at parity; still worth it for attribution or as a blend partner | defaults |
| tens of thousands | a boosted tree will usually be better and much faster to predict | `approx="auto"`, `fit_cap=200` |

`fit_cap` thins the sample *inside* a neighbourhood by striding through the sorted neighbours,
keeping the bandwidth. It is on by default at 200 and buys a large speed-up for a negligible
change in quality. `approx="auto"` switches on an approximate neighbour index above twenty
thousand rows; it is approximate about *which* neighbours, never about how many.

## Wide data

At high `p` the additive channel's knot budget collapses and it degenerates toward a ridge on
nearly-linear columns. Two things help:

```python
import numpy as np
from atlas import AtlasRegressor, SparseAdditive, default_channels

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 60))
y = np.tanh(X[:, 0]) + 0.8 * X[:, 1] + 0.3 * rng.normal(size=400)

m = AtlasRegressor(channels=[*default_channels("regression"), SparseAdditive()]).fit(X, y)
```

`SparseAdditive` drops whole features rather than shrinking them, and it is the one optional
channel measured to pay on regression *and* classification. It is off by default for one reason
only: it has no honest closed-form leave-one-out, so switching it on moves the whole model onto
the cross-validated path. Turn it on when `p` is large relative to `n` and you can afford that.

Also relevant at high `p`: the gradient channel declines itself when a local linear fit would be
unidentified, and says so. That is expected, not a failure.

## Heavy tails and contaminated errors

If some of your targets are simply *wrong* — a mis-keyed value, a sensor dropout, a fat-tailed
tick — the local fits chase them, because least squares does.

```python
from atlas import AtlasRegressor, RobustLocal, default_channels

m = AtlasRegressor(channels=[*default_channels("regression"), RobustLocal()]).fit(X, y)
```

`RobustLocal` refits the same neighbourhoods under a Huber loss. Under genuine contamination it
is worth a great deal; on twenty ordinary benchmark datasets the regime simply is not there, and
it costs a few times the fit. Switch it on when you have reason to believe the errors are
contaminated, not on the chance that they are. It is regression-only — the classifier's local
fit already has a bounded loss.

## Ordered data: the two channels for time

Everything else in the library looks *across* features. These two look *along* the rows, and
they are the only members that carry information about time.

```python
import numpy as np
from atlas import AtlasRegressor, Blocked, Reservoir, TemporalScale, default_channels

rng = np.random.default_rng(0)
season, n = 24, 900
t = np.arange(n + 2 * season)
series = 10 + 3 * np.sin(2 * np.pi * t / season) + 0.01 * t + rng.normal(size=len(t))
X = np.column_stack([np.roll(series, k) for k in (1, 2, 3, season, 2 * season)])[2 * season:]
y = series[2 * season:]

m = AtlasRegressor(
    channels=[*default_channels("regression", ordered=True),
              TemporalScale(season=season), Reservoir()],
    stack=Blocked(n_splits=5, embargo=season),
).fit(X, y)
```

`TemporalScale` fits the design aggregated over a ladder of windows that are **multiples of the
season**, which is why it needs `season` and why it refuses to guess it — estimating the cycle
from the data would be a search. `Reservoir` carries a fixed recurrent state, so it summarises
earlier rows in a way no feature of the current row does.

Both refuse exchangeable rows outright, and both say so rather than fitting quietly. Do not
switch on both expecting them to add up: they draw on the same past, and together they are not
better than the better one alone.

## Probabilities you can read as probabilities

Every member is fitted with balanced class weights, so its output sits on a 50/50 prior rather
than your data's base rate. The combiner is the one stage with a free intercept and therefore
the only place the prior can be corrected.

```python
import numpy as np
from atlas import AtlasClassifier, Convex

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 6))
y = (rng.random(400) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)

m = AtlasClassifier(combiner=Convex(class_weight=None)).fit(X, y)   # calibrated probabilities
```

The default (`"balanced"`) is what every published number here was measured under. Ranking
metrics never see the difference — the shift is monotone — so this changes calibration, not AUC.

## When exact attribution is the point

`attribution()` splits every prediction into one number per feature, exactly, and the identity
it guarantees is checked to machine precision. Some members cannot participate: a neighbour
vote, a single-index ridge function, a surface over a projection. They are booked as *opaque*
and reported as such rather than being quietly folded into a feature.

If attribution is the reason you are here, prefer channels that decompose — `Additive`,
`Pairwise`, `SparseAdditive`, `TemporalScale` — and read `opaque_share` to see how much of the
prediction the split does not cover.

## Which channels are on, and when

| channel | default | why |
|---|---|---|
| `Additive` | always on | carries axis-aligned structure at the one-dimensional rate whatever `p` is |
| `Pairwise` | regression only | clearly pays on regression; on classification it costs about twice the fit for nothing |
| `SingleIndex` | regression, **exchangeable rows only** | measured to hurt on series, where lag features already encode the dynamics |
| `SparseAdditive` | off | pays on both tasks, but forces the cross-validated stack |
| `Surface` | off | large capability on smooth low-dimensional structure; that regime is rare in practice |
| `RobustLocal` | off | large capability under contamination; that regime is rare in practice |
| `TemporalScale`, `Reservoir` | off, ordered rows only | need a season and an ordered stack, and refuse to guess either |
| `Gradient` | off | its wins are on synthetic structure, and it is a large fraction of a fit |

A channel that cannot build anything on your data will tell you so, by name and with the reason.
That warning is information, not noise: it usually means a setting is missing.
