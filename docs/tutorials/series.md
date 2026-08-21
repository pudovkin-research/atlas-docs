<p align="right" class="gh-only">
  <a href="../index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../assets/atlas-mark-dark.svg">
      <img src="../assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Time series

The model has no forecasting front door: you build lag features, it fits a table. Two decisions
then decide the result, and both were settled on ten real series across seven domains.

## 1. Use an ordered stack

Rows in a series are not exchangeable, and a shuffled split does not merely report a wrong
number — it produces a worse model, because the combiner spends weight on a leak that will not
be there at deployment.

```python
from atlas import AtlasRegressor, Blocked, RollingOrigin

AtlasRegressor(stack=Blocked(n_splits=5, embargo=24))        # purged k-fold, in row order
AtlasRegressor(stack=RollingOrigin(n_splits=5, embargo=24))  # train on the past only
```

`embargo` is in **rows** and should cover the horizon over which the target stays correlated
with itself — one cycle is a good default: 24 for hourly data with a daily rhythm, 7 for daily
data with a weekly one.

Measured on a series with autocorrelated features *and* autocorrelated errors, a shuffled stack
does both bad things at once: it reports a future far better than the one it gets, and the model
it produces is genuinely worse than the one an ordered stack produces. `Blocked` errs
conservative on both counts. `RollingOrigin` is stricter still, and its own report is a lower
bound rather than an estimate — its first folds train on a fraction of the series by
construction.

Both refuse the closed-form leave-one-out path, because excluding row `i` alone is no exclusion
when `t-1` and `t+1` are nearly the same row.

## 2. Feed it a delay vector, not a feature dump

This is the single biggest lever, and it is not obvious. A tree splits on the informative
coordinates and ignores the rest; a metric method computes a distance in *all* of them, so every
uninformative lag dilutes the ones that matter.

```python
import numpy as np, pandas as pd

def delay_design(y, season, lags=(1, 2, 3)):
    cols, names = [], []
    for L in (*lags, season, 2 * season):
        cols.append(np.roll(y, L)); names.append(f"y_lag{L}")
    pos = np.arange(len(y)) % season
    cols += [np.sin(2 * np.pi * pos / season), np.cos(2 * np.pi * pos / season)]
    names += ["season_sin", "season_cos"]
    X = pd.DataFrame(np.column_stack(cols), columns=names)
    warm = 2 * season                       # rows whose lags wrapped around
    return X.iloc[warm:].reset_index(drop=True), y[warm:]
```

Measured on ten real series across seven domains: on a delay design this model is ahead of
boosting at both sample sizes tried, and on a wide feature dump it falls to parity or behind.
The boosters barely notice the difference between the two designs; this model moves a great
deal. Series that carry genuine covariates — weather, calendar, holidays — are the exception
and prefer the wider design.

## 3. Consider an L1 combiner

On a lag design the members are weak and nearly redundant, so pruning is most of what a combiner
can usefully do:

```python
from atlas import ElasticNet

AtlasRegressor(stack=Blocked(n_splits=5, embargo=24), combiner=ElasticNet(alpha=0.5))
```

It helps here, keeping roughly a third of the members where ridge keeps all of them. On ordinary
tabular data the same change *costs* a little, because there the members genuinely differ — so
this is a setting for series, not a better default. `alpha=1` is refused: a pure lasso is convex
but not *strictly* convex, so on near-duplicate rungs its answer depends on the order of the
columns.

## 4. The two channels that carry time

Everything else in the library looks *across* features, and a lag design already contains that
kind of structure. These two look *along* the rows:

```python
from atlas import Reservoir, TemporalScale, default_channels

season = 24
channels = [*default_channels("regression", ordered=True),
            TemporalScale(season=season),      # the design aggregated over multiples of a cycle
            Reservoir()]                       # a fixed recurrent state over the rows
```

`TemporalScale` needs the season and refuses to guess it — reading the cycle off the data would
be a search. Its windows are multiples of that cycle, and they have to be: short windows in
absolute rows are just smoothed lags, which the design already gives you.

`Reservoir` is an echo state network with the parts this library cannot have removed: the
reservoir is a fixed deterministic cycle rather than a random draw, and the ridge penalty is a
ladder rather than a choice. It is the steadier of the two, and the only member here that
improves as `n` grows.

Do not switch both on expecting them to add up. They read the same past, and together they are
not better than the better one alone.

!!! note "The default library is different on ordered data"
    `SingleIndex` is on by default for regression and **out** of the default when the stack is
    ordered: on real series it is measured to hurt, because lag features already encode the
    dynamics linearly. The estimator says so when it drops it.

## Judge it with MASE, not R²

On an autocorrelated series, R² against the training mean is easy to make excellent by
predicting almost the last value. A model can hold an excellent R² and a MASE above one at the
same time — and the MASE is the one telling you it is useless.

```python
def mase(y_true, pred, y_train, season):
    scale = np.abs(y_train[season:] - y_train[:-season]).mean()   # scaled on TRAINING rows
    return float(np.abs(y_true - pred).mean() / max(scale, 1e-12))
```

## What to reach for, in order

1. An ordered stack. Nothing else matters if this is wrong.
2. A delay design rather than every lag you can compute.
3. `TemporalScale` with your season, if you know it.
4. `ElasticNet(alpha=0.5)` as the combiner.
5. `Reservoir()`, which helps most on the longer series.

`benchmarks/series.py` runs the comparison this page summarises, on ten real series against
gradient boosting.
