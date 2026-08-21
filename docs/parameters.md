<p align="right" class="gh-only">
  <a href="index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-mark-dark.svg">
      <img src="assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Parameter reference

Generated from the signatures by `docs/build.py`; the prose lives in
`docs/_descriptions.py`, and `TestDocsCoverParameters` fails the build if the two
come apart. Every default here is what the published numbers were measured under.

## `AtlasClassifier`

Binary classifier. sklearn-compatible: fit / predict / predict_proba.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `scales` | `'auto'` | The ladder of neighbourhood sizes `k`. `"auto"` derives it from the sample size and an unsupervised intrinsic-dimension estimate; `"fixed"` is the hand-picked `(10, 40, 160, 640)`; a sequence sets the rungs yourself. | Rarely. The ladder exists so that no single `k` has to be chosen — every rung is an expert and the combiner mixes them, and the model is insensitive to where the whole ladder sits. |
| `metric` | `'supervised'` | The space neighbours are measured in: `"supervised"` scales each axis by the global fit's `|beta|`, `"standard"` is a plain standardisation, `"none"` uses raw units. | Leave it. The supervised metric clearly beat the unsupervised rescalings it replaced, and it still wins on lag designs where one axis dominates. |
| `average` | `True` | Whether the neighbour vote (`vote@k`, classification) or neighbourhood mean (`mean@k`, regression) enters as an expert at each rung. | Turn it off only if you need every column to decompose per feature — the vote is the one member that does not, and `attribution()` reports it as opaque. |
| `channels` | `'default'` | The library above the neighbourhood spine: `"default"` asks `atlas.default_channels(task)`, or pass a list of channel objects. `[]` leaves the spine alone. | To add `Gradient()` for smooth low-dimensional structure, to drop `Pairwise()` when fit time matters, or to add a channel of your own. |
| `cat_features` | `None` | Categorical columns, by index or by name, or `"auto"` for a DataFrame. They become smoothed out-of-fold target statistics, fitted inside the fold. | Whenever you have categoricals. Untold, they are treated as numbers and their codes become distances. |
| `cat_scale` | `'logodds'` | The scale the target statistic is encoded on: `"logodds"` (weight of evidence) or `"prob"`. | Leave it. Regression always uses `"prob"` — a continuous target has no odds. |
| `cat_count` | `False` | Append `log(1 + n_level)` per categorical column, so the model can see how well supported each level was. | On high-cardinality columns with only a handful of rows per level, where it helps level, and nothing once levels are well supported. |
| `fit_cap` | `200` | The largest number of neighbours a single local fit uses. Above it the rung is thinned by striding through the sorted neighbours, which keeps the bandwidth. | Raise it only if fit time is irrelevant and you want the last decimal; the default is worth 2.4x on classification for that much. |
| `approx` | `False` | Approximate neighbour search: `False` (exact, the default), `True`, or `"auto"` which switches on at n >= 20 000. | At very large n where a 4–19% faster fit is worth perturbing every expert column slightly. Held-out quality was identical to four decimals when measured. |
| `C_local` | `0.3` | Inverse L2 penalty for the local fits at every rung. | Rarely — swept on real data with a held-out pool and nothing beat the default consistently. |
| `C_global` | `0.3` | Inverse L2 penalty for the global fit, the supervised metric that reuses its coefficients, and the parametric channels. | Rarely, and for the same reason. |
| `combiner` | `None` | How the experts are weighed: `Convex(C, class_weight)` by default, or `ElasticNet(C, alpha)` to add an L1 term. | `ElasticNet(alpha=0.5)` on lag designs, where members are weak and redundant. `Convex(class_weight=None)` when you need a calibrated probability. |
| `decision` | `None` | How a probability becomes a label: `Objective("ba")` maximises an objective on leave-one-out scores, `Fixed(0.5)` uses the number you give it. | `Fixed` when the cost matrix is decided outside the model; `Objective("f1")` or a callable when balanced accuracy is not what you optimise. |
| `stack` | `'auto'` | How the honest expert matrix is built: `"auto"` / `"loo"` / `"cv"`, or a stack object. Ordered data wants `Blocked` or `RollingOrigin`. | Always, when rows are not exchangeable. A shuffled split on an autocorrelated series overstates held-out R2 by 0.40 and leaves a worse model behind. |
| `verbose` | `True` | Whether the estimator logs progress. Messages go through `logging`, never `print`. | Set `False` in a loop; use `atlas.show()` to wire a console handler when your application has configured none. |
| `random_state` | `42` | Seed for the fold assignments and the categorical encoder's internal splits. | It does not change the model on the default stack: weights, probabilities and the threshold are all seed-free. It moves `fold_report_` and the cross-validated path. |

## `AtlasRegressor`

Regressor over the same library. sklearn-compatible: fit / predict.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `scales` | `'auto'` | The ladder of neighbourhood sizes `k`. `"auto"` derives it from the sample size and an unsupervised intrinsic-dimension estimate; `"fixed"` is the hand-picked `(10, 40, 160, 640)`; a sequence sets the rungs yourself. | Rarely. The ladder exists so that no single `k` has to be chosen — every rung is an expert and the combiner mixes them, and the model is insensitive to where the whole ladder sits. |
| `metric` | `'supervised'` | The space neighbours are measured in: `"supervised"` scales each axis by the global fit's `|beta|`, `"standard"` is a plain standardisation, `"none"` uses raw units. | Leave it. The supervised metric clearly beat the unsupervised rescalings it replaced, and it still wins on lag designs where one axis dominates. |
| `average` | `True` | Whether the neighbour vote (`vote@k`, classification) or neighbourhood mean (`mean@k`, regression) enters as an expert at each rung. | Turn it off only if you need every column to decompose per feature — the vote is the one member that does not, and `attribution()` reports it as opaque. |
| `channels` | `'default'` | The library above the neighbourhood spine: `"default"` asks `atlas.default_channels(task)`, or pass a list of channel objects. `[]` leaves the spine alone. | To add `Gradient()` for smooth low-dimensional structure, to drop `Pairwise()` when fit time matters, or to add a channel of your own. |
| `cat_features` | `None` | Categorical columns, by index or by name, or `"auto"` for a DataFrame. They become smoothed out-of-fold target statistics, fitted inside the fold. | Whenever you have categoricals. Untold, they are treated as numbers and their codes become distances. |
| `cat_count` | `False` | Append `log(1 + n_level)` per categorical column, so the model can see how well supported each level was. | On high-cardinality columns with only a handful of rows per level, where it helps level, and nothing once levels are well supported. |
| `fit_cap` | `200` | The largest number of neighbours a single local fit uses. Above it the rung is thinned by striding through the sorted neighbours, which keeps the bandwidth. | Raise it only if fit time is irrelevant and you want the last decimal; the default is worth 2.4x on classification for that much. |
| `approx` | `False` | Approximate neighbour search: `False` (exact, the default), `True`, or `"auto"` which switches on at n >= 20 000. | At very large n where a 4–19% faster fit is worth perturbing every expert column slightly. Held-out quality was identical to four decimals when measured. |
| `C_local` | `0.3` | Inverse L2 penalty for the local fits at every rung. | Rarely — swept on real data with a held-out pool and nothing beat the default consistently. |
| `C_global` | `0.3` | Inverse L2 penalty for the global fit, the supervised metric that reuses its coefficients, and the parametric channels. | Rarely, and for the same reason. |
| `combiner` | `None` | How the experts are weighed: `Convex(C, class_weight)` by default, or `ElasticNet(C, alpha)` to add an L1 term. | `ElasticNet(alpha=0.5)` on lag designs, where members are weak and redundant. `Convex(class_weight=None)` when you need a calibrated probability. |
| `stack` | `'auto'` | How the honest expert matrix is built: `"auto"` / `"loo"` / `"cv"`, or a stack object. Ordered data wants `Blocked` or `RollingOrigin`. | Always, when rows are not exchangeable. A shuffled split on an autocorrelated series overstates held-out R2 by 0.40 and leaves a worse model behind. |
| `verbose` | `True` | Whether the estimator logs progress. Messages go through `logging`, never `print`. | Set `False` in a loop; use `atlas.show()` to wire a console handler when your application has configured none. |
| `random_state` | `42` | Seed for the fold assignments and the categorical encoder's internal splits. | It does not change the model on the default stack: weights, probabilities and the threshold are all seed-free. It moves `fold_report_` and the cross-validated path. |

## Channels

The library above the spine. Pass a list as `channels=`.

### `Additive`

```python
Additive(knots=(5, 9, 17))
```

Per-feature piecewise-linear shape functions at several knot counts.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `knots` | `(5, 9, 17)` | Knot counts for the per-feature piecewise-linear shape functions. Each count is a separate expert. | Smoothness is a ladder for the same reason `k` is. `()` switches the channel off without removing it from the list. |

### `Pairwise`

```python
Pairwise(pairs=(4, 16), knots=(5,))
```

GA2M: main effects plus the top-K pairwise tensor-product surfaces.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `pairs` | `(4, 16)` | Ladder over K, the number of pairwise tensor-product surfaces kept. Candidate pairs are ranked by FAST on the additive fit's residual. | It pays clearly on regression and not on classification, which is why it is on by default for one and not the other. |
| `knots` | `(5,)` | Knot count for the ramps the pair surfaces are built from. | Rarely; the channel switches itself off when the degrees-of-freedom budget leaves no room for a surface. |

### `SparseAdditive`

```python
SparseAdditive(fractions=(0.4, 0.15, 0.05), knots: 'int' = 9)
```

Additive shape functions under a per-feature group-lasso penalty.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `fractions` | `(0.4, 0.15, 0.05)` | Ladder over the group-lasso penalty, as fractions of the lambda that would zero the fit. One group per feature, so whole features leave rather than being shrunk. | High `p`, where the additive channel's knot budget collapses toward two and it degenerates into a ridge on nearly-linear columns. The one optional channel measured to pay on regression AND classification, at small sample sizes. |
| `knots` | `9` | Knot count for the shape functions, capped by the same degrees-of-freedom budget the additive channel uses. | Rarely; at the high `p` this channel is for, the budget decides it anyway. |

### `SingleIndex`

```python
SingleIndex(ranks=(1, 2, 3), knots: 'int' = 9)
```

Ridge functions `g(theta' x)` along directional-regression directions.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `ranks` | `(1, 2, 3)` | Ladder of index ranks. `index@r` fits `sum_{j<=r} g_j(theta_j' x)` on directions read off slice moments in closed form, so no direction is searched. | On by default for regression, where it pays at no measurable cost; off for classification, and out of the default entirely when the rows are a sequence. |
| `knots` | `9` | Knots per index coordinate, cut at quantiles of the projection. | One number rather than a ladder: the rank already is one, and two ladders would multiply the width of a design the solve is cubic in. |

### `Surface`

```python
Surface(ranks=(2, 3), knots: 'int' = 40)
```

One smooth isotropic surface over a learned projection -- the only member that does not factorise into one-dimensional pieces. Bounded kernel by measurement, not by taste, and it keeps the closed-form leave-one-out, which the leakage suite cleared. Off by default: large capability, small measured effect. See `core/surface.py`.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `ranks` | `(2, 3)` | Ladder over the dimension of the projection the surface lives on. The only member that does not factorise into one-dimensional pieces. | Smooth low-dimensional structure that is not axis-aligned -- a bowl or a saddle in a learned plane. Its capability there is large and the regime is rare in practice, which is why it is off; it keeps its closed-form leave-one-out, so switching it on costs its own fit rather than the model's stack. |
| `knots` | `40` | Upper bound on radial basis centres, taken as a deterministic stride through the projected points and capped by the degrees-of-freedom budget. | Rarely; the budget decides it below about n = 200. |

### `RobustLocal`

```python
RobustLocal(scales=(40, 120, 360), c: 'float' = 1.345)
```

Local fits under a Huber loss, over the neighbourhoods the spine already searches.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `scales` | `(40, 120, 360)` | Neighbourhood sizes for the local Huber fits. The same core call the spine uses, with a Huber constant; at zero it reproduces least squares bit for bit. | Errors that are actually contaminated -- sensors, ticks, hand entry. Worth a great deal there and nothing on ordinary benchmark data, which is why it is off by default. It keeps the closed-form leave-one-out, so it costs its fit, not the stack. |
| `c` | `1.345` | The Huber constant, on a MAD scale. The default is the textbook value, chosen for near-full efficiency when the errors are in fact Gaussian. | Rarely; it is a textbook constant, not a knob to tune, and tuning it would be a search. |

### `TemporalScale`

```python
TemporalScale(season: 'int | None' = None, scales=(0.25, 0.5, 1.0, 2.0))
```

The design aggregated over a ladder of time scales -- the one channel that looks along the rows rather than across the features. Needs a `season` and needs ordered rows, and says so rather than guessing either. See `core/temporal.py`.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `season` | `None` | Rows in one cycle. The aggregation levels are multiples of it, and there is no default: estimating the cycle would be a search, so the channel declines without it. | Ordered data whose cycle you know. The channel refuses exchangeable rows outright. |
| `scales` | `(0.25, 0.5, 1.0, 2.0)` | Ladder of aggregation levels, as multiples of the season. Coarse levels hold level and trend and suppress noise; fine levels hold recent dynamics. | Multiples of the season are what carried the effect when it was measured; a ladder of short windows in absolute rows are just smoothed lags, which the design already has. |

### `Reservoir`

```python
Reservoir(penalties=(0.01, 0.1, 1.0, 10.0), units: 'int | None' = None)
```

A fixed deterministic recurrent state, read out by ridge at a ladder of penalties. The only member that carries a summary of earlier rows that no feature of this row contains. Ordered rows only. See `core/reservoir.py`.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `penalties` | `(0.01, 0.1, 1.0, 10.0)` | Ladder of ridge penalties for the readout. The reference implementation of this idea picks one by BIC; selecting is what this library declines to do, so each becomes an expert. | Ordered data where the useful summary of the past is not any single lag. |
| `units` | `None` | Reservoir size. `None` takes 40% of the pinned sample, capped at 200, which is the rule the `echos` reference uses. | Rarely; the reservoir is a fixed deterministic cycle, not a thing to tune. |

### `Gradient`

```python
Gradient(ranks=(2, 8), k_field=60, inner_splits=3, refine=0, refine_ranks='ladle', dr_ranks=())
```

Local slopes -> OPG subspace -> a potential reconstructed by a Laplacian solve.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `ranks` | `(2, 8)` | OPG subspace dimensions to reconstruct a potential in. Each rank is an expert. | The rank is deliberately not selected: it must match the structural dimension of the signal and the spectrum cannot estimate it. |
| `k_field` | `60` | Neighbours per local slope estimate. | It must exceed `p`: a local linear fit on fewer points is unidentified, and the channel declines itself (for regression) rather than integrating noise. |
| `inner_splits` | `3` | Folds for the internal CV the integration anchors come from. | Rarely. Integrating over rows whose anchor is their own target leaks, and the leak is invisible in out-of-fold metrics. |
| `refine` | `0` | MAVE refinement passes. `0` is one-shot OPG; higher re-ranks neighbourhoods inside the subspace found so far. | When one-shot OPG returns a flat spectrum — refinement rescued R2 0.05 -> 0.87 on a high-frequency target with ten noise dimensions. |
| `refine_ranks` | `'ladle'` | Ranks for the refined basis, or `"ladle"` to estimate one from the stability of the leading eigenspace under bootstrap. | Leave it at `"ladle"`; it is resolved once per model, because per fold it would build expert matrices of different widths. |
| `dr_ranks` | `()` | Ranks for a third basis from directional regression, an inverse method that costs 0.5 ms rather than tens of milliseconds. | Off: it costs seconds and buys nothing measurable. |

## Stacks

How the honest expert matrix is built. Pass one as `stack=`.

### `Auto`

```python
Auto(n_splits: 'int' = 5)
```

One library and closed-form leave-one-out wherever every channel provides it.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `n_splits` | `5` | Number of folds. Decides the fold report, the cross-validated path, and the cap on the neighbourhood ladder. | Lower it on small samples so that each fold still has rows to fit on. |

### `LeaveOneOut`

```python
LeaveOneOut(n_splits: 'int' = 5)
```

Force the closed-form path. A channel that cannot do it is an error, not a fallback.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `n_splits` | `5` | Number of folds. Decides the fold report, the cross-validated path, and the cap on the neighbourhood ladder. | Lower it on small samples so that each fold still has rows to fit on. |

### `CrossValidated`

```python
CrossValidated(n_splits: 'int' = 5)
```

One library per fold. What `stack="cv"` meant.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `n_splits` | `5` | Number of folds. Decides the fold report, the cross-validated path, and the cap on the neighbourhood ladder. | Lower it on small samples so that each fold still has rows to fit on. |

### `Blocked`

```python
Blocked(n_splits: 'int' = 5, embargo: 'int' = 0)
```

Contiguous folds in row order, with an embargo band purged around each one.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `n_splits` | `5` | Number of folds. Decides the fold report, the cross-validated path, and the cap on the neighbourhood ladder. | Lower it on small samples so that each fold still has rows to fit on. |
| `embargo` | `0` | Rows purged on each side of every validation block. | Set it to the horizon over which the target stays correlated with itself. Zero is plain blocked k-fold: honest about the split, silent about the correlation. |

### `RollingOrigin`

```python
RollingOrigin(n_splits: 'int' = 5, embargo: 'int' = 0, min_train: 'int' = 0)
```

Train on the past only, validate on the block that follows, with an embargo between.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `n_splits` | `5` | Number of folds. Decides the fold report, the cross-validated path, and the cap on the neighbourhood ladder. | Lower it on small samples so that each fold still has rows to fit on. |
| `embargo` | `0` | Rows left between the end of training and the start of the validation block. | Same rule as `Blocked`. |
| `min_train` | `0` | Smallest training block a fold may use; shorter folds are dropped. | Raise it when the earliest folds are too short to be informative — rolling origin's first fold trains on a fraction of the series by construction. |

## Combiners

How the experts are weighed. Pass one as `combiner=`.

### `Convex`

```python
Convex(C: 'float' = 0.1, class_weight: 'str | None' = 'balanced')
```

L2-penalised stacking: logistic on the members' log-odds, ridge on their predictions.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `C` | `0.1` | Inverse L2 penalty of the stacking program. | Defaults differ by task and the asymmetry is measured: 0.03 for classification (log-odds), 0.1 for regression (standardised targets). |
| `class_weight` | `'balanced'` | `"balanced"` keeps the 50/50 prior every expert is fitted under; `None` fits the unweighted likelihood so the intercept recovers the base rate. | `None` when the probability itself is the output. Ranking is identical either way. |

### `ElasticNet`

```python
ElasticNet(C: 'float' = 0.1, alpha: 'float' = 0.5, class_weight: 'str | None' = 'balanced', max_iter: 'int' = 5000, tol: 'float' = 1e-12)
```

The convex program with an L1 term. Not the default, and the measurement says why.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `C` | `0.1` | Inverse L2 penalty of the stacking program. | Defaults differ by task and the asymmetry is measured: 0.03 for classification (log-odds), 0.1 for regression (standardised targets). |
| `alpha` | `0.5` | Split of the penalty between L1 and L2: `0` is ridge, values near `1` are nearly a lasso. `1` is refused. | 0.5 on lag designs, where the members are weak and redundant. On tabular data it costs a little — there the members genuinely differ, and the rule is mix, do not select. |
| `class_weight` | `'balanced'` | `"balanced"` keeps the 50/50 prior every expert is fitted under; `None` fits the unweighted likelihood so the intercept recovers the base rate. | `None` when the probability itself is the output. Ranking is identical either way. |
| `max_iter` | `5000` | Coordinate-descent iteration cap. | Leave it. Strict convexity makes the optimum unique but does not make a solver reach it: near a pure lasso, a loose tolerance lets the answer drift with column order. |
| `tol` | `1e-12` | Coordinate-descent convergence tolerance. | Leave it, for the reason above. |

## Decisions

How a probability becomes a label. Pass one as `decision=` (classifier only).

### `Objective`

```python
Objective(objective='ba')
```

Pick the threshold that maximises an objective on leave-one-out probabilities.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `objective` | `'ba'` | A built-in name from `atlas.BUILTIN_OBJECTIVES` or a callable `(y_true, y_pred) -> float`, maximised on leave-one-out probabilities. | When balanced accuracy is not the thing you are paid for. |

### `Fixed`

```python
Fixed(threshold: 'float' = 0.5)
```

Use this number, whatever the data says. For a cost matrix decided outside the model.

| parameter | default | what it does | when to change it |
|---|---|---|---|
| `threshold` | `0.5` | The decision threshold, used as given. | When the cut is a business decision rather than a fitted one. |

