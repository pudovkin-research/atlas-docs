<p align="right" class="gh-only">
  <a href="index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-mark-dark.svg">
      <img src="assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# API reference

Generated from the code by `docs/build.py`.

## `AtlasClassifier`

### Methods

| call | returns |
|---|---|
| `attribution(X) -> 'dict'` | Exactly-attributable share of each prediction, on the log-odds scale. |
| `expert_matrix(X) -> 'np.ndarray'` | Every expert's prediction for these rows, unmixed: `(n, len(expert_names_))`. |
| `export_summary(path)` | Params, expert weights and the cross-validated table, as JSON. |
| `fit(X, y)` | Prepare -> resolve the ladder -> honest expert matrix -> combiner -> final library. |
| `get_params(deep: 'bool' = True) -> 'dict'` | The constructor arguments, as given — the sklearn contract, without sklearn. |
| `predict(X) -> 'np.ndarray'` | Labels in the same domain as the `y` passed to `fit`. |
| `predict_proba(X) -> 'np.ndarray'` | `(n, 2)` class probabilities. On the default combiner these carry a 50/50 prior. |
| `save_model(path)` | Pickle the fitted model. The neighbour index is left out and rebuilt on first use. |
| `set_params(**params)` | Set constructor arguments in place and return self, for `clone` and grid tools. |
| `summary() -> 'str'` | A human-readable line or three: experts, fold metrics, the weights that mattered. |

### Fitted attributes

| attribute | holds |
|---|---|
| `channels_` | The channel instances this model actually fitted -- copies of what was handed in. |
| `combiner_` | The combiner stage, as configured and used. |
| `decision_` | The rule that turned honest probabilities into `threshold_`. |
| `expert_weights_` | — |
| `feature_importances_` | — |
| `library_` | The fitted library: `channels_`, `names_`, the metric, the ladder `ks_`. |
| `opg_spectrum_` | — |
| `classes_` | The labels as given to `fit`, in sorted order |
| `threshold_` | The decision threshold `predict` uses |
| `oof_` | The honest expert matrix the combiner was fitted on |
| `expert_names_` | Column names of that matrix |
| `fold_report_` | Per-fold metrics — a diagnostic, never a generalisation estimate |
| `ladle_dim_` | Structural dimension the ladle named, if asked |

## `AtlasRegressor`

### Methods

| call | returns |
|---|---|
| `attribution(X) -> 'dict'` | The share of each prediction that decomposes exactly over features. |
| `expert_matrix(X) -> 'np.ndarray'` | Every expert's prediction for these rows, unmixed: `(n, len(expert_names_))`. |
| `export_summary(path)` | Params, expert weights and the cross-validated table, as JSON. |
| `fit(X, y)` | Prepare -> resolve the ladder -> honest expert matrix -> combiner -> final library. |
| `get_params(deep: 'bool' = True) -> 'dict'` | The constructor arguments, as given — the sklearn contract, without sklearn. |
| `predict(X) -> 'np.ndarray'` | Predictions in the target's own units, clamped to each neighbourhood's own range. |
| `save_model(path)` | Pickle the fitted model. The neighbour index is left out and rebuilt on first use. |
| `set_params(**params)` | Set constructor arguments in place and return self, for `clone` and grid tools. |
| `summary() -> 'str'` | — |

### Fitted attributes

| attribute | holds |
|---|---|
| `channels_` | The channel instances this model actually fitted -- copies of what was handed in. |
| `combiner_` | The combiner stage, as configured and used. |
| `expert_weights_` | — |
| `feature_importances_` | — |
| `library_` | The fitted library: `channels_`, `names_`, the metric, the ladder `ks_`. |
| `opg_spectrum_` | — |
| `oof_` | The honest expert matrix the combiner was fitted on |
| `expert_names_` | Column names of that matrix |
| `fold_report_` | Per-fold metrics — a diagnostic, never a generalisation estimate |
| `ladle_dim_` | Structural dimension the ladle named, if asked |

## Metrics

`from atlas import ...` — every metric takes `(y_true, y_pred)` and returns a float.

```
default_channels, roc_auc, average_precision, brier, balanced_accuracy, f1, mcc, precision, recall, sensitivity, specificity, npv, fbeta, cohen_kappa, compute_all, grid_threshold, r2, rmse, mae, compute_all_regression
```

`compute_all(y, p, threshold)` and `compute_all_regression(y, p)` return every metric for a task at once, as a dict.

