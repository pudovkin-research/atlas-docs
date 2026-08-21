<p align="right" class="gh-only">
  <a href="../index.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="../assets/atlas-mark-dark.svg">
      <img src="../assets/atlas-mark.svg" alt="ATLAS" width="52" height="52">
    </picture>
  </a>
</p>

# Extending it

Every stage between the data and the answer is an object you can replace. The library itself is
a list. Every block below runs; this one makes the data they use:

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 6))
y = np.tanh(X[:, 0]) - 0.7 * X[:, 1] + 0.3 * rng.normal(size=400)
y_bin = (rng.random(400) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
```

## A channel

Three methods are required. Two more are *asked for*, and the model adapts to the answer rather
than assuming it:

```python
import numpy as np
from atlas import AtlasRegressor, Additive

class Constant:
    """A deliberately trivial member, to show the contract."""

    def fit(self, space, y):
        self.mu_ = float(np.mean(y))
        return self

    def columns(self):
        return ["constant"]                       # names, in predict_matrix's column order

    def predict_matrix(self, space):
        return np.full((space.n, 1), self.mu_)    # (n, len(columns()))

    # optional ------------------------------------------------------------------
    def loo_matrix(self, space, y):
        """Leave-one-out predictions in closed form, or omit the method entirely.

        Without it the whole model falls back to the cross-validated stack — which is correct,
        not a failure: that is exactly what the gradient channel does.
        """
        return np.full((space.n, 1), self.mu_)

    def decompose(self, space):
        """Per-column split into shape / baseline / surfaces, or omit it.

        Without it `attribution()` reports the column as opaque and says so in `opaque_share`.
        It will never quietly count as explained.
        """
        return None

m = AtlasRegressor(channels=[Additive(), Constant()]).fit(X, y)
m.expert_weights_["constant"]
```

### The `space` argument

`space` carries the two geometries the model keeps apart, and a channel *declares* which it
reads by using it:

| attribute | what it is | what it is right for |
|---|---|---|
| `space.supervised` | axes scaled by the global fit's `\|β\|` | asking who a point's neighbours are |
| `space.standard` | plain standardisation | reading a slope or a shape function off the data |
| `space.raw` | the encoded input, unscaled | when you need the feature's own units |

Plus `space.n`, `space.p`, `space.task`, `space.df_n`, `space.C`, `space.C_local`,
`space.approx`, `space.random_state`, and two that only matter for data in row order:

| attribute | what it is |
|---|---|
| `space.ordered` | whether the rows are a sequence — the stack's own answer |
| `space.order` | where these rows sat in the original series, or `None` for a contiguous block |

`space.order` exists because a fold's *training* rows are not contiguous: `Blocked` cuts a block
out of the middle. A channel that reads along the rows must restart at every break in it, or it
will average — or recurse — straight across the hole.

The supervised metric distorts exactly the geometry a gradient is measured in, which is why the
additive, pairwise and gradient channels all read `space.standard`. That rule used to be a
comment; now it is a choice each channel makes visibly.

### Rules a channel must respect

- **Copies, not sharing.** The estimator deep-copies each channel per library, so a channel may
  keep fitted state without worrying about folds — and the instance you passed in comes back
  unfitted.
- **An empty block is legal.** Returning `(n, 0)` from `predict_matrix` is how a channel switches
  itself off; everything downstream tolerates it.
- **Ladders take `False`/`None`/`()`/an int/a sequence.** Use
  `atlas.core.channels.channel_ladder` for integer ladders and
  `atlas.core.sparse_additive.positive_ladder` for float ones, so a string is refused rather
  than turned into a ladder of characters: `tuple(float(v) for v in "05")` is `(0.0, 5.0)`, and
  it looks like a ladder of numbers.
- **Refuse settings where they are written.** A bad value should raise in `__init__`, not three
  frames down at fit time.
- **If you build nothing, say why.** Set `self.declined_` to a sentence during `fit`, and the
  estimator turns it into a warning naming your channel. A channel that builds nothing in
  silence is indistinguishable from one that is broken.
- **Add an `obstacle(n, p, task)`** returning a reason string when your channel cannot build on
  data of that shape. It is asked *before* anything is fitted, so a channel that will decline
  does not drag the whole model onto the cross-validated stack.
- **Column names must be unique** across the whole library — every weight and every attribution
  view is keyed by them, and the estimator refuses a clash.

## A combiner

```python
class MyCombiner:
    def fit(self, M, y, task):        # M is (n, n_experts)
        ...
        return coef, intercept        # np.ndarray, float

    def apply(self, M, coef, intercept, task):
        ...
```

One constraint, and it is the model's central claim rather than a style preference: **the
program must be strictly convex with a unique optimum.** Fancier combination rules — gates,
routers, simplexes, exponential weights, a GAM, a tree, an MLP — were all measured against plain
convex stacking and none of them beat it, while every form that *routes* rather than mixes was
worse. `ElasticNet` refuses `alpha=1` for the same reason a router is refused: a pure lasso's
optimum on near-duplicate rungs is a *set*, and which point of it comes back depends on column
order.

## A decision rule

```python
from atlas import AtlasClassifier

class Median:
    def threshold(self, y, p):        # p is leave-one-out, never in-sample
        return float(np.median(p))

AtlasClassifier(decision=Median()).fit(X, y_bin)
```

## A stack

```python
class EveryOtherRow:
    wants_loo = False                 # or True to allow the closed-form path
    n_splits = 2                      # used for the ladder cap and the fold report

    def folds(self, y, random_state, task):
        idx = np.arange(len(y))
        return [(idx[idx % 2 == 0], idx[idx % 2 == 1]),
                (idx[idx % 2 == 1], idx[idx % 2 == 0])]
```

The ladder is capped by the size of the *smallest* training fold your stack returns, so a stack
with very uneven folds gets a coarser ladder — deliberately, since otherwise a rung would
collapse inside the narrow fold and that fold would build a narrower expert matrix.

## Checking your work

```python
m = AtlasRegressor(channels=[Additive(), Constant()]).fit(X, y)

assert m.oof_.shape[1] == len(m.expert_names_)      # widths line up
m.attribution(X[:50])["opaque_share"]               # did it decompose, or is it opaque?
print(m.summary())
```

Then fit the same configuration twice and compare predictions: if they differ, something in
your channel is searching or drawing at random, and neither belongs here.

The suite in `tests/test_invariants.py` is the specification of everything above; the channel
protocol itself lives in `atlas/core/channels.py`.
