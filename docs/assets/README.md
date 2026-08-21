# The mark

Three circular charts onto one scalar field. The contours are that field's isolines, so every
line that leaves one chart continues in the next: the charts agree where they overlap, and the
region where all three agree is filled with the accent — that is the combiner.

| file | use |
|---|---|
| `atlas-mark.svg` | the mark, on light |
| `atlas-mark-dark.svg` | the mark, on dark |
| `atlas-mark-mono.svg` | one colour; takes `currentColor`, so it inherits from CSS |
| `atlas-lockup.svg` / `atlas-lockup-dark.svg` | mark, wordmark and attribution line |
| `favicon.svg` | the mark again, at favicon default size |

**One file serves every size.** All fifteen contour levels are present at 16 px as well as at
512; below roughly 24 px they stop resolving as separate lines and settle into tone. That is the
intended behaviour — do not cut a simplified small-size variant, because two assets drift and the
texture is what makes the mark the mark.

**Clear space** is one chart radius on every side. The mark is drawn to a tight bounding box, so
that is a quarter of the artwork's width.

**Do not retighten the viewBox.** The artwork spans x 11.71–88.29, y 9.50–82.25 in the mark's own
100-unit space; the files square that and pad it by one unit. A box any smaller cuts the bottom of
the lower two charts, which is exactly what a plausible-looking `11 7 78 78` did.

**Wordmark**: Space Grotesk 600, letter-spacing 0.09em. Attribution line: "SuperLearner" in
Space Grotesk 500 beside "by pudovkin-research" in IBM Plex Mono 400, muted. The shipped lockups
fall back to Helvetica/Arial where those faces are unavailable.

**Palette**

| token | value | where |
|---|---|---|
| ink | `#12253c` | contours, discs, wordmark |
| accent | `#d16631` | the agreeing region, and nothing else |
| paper | `#fbfaf6` | light ground |
| night | `#0a111a` | dark ground |
| on-night | `#e8e4dc` | contours and type on dark |

The accent is reserved. If something else on a page needs colour, it is not this one.
