"""20-80 scale conversion using SD-based standardization (Fangraphs convention).

Reference: https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/
  "50 is major league average, then each 10 point increment represents a
  standard deviation better or worse than average." Three standard
  deviations in either direction covers ~99.7% of the population, which is
  why the scale runs 20-80 rather than 0-100.

The original ``scale_to_2080`` in ``src/baseball2vec/tools.py`` uses
``MinMaxScaler``, which maps the sample min/max to 20/80. That does not
correspond to a fixed number of standard deviations and makes the scale
sensitive to single outliers. This module replaces it with an explicit
mean=50, SD=10 standardization, clipped to [20, 80].
"""

from __future__ import annotations

import pandas as pd

TARGET_MEAN = 50.0
TARGET_SD = 10.0


def scale_to_2080_sd(
    scores_df: pd.DataFrame,
    cols: list[str],
    target_mean: float = TARGET_MEAN,
    target_sd: float = TARGET_SD,
) -> pd.DataFrame:
    """Standardize each column to N(target_mean, target_sd) grades, clipped to [20, 80].

    grade = target_mean + target_sd * (x - mean(x)) / std(x)

    Z-scoring is invariant to any prior positive affine rescaling of *x*
    (e.g. an already-applied MinMax transform), so this produces the same
    grades whether *scores_df* holds a model's raw latent tool scores or an
    affinely-rescaled version of them.

    Args:
        scores_df: DataFrame with raw (or affinely-rescaled) tool score columns.
        cols: Column names to scale.
        target_mean: Scale mean (50, per the scouting-scale convention).
        target_sd: Points per standard deviation (10, per the scouting-scale
            convention — NOT 5).

    Returns:
        New DataFrame with the same columns scaled to [20, 80].
    """
    out = {}
    for c in cols:
        x = scores_df[c].astype(float)
        z = (x - x.mean()) / (x.std() + 1e-8)
        grade = target_mean + target_sd * z
        out[c] = grade.clip(20, 80)
    return pd.DataFrame(out, index=scores_df.index)


def build_scaled_results_sd(
    methods: dict[str, pd.DataFrame],
    tool_names: list[str],
    target_mean: float = TARGET_MEAN,
    target_sd: float = TARGET_SD,
) -> dict[str, pd.DataFrame]:
    """SD-based analogue of ``baseball2vec.baselines.build_scaled_results``.

    Scales each method's raw tool scores to mean=50/SD=10 (clipped to
    [20, 80]) instead of MinMax, and computes Pentagon Area from the result.

    Args:
        methods: Dict mapping method name -> raw tool score DataFrame.
        tool_names: Ordered list of tool column names.

    Returns:
        Dict mapping method name -> DataFrame with prefixed tool columns and
        an ``{method}_Area`` column.
    """
    from baseball2vec.tools import pentagon_area  # requires src/ on sys.path

    results: dict[str, pd.DataFrame] = {}
    for name, raw in methods.items():
        scaled = scale_to_2080_sd(raw, tool_names, target_mean, target_sd)
        out = scaled.copy()
        out.columns = [f"{name}_{c}" for c in tool_names]
        out[f"{name}_Area"] = scaled.apply(pentagon_area, axes=tool_names, axis=1)
        results[name] = out
    return results


def clip_fraction(
    scores_df: pd.DataFrame,
    cols: list[str],
    target_mean: float = TARGET_MEAN,
    target_sd: float = TARGET_SD,
) -> pd.Series:
    """Fraction of rows clipped at 20 or 80 for each column, before clipping.

    Diagnostic for how fat-tailed the real distribution is relative to the
    ~0.3% (3-sigma) that a true normal distribution would clip.
    """
    fracs = {}
    for c in cols:
        x = scores_df[c].astype(float)
        z = (x - x.mean()) / (x.std() + 1e-8)
        grade = target_mean + target_sd * z
        fracs[c] = float(((grade < 20) | (grade > 80)).mean())
    return pd.Series(fracs)
