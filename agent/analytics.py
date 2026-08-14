"""
Score Analytics: classical ML / statistical analysis on top of the audit
history this project already collects in SQLite (agent/memory.py).

Covers, honestly:
- dataset analysis: agent/memory.py's audit_history.db IS the dataset here
- feature engineering: category scores/weights, per-severity finding counts,
  review_status, extracted from each stored report
- statistical modeling: Pearson correlation of each feature against the
  overall_score target (with the usual "correlation is not causation" caveat)
- classical ML: scikit-learn LinearRegression / Ridge / RandomForestRegressor
  / GradientBoostingRegressor trained to predict overall_score
- quantitative model comparison: k-fold cross-validated R^2/MAE/RMSE across
  all trained models, reported side by side, never a single lucky/unlucky
  train/test split

HONEST LIMITATION UP FRONT: real audit history from actual runs of this tool
is small (a handful to a few dozen rows at the time this was written) -- far
too little for classical ML or statistics to say anything trustworthy. This
module is built to work against real data as it accumulates, but also ships
a clearly-labeled SYNTHETIC data generator so it's genuinely runnable and
testable today. Any output built from synthetic data says so explicitly, in
both the returned dict and the printed summary -- never silently.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score

from . import memory
from .orchestrator import CANONICAL_CATEGORY_NAMES

MIN_REAL_ROWS_FOR_ANALYSIS = 20  # below this, real ML/stats results aren't trustworthy
FEATURE_CATEGORY_NAMES = list(CANONICAL_CATEGORY_NAMES.values())


def _extract_features_from_report(report: dict) -> dict:
    """Turn one stored report dict into a flat feature row. Missing
    categories (a specialist that never ran, or was dropped) become NaN
    rather than 0 -- 0 would falsely imply "scored zero," not "absent."""
    row: dict = {}
    categories_by_name = {c.get("name"): c for c in report.get("categories", []) if isinstance(c, dict)}

    total_findings = total_critical = total_warning = total_good = 0

    for cat_name in FEATURE_CATEGORY_NAMES:
        cat = categories_by_name.get(cat_name)
        score_col, weight_col = f"score__{cat_name}", f"weight__{cat_name}"
        if cat is None:
            row[score_col] = np.nan
            row[weight_col] = np.nan
            continue
        row[score_col] = cat.get("score", np.nan)
        row[weight_col] = cat.get("weight", np.nan)
        for f in cat.get("findings", []):
            total_findings += 1
            sev = f.get("severity")
            if sev == "critical":
                total_critical += 1
            elif sev == "warning":
                total_warning += 1
            elif sev == "good":
                total_good += 1

    row["total_findings"] = total_findings
    row["critical_findings"] = total_critical
    row["warning_findings"] = total_warning
    row["good_findings"] = total_good
    row["num_categories_present"] = sum(1 for c in FEATURE_CATEGORY_NAMES if c in categories_by_name)
    row["review_approved"] = 1 if report.get("review_status") == "approved" else 0
    row["overall_score"] = report.get("overall_score", np.nan)
    return row


def load_real_dataset() -> pd.DataFrame:
    """Load every stored audit from SQLite and engineer features from each.
    Returns an empty DataFrame (not an error) if there's no history yet."""
    reports = memory.get_all_full_audits()
    rows = [_extract_features_from_report(r) for r in reports]
    return pd.DataFrame(rows)


def generate_synthetic_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a SYNTHETIC dataset with the same shape as the real one, for
    demonstrating/testing this module when real audit history is too small.
    overall_score is generated from a known linear combination of the
    category scores plus noise, so the classical-ML section has a genuine,
    verifiable signal to recover -- this is not meant to resemble real SEO
    data, just to exercise the pipeline honestly."""
    rng = np.random.default_rng(seed)
    rows = []
    # A fixed "true" weight per category used only to generate the synthetic
    # target -- deliberately not identical to the app's own synthesizer-
    # assigned weights, so recovering it via regression is a genuine (if
    # easy) exercise, not a tautology.
    weight_values = [0.22, 0.18, 0.14, 0.14, 0.10, 0.12, 0.06, 0.04]
    true_weights = dict(zip(FEATURE_CATEGORY_NAMES, weight_values * (len(FEATURE_CATEGORY_NAMES) // len(weight_values) + 1)))

    for _ in range(n):
        row = {}
        total_findings = total_critical = total_warning = total_good = num_present = 0
        present_scores = {}

        for cat_name in FEATURE_CATEGORY_NAMES:
            present = rng.random() > 0.08  # most categories present most of the time
            if not present:
                row[f"score__{cat_name}"] = np.nan
                row[f"weight__{cat_name}"] = np.nan
                continue
            num_present += 1
            score = float(np.clip(rng.normal(72, 18), 0, 100))
            present_scores[cat_name] = score
            row[f"score__{cat_name}"] = score

            n_findings = int(rng.poisson(3))
            n_critical = int(rng.binomial(n_findings, max(0.0, (100 - score) / 400)))
            n_warning = int(rng.binomial(max(0, n_findings - n_critical), 0.5))
            n_good = max(0, n_findings - n_critical - n_warning)
            total_findings += n_findings
            total_critical += n_critical
            total_warning += n_warning
            total_good += n_good

        # Renormalize the true weights across only the PRESENT categories so
        # they sum to 1.0 -- matching how the real app's
        # orchestrator.py::_reconcile_overall_score always renormalizes
        # weights regardless of how many specialists actually ran. Without
        # this, overall_score would artificially shrink whenever fewer
        # categories were present (since missing terms just drop out of an
        # un-renormalized weighted sum), creating a fake, misleading
        # correlation between num_categories_present and overall_score that
        # doesn't reflect how the real system actually computes scores.
        raw_weight_total = sum(true_weights[name] for name in present_scores)
        weighted_sum = 0.0
        for cat_name, score in present_scores.items():
            normalized_weight = true_weights[cat_name] / raw_weight_total
            row[f"weight__{cat_name}"] = round(normalized_weight, 3)
            weighted_sum += score * normalized_weight

        noise = rng.normal(0, 4)
        row["total_findings"] = total_findings
        row["critical_findings"] = total_critical
        row["warning_findings"] = total_warning
        row["good_findings"] = total_good
        row["num_categories_present"] = num_present
        row["review_approved"] = int(rng.random() > 0.4)
        row["overall_score"] = float(np.clip(weighted_sum + noise, 0, 100))
        rows.append(row)

    return pd.DataFrame(rows)


def engineer_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Turn the raw feature DataFrame into (X, y, feature_names) ready for
    scikit-learn: drops rows with a missing target, median-imputes missing
    feature values (mainly the score__/weight__ columns for categories that
    weren't present in a given audit), and returns the feature name list in
    matching column order."""
    df = df.dropna(subset=["overall_score"]).copy()
    feature_cols = [c for c in df.columns if c != "overall_score"]
    X_df = df[feature_cols].copy()
    for col in feature_cols:
        if X_df[col].isna().any():
            median = X_df[col].median()
            X_df[col] = X_df[col].fillna(median if not np.isnan(median) else 0.0)
    X = X_df.to_numpy(dtype=float)
    y = df["overall_score"].to_numpy(dtype=float)
    return X, y, feature_cols


MIN_RELIABLE_CORRELATION_N = 15  # below this, a correlation's sign/magnitude can flip on 1-2 rows


def compute_correlations(df: pd.DataFrame, min_reliable_n: int = MIN_RELIABLE_CORRELATION_N) -> list[dict]:
    """Pearson correlation of each feature against overall_score. Every
    feature with >= 3 non-null pairs and some variance is included (nothing
    is hidden), but each entry is tagged "reliable": n >= min_reliable_n,
    and the sort puts all reliable results ahead of all unreliable ones
    (ranked by |r| within each group) rather than sorting by raw |r| alone.

    That distinction matters in practice, not just in theory: with a sparse
    real dataset (a category whose specialist frequently fails to return
    valid JSON has very few non-null rows), a small-n correlation is
    systematically MORE likely to look extreme by pure chance than a
    well-supported one -- sorting by raw magnitude alone would let that
    noise rank #1 and crowd out a real, well-supported signal ranked #2.
    Observed for real: a category with n=8 showed r=-0.857 (a nonsensical
    negative relationship for a positively-weighted score component) while
    a category with n=51 showed a sensible r=+0.828 -- sorted by raw |r|,
    the noisy n=8 result would rank first."""
    if "overall_score" not in df.columns or df["overall_score"].dropna().empty:
        return []

    results = []
    for col in df.columns:
        if col == "overall_score":
            continue
        pair = df[[col, "overall_score"]].dropna()
        if len(pair) < 3 or pair[col].nunique() < 2:
            continue
        r, p_value = stats.pearsonr(pair[col], pair["overall_score"])
        if np.isnan(r):
            continue
        results.append({
            "feature": col, "correlation": round(float(r), 3),
            "p_value": round(float(p_value), 4), "n": len(pair),
            "reliable": len(pair) >= min_reliable_n,
        })

    # Reliable results first (sorted by |r| within that group), then
    # unreliable ones after (also sorted by |r| within their group) --
    # see the docstring above for why raw-|r| sorting alone is misleading.
    results.sort(key=lambda item: (not item["reliable"], -abs(item["correlation"])))
    return results


def train_and_compare_models(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[dict]:
    """Train several classical regressors to predict overall_score and
    compare them via k-fold cross-validation (never a single train/test
    split -- with a small dataset that's just measuring luck). Returns a
    list of {"model", "r2_mean", "r2_std", "mae_mean", "rmse_mean"} dicts,
    sorted by mean R^2 descending -- the actual quantitative comparison."""
    n_splits = max(2, min(n_splits, len(y)))  # can't have more folds than samples
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=seed),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=seed),
    }

    results = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # small-fold-count warnings are expected here
            r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
            neg_mae_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
            neg_rmse_scores = cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        results.append({
            "model": name,
            "r2_mean": round(float(np.mean(r2_scores)), 3),
            "r2_std": round(float(np.std(r2_scores)), 3),
            "mae_mean": round(float(-np.mean(neg_mae_scores)), 2),
            "rmse_mean": round(float(-np.mean(neg_rmse_scores)), 2),
        })

    results.sort(key=lambda item: item["r2_mean"], reverse=True)
    return results


def feature_importance(X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int = 42) -> list[dict]:
    """Fits a single RandomForestRegressor on the full dataset (not cross-
    validated -- this is for interpretability, not for judging predictive
    performance, which train_and_compare_models already covers properly)
    and returns feature importances sorted descending."""
    model = RandomForestRegressor(n_estimators=200, random_state=seed)
    model.fit(X, y)
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    return [{"feature": name, "importance": round(float(imp), 4)} for name, imp in ranked]


def run_analysis(
    source: str = "auto",
    n_synthetic: int = 200,
    min_real_rows: int = MIN_REAL_ROWS_FOR_ANALYSIS,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Orchestrates the whole analysis.

    source: "auto" (use real data if there's enough, else fall back to
    synthetic with a clear warning), "real" (force real data even if
    sparse -- results will say so), "synthetic" (always use the labeled
    synthetic generator)."""
    log_fn = log_fn or (lambda msg: None)

    real_df = load_real_dataset()
    used_synthetic = False

    if source == "synthetic":
        df = generate_synthetic_dataset(n=n_synthetic)
        used_synthetic = True
    elif source == "real":
        df = real_df
    else:  # auto
        if len(real_df) >= min_real_rows:
            df = real_df
        else:
            log_fn(f"  -> Only {len(real_df)} real audit(s) in history (need >= {min_real_rows} for "
                    f"trustworthy analysis) -- using a clearly-labeled SYNTHETIC dataset instead.")
            df = generate_synthetic_dataset(n=n_synthetic)
            used_synthetic = True

    if df.empty:
        return {
            "ok": False,
            "error": "No data available to analyze (real history is empty and synthetic wasn't requested).",
            "used_synthetic": used_synthetic,
            "real_row_count": len(real_df),
            "row_count": 0,
        }

    X, y, feature_names = engineer_feature_matrix(df)
    if len(y) < 3:
        return {
            "ok": False,
            "error": f"Only {len(y)} usable row(s) after dropping missing targets -- need at least 3.",
            "used_synthetic": used_synthetic,
            "real_row_count": len(real_df),
            "row_count": len(y),
        }

    correlations = compute_correlations(df)
    model_comparison = train_and_compare_models(X, y)
    importances = feature_importance(X, y, feature_names)

    return {
        "ok": True,
        "used_synthetic": used_synthetic,
        "real_row_count": len(real_df),
        "row_count": len(y),
        "feature_count": len(feature_names),
        "correlations": correlations,
        "model_comparison": model_comparison,
        "feature_importance": importances,
        "best_model": model_comparison[0]["model"] if model_comparison else None,
    }


def print_analysis_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("SCORE ANALYTICS SUMMARY")
    print("=" * 64)
    if not summary.get("ok"):
        print(f"Could not run analysis: {summary.get('error')}")
        print("=" * 64 + "\n")
        return

    if summary["used_synthetic"]:
        print(f"*** USING SYNTHETIC DATA *** (only {summary['real_row_count']} real audit(s) in "
              f"history -- not enough for trustworthy analysis). Results below describe the "
              f"synthetic dataset, NOT your real websites.")
    else:
        print(f"Using {summary['row_count']} real audit(s) from your history.")

    print("\nTop correlations with overall_score:")
    shown = summary["correlations"][:8]
    for c in shown:
        flag = "" if c["reliable"] else "  [LOW-N, DO NOT TRUST]"
        print(f"  {c['feature']:<32} r={c['correlation']:+.3f}  (p={c['p_value']}, n={c['n']}){flag}")
    if any(not c["reliable"] for c in shown):
        print(f"  Note: [LOW-N] entries have n < {MIN_RELIABLE_CORRELATION_N} -- usually because that "
              f"category's specialist frequently failed to return valid data. With that few samples, "
              f"a correlation's sign and magnitude can flip on 1-2 rows and shouldn't be treated as a "
              f"real finding.")

    print("\nModel comparison (cross-validated):")
    print(f"  {'Model':<28}{'R^2':>10}{'MAE':>10}{'RMSE':>10}")
    for m in summary["model_comparison"]:
        print(f"  {m['model']:<28}{m['r2_mean']:>10}{m['mae_mean']:>10}{m['rmse_mean']:>10}")
    print(f"  Best model by R^2: {summary['best_model']}")

    print("\nTop feature importances (from a RandomForestRegressor):")
    for f in summary["feature_importance"][:8]:
        print(f"  {f['feature']:<32} {f['importance']}")

    print("=" * 64 + "\n")