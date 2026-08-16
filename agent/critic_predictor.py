"""
Critic-Approval Predictor: a real, small, locally-trained classifier that
predicts whether the critic agent will approve a draft report -- trained on
real data this project's own pipeline already generates (every stored audit
has a `review_status`).

HONEST SCOPING, on two fronts:

1. "Fine-tuning llama-3.3-70b" isn't a real option here -- Groq is an
   inference API, not a fine-tuning platform. This is a genuinely small,
   genuinely local model instead: a classical classifier
   (LogisticRegression / RandomForestClassifier / GradientBoostingClassifier)
   trained on features engineered from stored audit reports, predicting the
   real review_approved outcome.

2. More precisely: this is TRAINED FROM SCRATCH, not "fine-tuned" in the
   strict sense of adjusting a pretrained model's weights. Calling it
   "fine-tuning" would overclaim. What it demonstrates is the same
   practical skill a resume line about fine-tuning is usually gesturing
   at -- adapting/training a model against a specific downstream task and
   dataset -- just via classical ML rather than adjusting pretrained
   weights. Described accurately here rather than dressed up.

Reuses agent/analytics.py's feature engineering and real/synthetic
fallback pattern. The only real difference: the prediction TARGET here is
review_approved (binary), not overall_score (continuous) -- so
review_approved is excluded from the feature set (using it as both input
and output would be target leakage, not a real classifier).

Standalone, like analytics.py and similarity_search.py -- NOT currently
wired into the live audit pipeline. If it were, the practical use is
predicting a low approval probability from a draft's specialist-report
features before spending an actual critic API call, to flag it for extra
synthesizer care or a --mode deep upgrade.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import analytics

MIN_REAL_ROWS_FOR_ANALYSIS = analytics.MIN_REAL_ROWS_FOR_ANALYSIS
MIN_ROWS_PER_CLASS = 5  # below this for either outcome, classification metrics aren't trustworthy


def engineer_classification_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Like analytics.engineer_feature_matrix, but the target is
    review_approved (binary) instead of overall_score, and review_approved
    is excluded from the feature set to avoid target leakage."""
    df = df.dropna(subset=["review_approved"]).copy()
    feature_cols = [c for c in df.columns if c not in ("overall_score", "review_approved")]
    X_df = df[feature_cols].copy()
    for col in feature_cols:
        if X_df[col].isna().any():
            median = X_df[col].median()
            X_df[col] = X_df[col].fillna(median if not np.isnan(median) else 0.0)
    X = X_df.to_numpy(dtype=float)
    y = df["review_approved"].to_numpy(dtype=int)
    return X, y, feature_cols


def class_balance(y: np.ndarray) -> dict:
    total = len(y)
    approved = int(np.sum(y == 1))
    rejected = int(np.sum(y == 0))
    return {
        "total": total, "approved": approved, "rejected": rejected,
        "approved_fraction": round(approved / total, 3) if total else None,
        "majority_class_baseline_accuracy": round(max(approved, rejected) / total, 3) if total else None,
    }


def train_and_compare_classifiers(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[dict]:
    """Train several classical classifiers to predict review_approved and
    compare via STRATIFIED k-fold cross-validation (stratified so each fold
    keeps roughly the real class balance -- with imbalanced data, a plain
    split can accidentally put all of one class in one fold). Always
    includes a majority-class DummyClassifier baseline -- if nothing beats
    blindly guessing the majority class, that's the most important thing
    this comparison can tell you, not a footnote."""
    smaller_class_size = int(np.bincount(y).min())
    n_splits = max(2, min(n_splits, smaller_class_size))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = {
        "MajorityClassBaseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
    }

    results = []
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
            f1 = cross_val_score(model, X, y, cv=skf, scoring="f1", error_score=np.nan)
            try:
                auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
                auc_mean = round(float(np.nanmean(auc)), 3)
            except ValueError:
                auc_mean = None  # can happen with too few samples/classes in a fold
        results.append({
            "model": name,
            "accuracy_mean": round(float(np.mean(acc)), 3),
            "f1_mean": round(float(np.nanmean(f1)), 3) if not np.all(np.isnan(f1)) else None,
            "roc_auc_mean": auc_mean,
        })

    results.sort(key=lambda item: item["accuracy_mean"], reverse=True)
    return results


def feature_importance(X: np.ndarray, y: np.ndarray, feature_names: list[str], seed: int = 42) -> list[dict]:
    model = RandomForestClassifier(n_estimators=200, random_state=seed)
    model.fit(X, y)
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda pair: pair[1], reverse=True)
    return [{"feature": name, "importance": round(float(imp), 4)} for name, imp in ranked]


def train_final_classifier(X: np.ndarray, y: np.ndarray):
    """Trains the final classifier (LogisticRegression -- interpretable and
    stable on a small dataset) on ALL available data. This is the actual
    trained-from-scratch model, ready for predict_approval_probability."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def predict_approval_probability(model, feature_row: np.ndarray) -> float:
    """feature_row: a single 1D feature vector, in the same column order
    engineer_classification_matrix produced. Returns P(approved). Looks up
    the "approved" (class 1) column explicitly via model.classes_ rather
    than assuming column order."""
    proba = model.predict_proba(feature_row.reshape(1, -1))[0]
    classes = list(model.classes_)
    return float(proba[classes.index(1)])


def run_analysis(
    source: str = "auto",
    n_synthetic: int = 200,
    min_real_rows: int = MIN_REAL_ROWS_FOR_ANALYSIS,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    log_fn = log_fn or (lambda msg: None)

    real_df = analytics.load_real_dataset()
    used_synthetic = False

    if source == "synthetic":
        df = analytics.generate_synthetic_dataset(n=n_synthetic)
        used_synthetic = True
    elif source == "real":
        df = real_df
    else:  # auto
        if len(real_df) >= min_real_rows:
            df = real_df
        else:
            log_fn(f"  -> Only {len(real_df)} real audit(s) in history (need >= {min_real_rows} for "
                    f"trustworthy analysis) -- using a clearly-labeled SYNTHETIC dataset instead.")
            df = analytics.generate_synthetic_dataset(n=n_synthetic)
            used_synthetic = True

    if df.empty:
        return {
            "ok": False, "error": "No data available (real history is empty and synthetic wasn't requested).",
            "used_synthetic": used_synthetic, "real_row_count": len(real_df), "row_count": 0,
        }

    X, y, feature_names = engineer_classification_matrix(df)
    balance = class_balance(y)

    if balance["total"] < 6 or min(balance["approved"], balance["rejected"]) < MIN_ROWS_PER_CLASS:
        return {
            "ok": False,
            "error": (f"Not enough examples of both outcomes to train reliably (approved="
                      f"{balance['approved']}, rejected={balance['rejected']}, need >= "
                      f"{MIN_ROWS_PER_CLASS} of each)."),
            "used_synthetic": used_synthetic,
            "real_row_count": len(real_df),
            "row_count": balance["total"],
            "class_balance": balance,
        }

    comparison = train_and_compare_classifiers(X, y)
    importances = feature_importance(X, y, feature_names)
    train_final_classifier(X, y)  # trained here to prove it runs end-to-end; not returned (not JSON-safe)

    baseline = next((m for m in comparison if m["model"] == "MajorityClassBaseline"), None)
    best_non_baseline = max(
        (m for m in comparison if m["model"] != "MajorityClassBaseline"),
        key=lambda m: m["accuracy_mean"], default=None,
    )
    beats_baseline = bool(
        best_non_baseline and baseline and best_non_baseline["accuracy_mean"] > baseline["accuracy_mean"]
    )

    return {
        "ok": True,
        "used_synthetic": used_synthetic,
        "real_row_count": len(real_df),
        "row_count": balance["total"],
        "class_balance": balance,
        "model_comparison": comparison,
        "feature_importance": importances,
        "feature_names": feature_names,
        "best_model": comparison[0]["model"] if comparison else None,
        "beats_majority_baseline": beats_baseline,
    }


def print_analysis_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("CRITIC-APPROVAL PREDICTOR SUMMARY")
    print("=" * 64)
    if not summary.get("ok"):
        print(f"Could not train: {summary.get('error')}")
        if summary.get("class_balance"):
            b = summary["class_balance"]
            print(f"(Class balance so far: {b['approved']} approved / {b['rejected']} rejected)")
        print("=" * 64 + "\n")
        return

    if summary["used_synthetic"]:
        print(f"*** USING SYNTHETIC DATA *** (only {summary['real_row_count']} real audit(s) in "
              f"history). Results below describe the synthetic dataset, NOT real approval patterns.")
    else:
        print(f"Using {summary['row_count']} real audit(s) from your history.")

    bal = summary["class_balance"]
    print(f"\nClass balance: {bal['approved']} approved / {bal['rejected']} rejected "
          f"({bal['approved_fraction'] * 100:.1f}% approved). Majority-class baseline accuracy: "
          f"{bal['majority_class_baseline_accuracy']}")

    print("\nModel comparison (stratified cross-validated):")
    print(f"  {'Model':<28}{'Accuracy':>10}{'F1':>8}{'ROC-AUC':>10}")
    for m in summary["model_comparison"]:
        f1_str = f"{m['f1_mean']}" if m["f1_mean"] is not None else "n/a"
        auc_str = f"{m['roc_auc_mean']}" if m["roc_auc_mean"] is not None else "n/a"
        print(f"  {m['model']:<28}{m['accuracy_mean']:>10}{f1_str:>8}{auc_str:>10}")
    print(f"  Best model: {summary['best_model']}")
    if not summary["beats_majority_baseline"]:
        print("  *** WARNING: no model beat the majority-class baseline -- the features here don't "
              "carry a real predictive signal for approval yet (or the dataset is too small/imbalanced).")

    print("\nTop feature importances (from a RandomForestClassifier):")
    for f in summary["feature_importance"][:8]:
        print(f"  {f['feature']:<32} {f['importance']}")

    print("=" * 64 + "\n")
