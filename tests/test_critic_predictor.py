from __future__ import annotations

import numpy as np
import pandas as pd

from agent import critic_predictor as cp


def _sample_report(overall_score=75.0, review_status="approved"):
    return {
        "overall_score": overall_score, "review_status": review_status,
        "categories": [
            {"name": "Technical SEO", "score": 80, "weight": 1.0, "findings": [
                {"severity": "warning", "issue": "x", "recommendation": "y"},
            ]},
        ],
    }


class TestEngineerClassificationMatrix:
    def test_target_is_review_approved_not_overall_score(self):
        df = pd.DataFrame({
            "score__Technical SEO": [80, 70, 60],
            "overall_score": [75, 65, 55],
            "review_approved": [1, 0, 1],
        })
        X, y, names = cp.engineer_classification_matrix(df)
        assert list(y) == [1, 0, 1]

    def test_review_approved_excluded_from_features_no_leakage(self):
        df = pd.DataFrame({
            "score__Technical SEO": [80, 70, 60],
            "overall_score": [75, 65, 55],
            "review_approved": [1, 0, 1],
        })
        X, y, names = cp.engineer_classification_matrix(df)
        assert "review_approved" not in names

    def test_overall_score_also_excluded_from_features(self):
        """overall_score isn't the target here, but it's downstream of the
        same pipeline stage as review_approved and excluding it keeps the
        feature set focused on inputs available before synthesis."""
        df = pd.DataFrame({
            "score__Technical SEO": [80, 70, 60],
            "overall_score": [75, 65, 55],
            "review_approved": [1, 0, 1],
        })
        X, y, names = cp.engineer_classification_matrix(df)
        assert "overall_score" not in names

    def test_drops_rows_with_missing_target(self):
        df = pd.DataFrame({
            "score__Technical SEO": [80, 70, 60],
            "overall_score": [75, 65, 55],
            "review_approved": [1, np.nan, 0],
        })
        X, y, names = cp.engineer_classification_matrix(df)
        assert len(y) == 2

    def test_imputes_missing_feature_values(self):
        df = pd.DataFrame({
            "score__Technical SEO": [80, np.nan, 60],
            "overall_score": [75, 65, 55],
            "review_approved": [1, 0, 1],
        })
        X, y, names = cp.engineer_classification_matrix(df)
        assert not np.isnan(X).any()


class TestClassBalance:
    def test_counts_correctly(self):
        balance = cp.class_balance(np.array([1, 1, 0, 1, 0]))
        assert balance["approved"] == 3
        assert balance["rejected"] == 2
        assert balance["total"] == 5

    def test_approved_fraction(self):
        balance = cp.class_balance(np.array([1, 1, 0, 0]))
        assert balance["approved_fraction"] == 0.5

    def test_majority_baseline_accuracy(self):
        balance = cp.class_balance(np.array([1, 1, 1, 0]))
        assert balance["majority_class_baseline_accuracy"] == 0.75

    def test_empty_array(self):
        balance = cp.class_balance(np.array([]))
        assert balance["total"] == 0
        assert balance["approved_fraction"] is None


class TestTrainAndCompareClassifiers:
    def test_returns_all_four_including_baseline(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = (X[:, 0] > 0).astype(int)
        results = cp.train_and_compare_classifiers(X, y, n_splits=4)
        names = {r["model"] for r in results}
        assert names == {"MajorityClassBaseline", "LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"}

    def test_sorted_by_accuracy_descending(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = (X[:, 0] > 0).astype(int)
        results = cp.train_and_compare_classifiers(X, y, n_splits=4)
        accs = [r["accuracy_mean"] for r in results]
        assert accs == sorted(accs, reverse=True)

    def test_clean_learnable_signal_gets_near_perfect_accuracy(self):
        rng = np.random.default_rng(0)
        n = 100
        score = rng.uniform(40, 100, n)
        y = (score > 75).astype(int)
        X = score.reshape(-1, 1)
        results = cp.train_and_compare_classifiers(X, y, n_splits=5)
        best = max(r["accuracy_mean"] for r in results if r["model"] != "MajorityClassBaseline")
        assert best > 0.9

    def test_pure_noise_target_stays_near_baseline(self):
        """Sanity check the comparison doesn't fool itself: with a target
        that's genuinely random and uncorrelated with the features, no
        model should meaningfully beat the majority-class baseline."""
        rng = np.random.default_rng(0)
        n = 150
        X = rng.normal(size=(n, 4))
        y = rng.integers(0, 2, n)  # pure noise, independent of X
        results = cp.train_and_compare_classifiers(X, y, n_splits=5)
        baseline_acc = next(r["accuracy_mean"] for r in results if r["model"] == "MajorityClassBaseline")
        best_acc = max(r["accuracy_mean"] for r in results if r["model"] != "MajorityClassBaseline")
        assert best_acc < baseline_acc + 0.15  # generous margin for small-sample CV noise

    def test_handles_small_minority_class_without_crashing(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = np.array([1, 1, 1, 1, 1, 0])  # only 1 of the minority class
        results = cp.train_and_compare_classifiers(X, y, n_splits=5)
        assert len(results) == 4


class TestFeatureImportance:
    def test_returns_sorted_descending(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = (X[:, 0] > 0).astype(int)
        results = cp.feature_importance(X, y, ["a", "b", "c"])
        importances = [r["importance"] for r in results]
        assert importances == sorted(importances, reverse=True)


class TestTrainFinalClassifierAndPredict:
    def test_predict_approval_probability_returns_value_between_0_and_1(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        model = cp.train_final_classifier(X, y)
        prob = cp.predict_approval_probability(model, X[0])
        assert 0.0 <= prob <= 1.0

    def test_high_score_input_predicts_higher_approval_than_low_score(self):
        rng = np.random.default_rng(0)
        n = 200
        score = rng.uniform(0, 100, n)
        y = (score > 70).astype(int)
        X = score.reshape(-1, 1)
        model = cp.train_final_classifier(X, y)
        high_prob = cp.predict_approval_probability(model, np.array([95.0]))
        low_prob = cp.predict_approval_probability(model, np.array([10.0]))
        assert high_prob > low_prob


class TestRunAnalysis:
    def test_synthetic_source_always_uses_synthetic(self, monkeypatch):
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits",
                             lambda: [_sample_report() for _ in range(100)])
        summary = cp.run_analysis(source="synthetic", n_synthetic=100)
        assert summary["used_synthetic"] is True

    def test_insufficient_class_balance_returns_not_ok(self, monkeypatch):
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits", lambda: (
            [_sample_report(review_status="not_approved") for _ in range(20)]
            + [_sample_report(review_status="approved") for _ in range(2)]
        ))
        summary = cp.run_analysis(source="real")
        assert summary["ok"] is False
        assert "class_balance" in summary
        assert summary["class_balance"]["approved"] == 2

    def test_empty_real_data_with_real_source_returns_not_ok(self, monkeypatch):
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits", lambda: [])
        summary = cp.run_analysis(source="real")
        assert summary["ok"] is False

    def test_successful_run_includes_all_expected_sections(self, monkeypatch):
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits", lambda: [])
        summary = cp.run_analysis(source="synthetic", n_synthetic=150)
        assert summary["ok"] is True
        assert "class_balance" in summary
        assert "model_comparison" in summary
        assert "feature_importance" in summary
        assert "beats_majority_baseline" in summary

    def test_sufficient_balanced_real_data_uses_real(self, monkeypatch):
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits", lambda: (
            [_sample_report(overall_score=90 + i * 0.1, review_status="approved") for i in range(15)]
            + [_sample_report(overall_score=50 + i * 0.1, review_status="not_approved") for i in range(15)]
        ))
        summary = cp.run_analysis(source="auto", min_real_rows=20)
        assert summary["used_synthetic"] is False
        assert summary["row_count"] == 30

    def test_result_is_json_serializable(self, monkeypatch):
        """The trained model itself must never leak into the returned dict
        (sklearn models aren't JSON-serializable) -- this would break
        `--out results.json` in the CLI."""
        import json
        monkeypatch.setattr(cp.analytics.memory, "get_all_full_audits", lambda: [])
        summary = cp.run_analysis(source="synthetic", n_synthetic=100)
        json.dumps(summary)  # should not raise


class TestPrintAnalysisSummary:
    def test_does_not_raise_on_success(self, capsys):
        summary = cp.run_analysis(source="synthetic", n_synthetic=100)
        cp.print_analysis_summary(summary)
        captured = capsys.readouterr()
        assert "CRITIC-APPROVAL PREDICTOR SUMMARY" in captured.out

    def test_does_not_raise_on_failure_with_class_balance(self, capsys):
        cp.print_analysis_summary({
            "ok": False, "error": "not enough data",
            "class_balance": {"approved": 2, "rejected": 25},
        })
        captured = capsys.readouterr()
        assert "not enough data" in captured.out
        assert "2 approved" in captured.out

    def test_does_not_raise_on_failure_without_class_balance(self, capsys):
        cp.print_analysis_summary({"ok": False, "error": "no data"})
        captured = capsys.readouterr()
        assert "no data" in captured.out

    def test_synthetic_warning_visible(self, capsys):
        summary = cp.run_analysis(source="synthetic", n_synthetic=100)
        cp.print_analysis_summary(summary)
        captured = capsys.readouterr()
        assert "SYNTHETIC" in captured.out
