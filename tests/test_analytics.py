from __future__ import annotations

import numpy as np
import pandas as pd

from agent import analytics


def _sample_report(overall_score=75.0, review_status="approved", categories=None):
    if categories is None:
        categories = [
            {"name": "Technical SEO", "score": 80, "weight": 0.5, "findings": [
                {"severity": "warning", "issue": "x", "recommendation": "y"},
            ]},
            {"name": "Web Security", "score": 70, "weight": 0.5, "findings": [
                {"severity": "critical", "issue": "a", "recommendation": "b"},
                {"severity": "good", "issue": "c", "recommendation": "d"},
            ]},
        ]
    return {"overall_score": overall_score, "review_status": review_status, "categories": categories}


class TestExtractFeaturesFromReport:
    def test_extracts_present_category_score_and_weight(self):
        row = analytics._extract_features_from_report(_sample_report())
        assert row["score__Technical SEO"] == 80
        assert row["weight__Technical SEO"] == 0.5

    def test_missing_category_is_nan_not_zero(self):
        row = analytics._extract_features_from_report(_sample_report())
        assert np.isnan(row["score__Link Health"])
        assert np.isnan(row["weight__Link Health"])

    def test_counts_findings_by_severity_across_categories(self):
        row = analytics._extract_features_from_report(_sample_report())
        assert row["total_findings"] == 3
        assert row["critical_findings"] == 1
        assert row["warning_findings"] == 1
        assert row["good_findings"] == 1

    def test_counts_present_categories(self):
        row = analytics._extract_features_from_report(_sample_report())
        assert row["num_categories_present"] == 2

    def test_review_approved_flag(self):
        approved = analytics._extract_features_from_report(_sample_report(review_status="approved"))
        rejected = analytics._extract_features_from_report(_sample_report(review_status="not_approved"))
        assert approved["review_approved"] == 1
        assert rejected["review_approved"] == 0

    def test_overall_score_passthrough(self):
        row = analytics._extract_features_from_report(_sample_report(overall_score=42.5))
        assert row["overall_score"] == 42.5

    def test_handles_report_with_no_categories(self):
        row = analytics._extract_features_from_report({"overall_score": 50, "categories": []})
        assert row["total_findings"] == 0
        assert row["num_categories_present"] == 0


class TestLoadRealDataset:
    def test_builds_dataframe_from_memory(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [
            _sample_report(overall_score=80), _sample_report(overall_score=60),
        ])
        df = analytics.load_real_dataset()
        assert len(df) == 2
        assert list(df["overall_score"]) == [80, 60]

    def test_empty_history_returns_empty_dataframe_not_error(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [])
        df = analytics.load_real_dataset()
        assert df.empty


class TestGenerateSyntheticDataset:
    def test_generates_requested_row_count(self):
        df = analytics.generate_synthetic_dataset(n=50, seed=1)
        assert len(df) == 50

    def test_overall_score_within_valid_range(self):
        df = analytics.generate_synthetic_dataset(n=100, seed=1)
        assert df["overall_score"].min() >= 0
        assert df["overall_score"].max() <= 100

    def test_same_seed_is_reproducible(self):
        df1 = analytics.generate_synthetic_dataset(n=30, seed=7)
        df2 = analytics.generate_synthetic_dataset(n=30, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = analytics.generate_synthetic_dataset(n=30, seed=1)
        df2 = analytics.generate_synthetic_dataset(n=30, seed=2)
        assert not df1["overall_score"].equals(df2["overall_score"])

    def test_present_category_weights_sum_to_one(self):
        """Regression guard: an earlier version didn't renormalize weights
        for missing categories, creating a fake correlation between
        num_categories_present and overall_score that doesn't reflect how
        the real app (which always renormalizes) actually behaves."""
        df = analytics.generate_synthetic_dataset(n=50, seed=3)
        weight_cols = [c for c in df.columns if c.startswith("weight__")]
        for _, row in df.iterrows():
            total = sum(row[c] for c in weight_cols if not pd.isna(row[c]))
            assert abs(total - 1.0) < 0.01

    def test_higher_category_scores_correlate_with_higher_overall_score(self):
        """Sanity check the generator actually encodes a real signal (not
        pure noise) -- Technical SEO has the highest true weight, so it
        should correlate meaningfully with overall_score."""
        df = analytics.generate_synthetic_dataset(n=300, seed=5)
        corr = df["score__Technical SEO"].corr(df["overall_score"])
        assert corr > 0.3


class TestEngineerFeatureMatrix:
    def test_drops_rows_with_missing_target(self):
        df = pd.DataFrame({"score__Technical SEO": [80, 70, 60], "overall_score": [75, np.nan, 65]})
        X, y, names = analytics.engineer_feature_matrix(df)
        assert len(y) == 2

    def test_imputes_missing_feature_values(self):
        df = pd.DataFrame({"score__Technical SEO": [80, np.nan, 60], "overall_score": [75, 70, 65]})
        X, y, names = analytics.engineer_feature_matrix(df)
        assert not np.isnan(X).any()

    def test_feature_names_exclude_target(self):
        df = pd.DataFrame({"score__Technical SEO": [80], "overall_score": [75]})
        X, y, names = analytics.engineer_feature_matrix(df)
        assert "overall_score" not in names


class TestComputeCorrelations:
    def test_returns_sorted_by_absolute_strength(self):
        rng = np.random.default_rng(0)
        n = 50
        strong = np.linspace(0, 100, n)
        weak = rng.normal(50, 30, n)
        target = strong + rng.normal(0, 1, n)
        df = pd.DataFrame({"strong_feature": strong, "weak_feature": weak, "overall_score": target})
        results = analytics.compute_correlations(df)
        assert results[0]["feature"] == "strong_feature"

    def test_skips_constant_column(self):
        df = pd.DataFrame({"constant": [5, 5, 5, 5], "overall_score": [10, 20, 30, 40]})
        results = analytics.compute_correlations(df)
        assert all(r["feature"] != "constant" for r in results)

    def test_empty_target_returns_empty_list(self):
        df = pd.DataFrame({"x": [1, 2, 3], "overall_score": [np.nan, np.nan, np.nan]})
        assert analytics.compute_correlations(df) == []

    def test_never_raises_on_all_nan_feature(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan], "overall_score": [10, 20, 30]})
        results = analytics.compute_correlations(df)  # should not raise
        assert all(r["feature"] != "x" for r in results)

    def test_low_n_correlation_is_flagged_unreliable(self):
        df = pd.DataFrame({
            "sparse_feature": [80, 85, 90] + [np.nan] * 50,
            "overall_score": list(range(53)),
        })
        results = analytics.compute_correlations(df)
        sparse = next(r for r in results if r["feature"] == "sparse_feature")
        assert sparse["reliable"] is False

    def test_high_n_correlation_is_flagged_reliable(self):
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({"dense_feature": rng.normal(50, 10, n), "overall_score": rng.normal(70, 10, n)})
        results = analytics.compute_correlations(df, min_reliable_n=15)
        dense = next(r for r in results if r["feature"] == "dense_feature")
        assert dense["reliable"] is True

    def test_reliable_result_ranks_above_unreliable_even_with_smaller_raw_correlation(self):
        """Regression test for a real observed issue: a category present in
        only 8 of 61 real audits showed r=-0.857 (a nonsensical negative
        correlation for a positively-weighted score component -- a
        statistical artifact of the tiny sample), while a category present
        in 51 of 61 showed a sensible r=+0.828. Sorted by raw |r| alone,
        the noisy n=8 result ranked #1, ahead of the well-supported n=51
        result. A well-supported, smaller-magnitude correlation must rank
        ahead of a large-magnitude, poorly-supported one."""
        rng = np.random.default_rng(0)
        n = 61
        overall = rng.normal(70, 10, n)

        reliable_feature = np.full(n, np.nan)
        reliable_idx = rng.choice(n, 51, replace=False)
        reliable_feature[reliable_idx] = overall[reliable_idx] * 0.9 + rng.normal(0, 3, 51)

        unreliable_feature = np.full(n, np.nan)
        unreliable_idx = rng.choice(n, 8, replace=False)
        unreliable_feature[unreliable_idx] = 100 - overall[unreliable_idx] + rng.normal(0, 2, 8)

        df = pd.DataFrame({
            "reliable_feature": reliable_feature,
            "unreliable_feature": unreliable_feature,
            "overall_score": overall,
        })
        results = analytics.compute_correlations(df)

        # The unreliable one must have the larger raw magnitude (that's the
        # whole point of this test setup) ...
        unreliable = next(r for r in results if r["feature"] == "unreliable_feature")
        reliable = next(r for r in results if r["feature"] == "reliable_feature")
        assert abs(unreliable["correlation"]) > abs(reliable["correlation"])

        # ... but it must NOT rank first.
        assert results[0]["feature"] == "reliable_feature"
        assert results[0]["reliable"] is True

    def test_all_correlations_still_included_nothing_hidden(self):
        """Unreliable results are de-prioritized in ranking, never removed
        -- full transparency, matching the rest of this project's ethos."""
        df = pd.DataFrame({
            "sparse_feature": [80, 85, 90] + [np.nan] * 50,
            "overall_score": list(range(53)),
        })
        results = analytics.compute_correlations(df)
        assert any(r["feature"] == "sparse_feature" for r in results)


class TestTrainAndCompareModels:
    def test_returns_all_four_models(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = X[:, 0] * 10 + rng.normal(0, 1, 40)
        results = analytics.train_and_compare_models(X, y, n_splits=4)
        names = {r["model"] for r in results}
        assert names == {"LinearRegression", "Ridge", "RandomForestRegressor", "GradientBoostingRegressor"}

    def test_sorted_by_r2_descending(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = X[:, 0] * 10 + rng.normal(0, 1, 40)
        results = analytics.train_and_compare_models(X, y, n_splits=4)
        r2_values = [r["r2_mean"] for r in results]
        assert r2_values == sorted(r2_values, reverse=True)

    def test_handles_tiny_dataset_without_crashing(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        results = analytics.train_and_compare_models(X, y, n_splits=5)  # more folds than sensible
        assert len(results) == 4

    def test_strong_linear_signal_gets_high_r2(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 2))
        y = X[:, 0] * 20 + X[:, 1] * 5 + rng.normal(0, 0.5, 100)
        results = analytics.train_and_compare_models(X, y, n_splits=5)
        best = results[0]
        assert best["r2_mean"] > 0.9


class TestFeatureImportance:
    def test_returns_sorted_descending(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = X[:, 0] * 20 + rng.normal(0, 1, 60)
        results = analytics.feature_importance(X, y, ["a", "b", "c"])
        importances = [r["importance"] for r in results]
        assert importances == sorted(importances, reverse=True)

    def test_most_predictive_feature_ranks_first(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 2))
        y = X[:, 0] * 50 + rng.normal(0, 0.1, 100)  # column 0 dominates
        results = analytics.feature_importance(X, y, ["dominant", "noise"])
        assert results[0]["feature"] == "dominant"


class TestRunAnalysis:
    def test_synthetic_source_always_uses_synthetic(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [_sample_report() for _ in range(100)])
        summary = analytics.run_analysis(source="synthetic", n_synthetic=50)
        assert summary["used_synthetic"] is True
        assert summary["row_count"] == 50

    def test_auto_uses_real_when_enough_rows(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits",
                             lambda: [_sample_report(overall_score=70 + i) for i in range(25)])
        summary = analytics.run_analysis(source="auto", min_real_rows=20)
        assert summary["used_synthetic"] is False
        assert summary["row_count"] == 25

    def test_auto_falls_back_to_synthetic_when_too_few_real_rows(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits",
                             lambda: [_sample_report() for _ in range(3)])
        summary = analytics.run_analysis(source="auto", min_real_rows=20, n_synthetic=40)
        assert summary["used_synthetic"] is True
        assert summary["row_count"] == 40

    def test_real_source_forces_real_even_if_sparse(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits",
                             lambda: [_sample_report(overall_score=70 + i) for i in range(4)])
        summary = analytics.run_analysis(source="real")
        assert summary["used_synthetic"] is False
        assert summary["row_count"] == 4

    def test_empty_real_data_with_real_source_returns_not_ok(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [])
        summary = analytics.run_analysis(source="real")
        assert summary["ok"] is False
        assert "No data" in summary["error"]

    def test_too_few_rows_after_dropping_missing_targets_returns_not_ok(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits",
                             lambda: [_sample_report(overall_score=None), _sample_report(overall_score=None)])
        summary = analytics.run_analysis(source="real")
        assert summary["ok"] is False

    def test_successful_run_includes_all_expected_sections(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [])
        summary = analytics.run_analysis(source="synthetic", n_synthetic=60)
        assert summary["ok"] is True
        assert "correlations" in summary
        assert "model_comparison" in summary
        assert "feature_importance" in summary
        assert summary["best_model"] in {"LinearRegression", "Ridge", "RandomForestRegressor", "GradientBoostingRegressor"}

    def test_logs_fallback_reason_when_using_synthetic(self, monkeypatch):
        monkeypatch.setattr(analytics.memory, "get_all_full_audits", lambda: [_sample_report()])
        logs = []
        analytics.run_analysis(source="auto", min_real_rows=20, n_synthetic=30, log_fn=logs.append)
        assert any("synthetic" in msg.lower() for msg in logs)


class TestPrintAnalysisSummary:
    def test_does_not_raise_on_success(self, capsys):
        summary = analytics.run_analysis(source="synthetic", n_synthetic=40)
        analytics.print_analysis_summary(summary)  # should not raise
        captured = capsys.readouterr()
        assert "SCORE ANALYTICS SUMMARY" in captured.out

    def test_does_not_raise_on_failure(self, capsys):
        analytics.print_analysis_summary({"ok": False, "error": "no data"})
        captured = capsys.readouterr()
        assert "no data" in captured.out

    def test_synthetic_warning_is_visible_in_output(self, capsys):
        summary = analytics.run_analysis(source="synthetic", n_synthetic=40)
        analytics.print_analysis_summary(summary)
        captured = capsys.readouterr()
        assert "SYNTHETIC" in captured.out

    def test_low_n_correlation_warning_is_visible_in_output(self, capsys, monkeypatch):
        def fake_run_analysis_result():
            return {
                "ok": True, "used_synthetic": False, "real_row_count": 61, "row_count": 61,
                "feature_count": 2,
                "correlations": [
                    {"feature": "score__Technical SEO", "correlation": 0.828, "p_value": 0.0, "n": 51, "reliable": True},
                    {"feature": "score__Best Practices", "correlation": -0.857, "p_value": 0.0066, "n": 8, "reliable": False},
                ],
                "model_comparison": [
                    {"model": "Ridge", "r2_mean": 0.8, "r2_std": 0.1, "mae_mean": 3.0, "rmse_mean": 4.0},
                ],
                "feature_importance": [{"feature": "score__Technical SEO", "importance": 0.5}],
                "best_model": "Ridge",
            }
        analytics.print_analysis_summary(fake_run_analysis_result())
        captured = capsys.readouterr()
        assert "LOW-N" in captured.out
        assert "do not trust" in captured.out.lower() or "shouldn't be treated as a" in captured.out.lower()