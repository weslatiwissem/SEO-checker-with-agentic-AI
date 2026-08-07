from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent import orchestrator


# --------------------------------------------------------------------------
# _reconcile_overall_score
# --------------------------------------------------------------------------

class TestReconcileOverallScore:
    def _log(self):
        return lambda msg: None

    def test_recomputes_wrong_overall_score(self):
        report = {
            "overall_score": 50.0,  # wrong -- model's bad arithmetic
            "grade": "F",
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": 80.0, "weight": 0.5},
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 90.0
        assert report["grade"] == "A"

    def test_leaves_correct_score_untouched(self):
        report = {
            "overall_score": 90.0,
            "grade": "A",
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": 80.0, "weight": 0.5},
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 90.0

    def test_normalizes_weights_that_dont_sum_to_one(self):
        report = {
            "overall_score": 0,
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.6},
                {"name": "B", "score": 50.0, "weight": 0.6},  # sums to 1.2
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        total_weight = sum(c["weight"] for c in report["categories"])
        assert abs(total_weight - 1.0) < 0.01

    def test_excludes_categories_with_missing_score_from_average(self):
        report = {
            "overall_score": 0,
            "categories": [
                {"name": "A", "score": 100.0, "weight": 0.5},
                {"name": "B", "score": None, "weight": 0.5},  # failed specialist
            ],
        }
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 100.0  # only A counted

    def test_noop_when_no_categories(self):
        report = {"overall_score": 42, "categories": []}
        orchestrator._reconcile_overall_score(report, self._log())
        assert report["overall_score"] == 42

    def test_grade_boundaries(self):
        for score, expected_grade in [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (30, "F")]:
            report = {"overall_score": 0, "categories": [{"name": "A", "score": score, "weight": 1.0}]}
            orchestrator._reconcile_overall_score(report, self._log())
            assert report["grade"] == expected_grade, f"score {score} should be grade {expected_grade}"


# --------------------------------------------------------------------------
# _recover_or_drop_empty_categories
# --------------------------------------------------------------------------

class TestRecoverOrDropEmptyCategories:
    def _log(self):
        return lambda msg: None

    def test_recovers_findings_from_specialist_report(self):
        report = {"categories": [{"name": "Link Health", "score": 80, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": 80, "findings": [
                {"severity": "warning", "issue": "2 broken links found.", "recommendation": "Fix them."},
            ]},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert len(report["categories"]) == 1
        assert report["categories"][0]["findings"]

    def test_drops_category_with_no_recoverable_findings(self):
        report = {"categories": [{"name": "Link Health", "score": None, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": None, "findings": [],
                      "raw_evidence_notes": "Specialist failed to complete: timeout"},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert report["categories"] == []

    def test_keeps_categories_that_already_have_findings(self):
        report = {"categories": [{"name": "Technical SEO", "score": 90, "findings": [
            {"severity": "good", "issue": "fine", "recommendation": ""},
        ]}]}
        orchestrator._recover_or_drop_empty_categories(report, {}, self._log())
        assert len(report["categories"]) == 1

    def test_recovers_score_too_if_draft_score_missing(self):
        report = {"categories": [{"name": "Link Health", "score": None, "findings": []}]}
        specialist_reports = {
            "links": {"category": "Link Health", "score": 65, "findings": [
                {"severity": "warning", "issue": "issue", "recommendation": "fix"},
            ]},
        }
        orchestrator._recover_or_drop_empty_categories(report, specialist_reports, self._log())
        assert report["categories"][0]["score"] == 65


# --------------------------------------------------------------------------
# Full pipeline, everything mocked -- verifies stage wiring, not real agent behavior
# --------------------------------------------------------------------------

class TestRunFullAuditWiring:
    def test_full_pipeline_wires_stages_together(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)

        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {
            "specialists": ["technical_seo"], "reasoning": "test",
        })

        fake_specialist_result = {
            "category": "Technical SEO", "score": 88, "findings": [
                {"severity": "good", "issue": "All good.", "recommendation": ""},
            ],
            "raw_evidence_notes": "checked stuff",
        }
        fake_agent = MagicMock()
        fake_agent.run.return_value = dict(fake_specialist_result)
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com",
            "overall_score": 88.0,
            "grade": "B",
            "summary": "Solid technical foundation.",
            "categories": [{"name": "Technical SEO", "score": 88.0, "weight": 1.0, "findings": [
                {"severity": "good", "issue": "All good.", "recommendation": ""},
            ]}],
            "quick_wins": [],
            "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")

        assert result["review_status"] == "approved"
        assert result["overall_score"] == 88.0
        assert "_specialist_reports" in result
        assert "_reflection_log" in result
        # No previous audit -> schema validation still includes a "trend" key
        # (its default), but it must be None, not a fabricated trend block.
        assert result["trend"] is None

    def test_not_approved_report_surfaces_unresolved_issues(self, monkeypatch):
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: None)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": ["technical_seo"], "reasoning": "t"})

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 40, "findings": [
            {"severity": "critical", "issue": "Broken.", "recommendation": "Fix."},
        ]}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com", "overall_score": 40.0, "grade": "F",
            "summary": "Needs work.",
            "categories": [{"name": "Technical SEO", "score": 40.0, "weight": 1.0, "findings": [
                {"severity": "critical", "issue": "Broken.", "recommendation": "Fix."},
            ]}],
            "quick_wins": [], "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": False, "issues": ["score too low for findings"], "instructions_for_revision": "..."}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert result["review_status"] == "not_approved"
        assert result["unresolved_review_issues"] == ["score too low for findings"]

    def test_previous_audit_produces_trend_block(self, monkeypatch):
        previous = {"overall_score": 70.0, "_timestamp": "2026-01-01T00:00:00+00:00"}
        monkeypatch.setattr(orchestrator.memory, "get_last_audit", lambda url: previous)
        monkeypatch.setattr(orchestrator.memory, "save_audit", lambda url, report: 1)
        monkeypatch.setattr(orchestrator, "run_planner", lambda *a, **kw: {"specialists": ["technical_seo"], "reasoning": "t"})

        fake_agent = MagicMock()
        fake_agent.run.return_value = {"category": "Technical SEO", "score": 85, "findings": []}
        fake_agent.tool_call_log = []
        monkeypatch.setattr(orchestrator, "build_specialist", lambda *a, **kw: fake_agent)

        fake_draft = {
            "url": "https://example.com", "overall_score": 85.0, "grade": "B",
            "summary": "Improved since last time.",
            "categories": [{"name": "Technical SEO", "score": 85.0, "weight": 1.0, "findings": []}],
            "quick_wins": [], "data_limitations": "",
        }
        monkeypatch.setattr(orchestrator, "reflect_and_revise", lambda *a, **kw: (
            dict(fake_draft), [{"round": 1, "review": {"approved": True, "issues": [], "instructions_for_revision": ""}}]
        ))

        result = orchestrator.run_full_audit("https://example.com", use_memory=True, mode="quick")
        assert result["trend"]["previous_score"] == 70.0
        assert result["trend"]["score_delta"] == 15.0
