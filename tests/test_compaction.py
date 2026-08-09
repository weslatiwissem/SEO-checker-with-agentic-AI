from __future__ import annotations

from agent import compaction


def _finding(severity="warning", issue="An issue.", recommendation="Fix it."):
    return {"severity": severity, "issue": issue, "recommendation": recommendation}


class TestEstimateTokens:
    def test_scales_roughly_with_content_size(self):
        small = {"a": "x" * 40}
        big = {"a": "x" * 4000}
        assert compaction.estimate_tokens(big) > compaction.estimate_tokens(small)

    def test_empty_payload_is_near_zero(self):
        assert compaction.estimate_tokens({}) < 5


class TestCompactFindings:
    def test_under_the_cap_is_left_alone(self):
        findings = [_finding() for _ in range(3)]
        result, dropped = compaction._compact_findings(findings, max_findings=12)
        assert len(result) == 3
        assert dropped == 0

    def test_drops_good_findings_before_warning_or_critical(self):
        findings = (
            [_finding(severity="critical") for _ in range(2)]
            + [_finding(severity="warning") for _ in range(2)]
            + [_finding(severity="good") for _ in range(6)]
        )
        result, dropped = compaction._compact_findings(findings, max_findings=4)
        assert dropped == 6
        severities = [f["severity"] for f in result]
        assert severities.count("critical") == 2
        assert severities.count("warning") == 2
        assert severities.count("good") == 0

    def test_never_drops_a_critical_finding_if_it_can_be_avoided(self):
        findings = [_finding(severity="critical") for _ in range(3)] + [_finding(severity="good") for _ in range(20)]
        result, dropped = compaction._compact_findings(findings, max_findings=3)
        severities = [f["severity"] for f in result]
        assert severities.count("critical") == 3
        assert dropped == 20

    def test_truncates_overlong_issue_and_recommendation_text(self):
        long_text = "x" * 1000
        findings = [_finding(issue=long_text, recommendation=long_text)]
        result, _ = compaction._compact_findings(findings, max_findings=12)
        assert len(result[0]["issue"]) < 1000
        assert "truncated" in result[0]["issue"]
        assert len(result[0]["recommendation"]) < 1000

    def test_non_list_input_passed_through_unchanged(self):
        result, dropped = compaction._compact_findings(None, max_findings=12)
        assert result is None
        assert dropped == 0


class TestCompactReport:
    def test_compacts_findings_in_every_category(self):
        report = {
            "categories": [
                {"name": "A", "findings": [_finding(severity="good") for _ in range(30)]},
                {"name": "B", "findings": [_finding()]},
            ]
        }
        result = compaction.compact_report(report)
        assert len(result["categories"][0]["findings"]) <= compaction.COMPACTION_MAX_FINDINGS_PER_CATEGORY
        assert len(result["categories"][1]["findings"]) == 1

    def test_adds_compaction_note_only_when_something_was_dropped(self):
        report = {"categories": [{"name": "A", "findings": [_finding(severity="good") for _ in range(30)]}]}
        result = compaction.compact_report(report)
        assert "_compaction_note" in result["categories"][0]

    def test_no_note_when_nothing_dropped(self):
        report = {"categories": [{"name": "A", "findings": [_finding()]}]}
        result = compaction.compact_report(report)
        assert "_compaction_note" not in result["categories"][0]

    def test_does_not_mutate_the_original(self):
        original = {"categories": [{"name": "A", "findings": [_finding(severity="good") for _ in range(30)]}]}
        import copy
        original_copy = copy.deepcopy(original)
        compaction.compact_report(original)
        assert original == original_copy

    def test_non_report_shaped_input_passed_through(self):
        assert compaction.compact_report({"foo": "bar"}) == {"foo": "bar"}
        assert compaction.compact_report(None) is None


class TestCompactSpecialistReports:
    def test_truncates_raw_evidence_notes(self):
        reports = {"technical_seo": {"category": "Technical SEO", "findings": [], "raw_evidence_notes": "x" * 2000}}
        result = compaction.compact_specialist_reports(reports)
        assert len(result["technical_seo"]["raw_evidence_notes"]) < 2000

    def test_compacts_findings_per_specialist(self):
        reports = {"content": {"category": "Content", "findings": [_finding(severity="good") for _ in range(30)]}}
        result = compaction.compact_specialist_reports(reports)
        assert len(result["content"]["findings"]) <= compaction.COMPACTION_MAX_FINDINGS_PER_CATEGORY

    def test_critic_feedback_previous_draft_is_compacted(self):
        reports = {
            "technical_seo": {"category": "Technical SEO", "findings": []},
            "_critic_feedback": {
                "previous_draft": {"categories": [{"name": "A", "findings": [_finding(severity="good") for _ in range(30)]}]},
                "issues": ["issue 1"],
                "instructions": "x" * 2000,
            },
        }
        result = compaction.compact_specialist_reports(reports)
        assert len(result["_critic_feedback"]["previous_draft"]["categories"][0]["findings"]) <= compaction.COMPACTION_MAX_FINDINGS_PER_CATEGORY
        assert len(result["_critic_feedback"]["instructions"]) < 2000
        assert result["_critic_feedback"]["issues"] == ["issue 1"]  # untouched

    def test_non_dict_input_passed_through(self):
        assert compaction.compact_specialist_reports(None) is None

    def test_does_not_mutate_the_original(self):
        import copy
        original = {"technical_seo": {"category": "Technical SEO", "findings": [_finding(severity="good") for _ in range(30)]}}
        original_copy = copy.deepcopy(original)
        compaction.compact_specialist_reports(original)
        assert original == original_copy


class TestMaybeCompactSpecialistReports:
    def test_noop_below_threshold_returns_same_object(self):
        reports = {"technical_seo": {"category": "Technical SEO", "findings": [_finding()]}}
        result = compaction.maybe_compact_specialist_reports(reports, threshold=100000)
        assert result is reports  # same object, not even a copy

    def test_compacts_above_threshold(self):
        reports = {"content": {"category": "Content", "findings": [_finding(severity="good", issue="x" * 500) for _ in range(30)]}}
        result = compaction.maybe_compact_specialist_reports(reports, threshold=10)
        assert result is not reports
        assert len(result["content"]["findings"]) <= compaction.COMPACTION_MAX_FINDINGS_PER_CATEGORY

    def test_logs_before_and_after_estimate_when_compacting(self):
        logs = []
        reports = {"content": {"category": "Content", "findings": [_finding(severity="good", issue="x" * 500) for _ in range(30)]}}
        compaction.maybe_compact_specialist_reports(reports, threshold=10, log_fn=logs.append)
        assert any("compacted" in msg.lower() for msg in logs)

    def test_no_log_when_under_threshold(self):
        logs = []
        reports = {"technical_seo": {"category": "Technical SEO", "findings": [_finding()]}}
        compaction.maybe_compact_specialist_reports(reports, threshold=100000, log_fn=logs.append)
        assert logs == []


class TestMaybeCompactReport:
    def test_noop_below_threshold(self):
        report = {"categories": [{"name": "A", "findings": [_finding()]}]}
        result = compaction.maybe_compact_report(report, threshold=100000)
        assert result is report

    def test_compacts_above_threshold(self):
        report = {"categories": [{"name": "A", "findings": [_finding(severity="good", issue="x" * 500) for _ in range(30)]}]}
        result = compaction.maybe_compact_report(report, threshold=10)
        assert result is not report
        assert len(result["categories"][0]["findings"]) <= compaction.COMPACTION_MAX_FINDINGS_PER_CATEGORY


class TestIntegrationWithSynthesizerAndCritic:
    """Confirms the actual wiring, not just the compaction functions in
    isolation -- run_synthesizer and critique should call the maybe_compact_*
    functions with the real specialist_reports/draft before building the
    payload sent to the model."""

    def test_run_synthesizer_compacts_before_building_payload(self, monkeypatch):
        from agent import synthesizer

        captured = {}

        def fake_compact(reports, log_fn=None):
            captured["called_with"] = reports
            return {"compacted": True}

        monkeypatch.setattr(synthesizer, "maybe_compact_specialist_reports", fake_compact)

        class FakeAgent:
            def __init__(self, *a, **kw):
                pass

            def run(self, message, expect_json=True):
                captured["message"] = message
                return {"ok": True}

        monkeypatch.setattr(synthesizer, "ToolAgent", FakeAgent)

        original_reports = {"technical_seo": {"category": "Technical SEO", "findings": []}}
        synthesizer.run_synthesizer("https://example.com", original_reports, None)
        assert captured["called_with"] is original_reports
        assert '"compacted": true' in captured["message"].lower()

    def test_critique_compacts_both_draft_and_specialist_reports(self, monkeypatch):
        from agent import critic

        captured = {}
        monkeypatch.setattr(critic, "maybe_compact_report", lambda r, log_fn=None: {"draft_compacted": True})
        monkeypatch.setattr(critic, "maybe_compact_specialist_reports", lambda r, log_fn=None: {"reports_compacted": True})

        class FakeAgent:
            def __init__(self, *a, **kw):
                pass

            def run(self, message, expect_json=True):
                captured["message"] = message
                return {"approved": True, "issues": [], "instructions_for_revision": ""}

        monkeypatch.setattr(critic, "ToolAgent", FakeAgent)

        critic.critique({"url": "https://example.com"}, {"technical_seo": {}})
        assert "draft_compacted" in captured["message"]
        assert "reports_compacted" in captured["message"]