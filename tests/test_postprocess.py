"""Tests for agent/postprocess.py -- the deterministic ground-truth layer.

Each test here corresponds to a specific, documented bug in PROJECT_HANDOFF.md
(hallucinated SSL status, competitive-specialist overlap, summary/trend
mismatches, bot-blocked fetches poisoning findings, CWV data being ignored).
"""
from agent import postprocess


# --------------------------------------------------------------------------
# reconcile_ssl_findings
# --------------------------------------------------------------------------

class TestReconcileSslFindings:
    def test_noop_when_ssl_tool_never_called(self):
        result = {"category": "Web Security", "findings": [{"severity": "good", "issue": "fine", "recommendation": ""}]}
        out = postprocess.reconcile_ssl_findings(result, tool_call_log=[])
        assert out["findings"] == result["findings"]

    def test_valid_cert_overrides_model_claim_of_expiry(self, sample_ssl_tool_call_log_valid):
        # Model hallucinated an expired-certificate finding despite the tool
        # proving the cert is valid -- this is the exact bug from the handoff.
        result = {
            "category": "Web Security",
            "findings": [
                {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it."},
                {"severity": "good", "issue": "Unrelated: HSTS header present.", "recommendation": ""},
            ],
        }
        out = postprocess.reconcile_ssl_findings(result, sample_ssl_tool_call_log_valid)
        ssl_findings = [f for f in out["findings"] if "certificate" in f["issue"].lower()]
        assert len(ssl_findings) == 1
        assert ssl_findings[0]["severity"] == "good"
        assert "VALID" in ssl_findings[0]["issue"]
        # Unrelated finding must survive untouched.
        assert any("HSTS" in f["issue"] for f in out["findings"])

    def test_expired_cert_produces_critical_finding(self, sample_ssl_tool_call_log_expired):
        result = {"category": "Web Security", "findings": []}
        out = postprocess.reconcile_ssl_findings(result, sample_ssl_tool_call_log_expired)
        assert len(out["findings"]) == 1
        assert out["findings"][0]["severity"] == "critical"
        assert "EXPIRED" in out["findings"][0]["issue"]

    def test_failed_handshake_produces_critical_finding(self):
        log = [{"name": "check_ssl_certificate", "result": {
            "ok": True, "has_valid_ssl": False, "error": "connection refused",
        }}]
        out = postprocess.reconcile_ssl_findings({"findings": []}, log)
        assert out["findings"][0]["severity"] == "critical"
        assert "connection refused" in out["findings"][0]["issue"]

    def test_uses_latest_ssl_call_if_multiple(self):
        log = [
            {"name": "check_ssl_certificate", "result": {"ok": True, "has_valid_ssl": True, "is_expired": True, "ssl_status_summary": "OLD"}},
            {"name": "check_ssl_certificate", "result": {"ok": True, "has_valid_ssl": True, "is_expired": False, "ssl_status_summary": "Certificate is VALID and NOT expired (NEW)."}},
        ]
        out = postprocess.reconcile_ssl_findings({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"
        assert "NEW" in out["findings"][0]["issue"]

    def test_ignores_failed_ssl_tool_calls(self):
        log = [{"name": "check_ssl_certificate", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": [{"severity": "good", "issue": "unrelated", "recommendation": ""}]}
        out = postprocess.reconcile_ssl_findings(result, log)
        # ok: False result should be treated as if the tool was never called.
        assert out["findings"] == result["findings"]


# --------------------------------------------------------------------------
# strip_competitive_onpage_overlap
# --------------------------------------------------------------------------

class TestStripCompetitiveOnpageOverlap:
    def test_removes_title_length_judgment(self):
        result = {"category": "Competitive", "findings": [
            {"severity": "warning", "issue": "The title is well within the 50-60 character recommendation.", "recommendation": ""},
            {"severity": "good", "issue": "Structured data usage matches industry norms.", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert len(out["findings"]) == 1
        assert "Structured data" in out["findings"][0]["issue"]

    def test_removes_any_canonical_mention_unconditionally(self):
        result = {"findings": [
            {"severity": "good", "issue": "Canonical link element present and correctly points to the primary URL.", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert out["findings"] == []

    def test_meta_description_requires_context_word(self):
        # "meta description" alone with no length/quality context shouldn't
        # match by topic-keyword alone unless a context word is present too.
        result = {"findings": [
            {"severity": "good", "issue": "meta description is missing entirely", "recommendation": ""},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert out["findings"] == []  # "missing" is a context keyword -> stripped

    def test_keeps_findings_with_no_overlap(self):
        result = {"findings": [
            {"severity": "warning", "issue": "Competitor uses more schema.org markup types.", "recommendation": "Add Product schema."},
        ]}
        out = postprocess.strip_competitive_onpage_overlap(result)
        assert len(out["findings"]) == 1


# --------------------------------------------------------------------------
# fix_summary_trend_mismatch
# --------------------------------------------------------------------------

class TestFixSummaryTrendMismatch:
    def test_direction_mismatch_appends_note(self):
        report = {
            "trend": {"score_delta": -5.0, "previous_score": 80, "previous_timestamp": "t"},
            "summary": "The site has improved significantly since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" in report["summary"]

    def test_magnitude_mismatch_appends_note(self):
        report = {
            "trend": {"score_delta": 8.0, "previous_score": 70, "previous_timestamp": "t"},
            "summary": "Scores improved by 1.5 points since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" in report["summary"]

    def test_matching_direction_and_magnitude_is_left_alone(self):
        report = {
            "trend": {"score_delta": 5.0, "previous_score": 75, "previous_timestamp": "t"},
            "summary": "The score increased by 5 points since the last audit.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" not in report["summary"]

    def test_noop_without_trend(self):
        report = {"summary": "The site improved a lot."}
        postprocess.fix_summary_trend_mismatch(report)
        assert report["summary"] == "The site improved a lot."

    def test_noop_without_directional_claim(self):
        report = {
            "trend": {"score_delta": -5.0},
            "summary": "The site has several accessibility issues to fix.",
        }
        postprocess.fix_summary_trend_mismatch(report)
        assert "Note:" not in report["summary"]


# --------------------------------------------------------------------------
# fix_fabricated_trend_claim
# --------------------------------------------------------------------------

class TestFixFabricatedTrendClaim:
    def test_flags_fabricated_comparison_on_first_audit(self):
        report = {"summary": "Compared to the previous audit, this site has improved."}
        postprocess.fix_fabricated_trend_claim(report)
        assert "fabricated" in report["summary"].lower()

    def test_noop_when_real_trend_exists(self):
        report = {"trend": {"score_delta": 2.0}, "summary": "Compared to the previous audit, up slightly."}
        postprocess.fix_fabricated_trend_claim(report)
        assert "fabricated" not in report["summary"].lower()

    def test_noop_when_no_previous_audit_phrase_used(self):
        report = {"summary": "This site has strong technical SEO fundamentals."}
        postprocess.fix_fabricated_trend_claim(report)
        assert report["summary"] == "This site has strong technical SEO fundamentals."


# --------------------------------------------------------------------------
# reconcile_likely_blocked
# --------------------------------------------------------------------------

class TestReconcileLikelyBlocked:
    def test_overrides_findings_when_blocked(self):
        log = [{"name": "fetch_page", "result": {"ok": True, "status_code": 403, "likely_blocked": True}}]
        result = {"category": "Content", "score": 20, "findings": [{"severity": "critical", "issue": "No H1 found", "recommendation": "Add one"}]}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] is None
        assert len(out["findings"]) == 1
        assert "block" in out["findings"][0]["issue"].lower()

    def test_noop_when_not_blocked(self):
        log = [{"name": "fetch_page", "result": {"ok": True, "status_code": 200, "likely_blocked": False}}]
        result = {"score": 90, "findings": [{"severity": "good", "issue": "fine", "recommendation": ""}]}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] == 90
        assert out["findings"] == result["findings"]

    def test_noop_when_fetch_page_never_called(self):
        result = {"score": 90, "findings": []}
        out = postprocess.reconcile_likely_blocked(result, [])
        assert out["score"] == 90

    def test_uses_latest_fetch_page_call(self):
        log = [
            {"name": "fetch_page", "result": {"ok": True, "status_code": 200, "likely_blocked": False}},
            {"name": "fetch_page", "result": {"ok": True, "status_code": 503, "likely_blocked": True}},
        ]
        result = {"score": 90, "findings": []}
        out = postprocess.reconcile_likely_blocked(result, log)
        assert out["score"] is None


# --------------------------------------------------------------------------
# reconcile_core_web_vitals
# --------------------------------------------------------------------------

class TestReconcileCoreWebVitals:
    CWV_LOG = [{
        "name": "check_core_web_vitals",
        "result": {
            "ok": True,
            "lab_data": {"performance_score_0_100": 42, "lcp_ms": 4200, "cls": 0.25},
            "good_thresholds_2026": {"lcp_ms": 2500, "cls": 0.1, "inp_ms": 200},
        },
    }]

    def test_injects_canonical_finding_when_model_ignored_real_data(self):
        result = {"category": "Page Speed", "findings": [
            {"severity": "warning", "issue": "Page load time is higher than expected.", "recommendation": "Optimize images."},
        ]}
        out = postprocess.reconcile_core_web_vitals(result, self.CWV_LOG)
        assert out["_real_cwv_available"] is True
        assert any("42" in f["issue"] for f in out["findings"])
        assert any(f["severity"] == "critical" for f in out["findings"])  # score 42 < 50

    def test_noop_when_model_already_cites_real_numbers(self):
        result = {"findings": [
            {"severity": "critical", "issue": "Lighthouse performance score is 42/100 with LCP at 4200ms.", "recommendation": "Fix it."},
        ]}
        out = postprocess.reconcile_core_web_vitals(result, self.CWV_LOG)
        assert len(out["findings"]) == 1  # nothing appended
        assert "_real_cwv_available" not in out

    def test_noop_when_cwv_tool_never_called(self):
        result = {"findings": []}
        out = postprocess.reconcile_core_web_vitals(result, [])
        assert out["findings"] == []

    def test_noop_when_cwv_tool_failed(self):
        log = [{"name": "check_core_web_vitals", "result": {"ok": False, "error": "timeout"}}]
        result = {"findings": []}
        out = postprocess.reconcile_core_web_vitals(result, log)
        assert out["findings"] == []

    def test_good_severity_for_high_score(self):
        log = [{"name": "check_core_web_vitals", "result": {
            "ok": True, "lab_data": {"performance_score_0_100": 96, "lcp_ms": 1200, "cls": 0.02},
            "good_thresholds_2026": {"lcp_ms": 2500, "cls": 0.1, "inp_ms": 200},
        }}]
        out = postprocess.reconcile_core_web_vitals({"findings": []}, log)
        assert out["findings"][0]["severity"] == "good"


# --------------------------------------------------------------------------
# fix_stale_cwv_data_limitations
# --------------------------------------------------------------------------

class TestFixStaleCwvDataLimitations:
    def test_corrects_stale_claim_when_real_data_was_obtained(self):
        report = {"data_limitations": "No real Core Web Vitals data was available for this audit."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=True)
        assert "Note:" in report["data_limitations"]

    def test_noop_when_no_real_cwv_data_obtained(self):
        report = {"data_limitations": "No real Core Web Vitals data was available."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=False)
        assert "Note:" not in report["data_limitations"]

    def test_noop_when_text_is_not_stale(self):
        report = {"data_limitations": "Link checking was a sample, not a full crawl."}
        postprocess.fix_stale_cwv_data_limitations(report, had_real_cwv=True)
        assert "Note:" not in report["data_limitations"]
