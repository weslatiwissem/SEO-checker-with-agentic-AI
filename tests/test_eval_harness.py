"""Tests for agent/eval_harness.py. run_full_audit is always mocked here --
these tests never make a real network or LLM call, and never spend Groq
quota, matching the harness's own design goal of deterministic grading."""
from __future__ import annotations

from agent import eval_harness


def _report_with_category(name: str, findings: list[dict], review_status: str = "approved") -> dict:
    return {
        "url": "https://example.com",
        "review_status": review_status,
        "categories": [{"name": name, "score": 80, "weight": 1.0, "findings": findings}],
    }


class TestGradeCase:
    def _case(self, **overrides):
        base = {
            "name": "test-case",
            "url": "https://example.com",
            "expected_findings": [
                {"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"},
            ],
        }
        base.update(overrides)
        return base

    def test_passes_when_finding_matches_topic_and_severity(self):
        report = _report_with_category("Web Security", [
            {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it."},
        ])
        graded = eval_harness.grade_case(self._case(), report)
        assert graded["all_passed"] is True
        assert graded["caught_count"] == 1

    def test_tolerates_reworded_finding_not_exact_phrase(self):
        """Same matcher used for Lighthouse dedup -- ordinary rewording
        shouldn't cause a false miss."""
        report = _report_with_category("Web Security", [
            {"severity": "critical", "issue": "The site's certificate is no longer valid; it expired.", "recommendation": "Get a new one."},
        ])
        graded = eval_harness.grade_case(self._case(), report)
        assert graded["all_passed"] is True

    def test_fails_when_category_missing_from_report(self):
        report = {"url": "https://example.com", "review_status": "approved", "categories": []}
        graded = eval_harness.grade_case(self._case(), report)
        assert graded["all_passed"] is False
        assert graded["results"][0]["category_present_in_report"] is False

    def test_fails_when_topic_not_mentioned(self):
        report = _report_with_category("Web Security", [
            {"severity": "warning", "issue": "Missing Content-Security-Policy header.", "recommendation": "Add one."},
        ])
        graded = eval_harness.grade_case(self._case(), report)
        assert graded["all_passed"] is False
        assert graded["results"][0]["topic_caught"] is False

    def test_fails_when_severity_too_low(self):
        report = _report_with_category("Web Security", [
            {"severity": "warning", "issue": "SSL certificate appears expired.", "recommendation": "Renew it."},
        ])
        case = self._case()
        case["expected_findings"][0]["min_severity"] = "critical"
        graded = eval_harness.grade_case(case, report)
        assert graded["all_passed"] is False
        assert graded["results"][0]["topic_caught"] is True
        assert graded["results"][0]["severity_ok"] is False

    def test_passes_when_severity_exceeds_minimum(self):
        report = _report_with_category("Web Security", [
            {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it."},
        ])
        case = self._case()
        case["expected_findings"][0]["min_severity"] = "warning"  # critical satisfies warning minimum
        graded = eval_harness.grade_case(case, report)
        assert graded["all_passed"] is True

    def test_forbidden_phrase_fails_even_if_topic_and_severity_ok(self):
        """The exact regression this exists for: the topic is caught and
        severity looks right, but the finding also contains the specific
        wrong claim we know must never appear (e.g. 'is valid' for a site
        with a definitely-expired cert)."""
        case = self._case()
        case["expected_findings"][0]["must_not_contain"] = ["is valid", "not expired"]
        report = _report_with_category("Web Security", [
            {"severity": "critical", "issue": "SSL certificate is valid but shows an expired warning.", "recommendation": "Investigate."},
        ])
        graded = eval_harness.grade_case(case, report)
        assert graded["all_passed"] is False
        assert "is valid" in graded["results"][0]["forbidden_phrase_hits"]

    def test_multiple_expected_findings_graded_independently(self):
        case = self._case(expected_findings=[
            {"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"},
            {"category": "On-Page Content", "keywords": ["meta", "description"], "min_severity": "warning"},
        ])
        report = {
            "url": "https://example.com", "review_status": "approved",
            "categories": [
                {"name": "Web Security", "score": 50, "weight": 0.5, "findings": [
                    {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew."},
                ]},
                {"name": "On-Page Content", "score": 70, "weight": 0.5, "findings": [
                    {"severity": "good", "issue": "Everything looks fine.", "recommendation": ""},
                ]},
            ],
        }
        graded = eval_harness.grade_case(case, report)
        assert graded["expected_count"] == 2
        assert graded["caught_count"] == 1
        assert graded["all_passed"] is False

    def test_never_raises_on_malformed_report(self):
        graded = eval_harness.grade_case(self._case(), {})
        assert graded["all_passed"] is False
        assert graded["caught_count"] == 0

    def test_real_expired_ssl_benchmark_case_catches_hallucinated_valid_claim(self):
        """Uses the actual expired-ssl-certificate case from BENCHMARK_CASES
        (not a synthetic one) to confirm its must_not_contain guard would
        genuinely catch the exact hallucination reconcile_ssl_findings
        exists to prevent, if that deterministic fix ever regressed."""
        real_case = next(c for c in eval_harness.BENCHMARK_CASES if c["name"] == "expired-ssl-certificate")
        hallucinated_report = _report_with_category("Web Security", [
            {"severity": "good", "issue": "SSL certificate is valid and not expired.", "recommendation": "No action needed."},
        ])
        graded = eval_harness.grade_case(real_case, hallucinated_report)
        assert graded["all_passed"] is False
        assert graded["results"][0]["forbidden_phrase_hits"]

    def test_real_expired_ssl_benchmark_case_passes_correct_report(self):
        real_case = next(c for c in eval_harness.BENCHMARK_CASES if c["name"] == "expired-ssl-certificate")
        correct_report = _report_with_category("Web Security", [
            {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it immediately."},
        ])
        graded = eval_harness.grade_case(real_case, correct_report)
        assert graded["all_passed"] is True


class TestRunEval:
    def _fake_case(self):
        return [{
            "name": "fake-case", "url": "https://example.com",
            "expected_findings": [
                {"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"},
            ],
        }]

    def test_grading_uses_no_llm_calls_only_the_audit_does(self, monkeypatch):
        """Core design guarantee: run_eval's own grading step never touches
        run_full_audit again or calls out to any model -- only the single
        audit call per case does."""
        call_count = {"n": 0}

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            call_count["n"] += 1
            return _report_with_category("Web Security", [
                {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew it."},
            ])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(mode="quick", cases=self._fake_case())
        assert call_count["n"] == 1  # exactly one audit call for one case, nothing extra
        assert summary["recall"] == 1.0

    def test_one_failing_case_does_not_abort_the_others(self, monkeypatch):
        cases = [
            {"name": "broken-case", "url": "https://unreachable.example.com",
             "expected_findings": [{"category": "Web Security", "keywords": ["ssl"], "min_severity": "warning"}]},
            {"name": "good-case", "url": "https://example.com",
             "expected_findings": [{"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"}]},
        ]

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            if "unreachable" in url:
                raise RuntimeError("simulated total pipeline failure")
            return _report_with_category("Web Security", [
                {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew."},
            ])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(mode="quick", cases=cases)
        assert summary["case_count"] == 2
        assert summary["cases"][0]["audit_error"] == "simulated total pipeline failure"
        assert summary["cases"][0]["all_passed"] is False
        assert summary["cases"][1]["all_passed"] is True  # second case still ran and graded

    def test_recall_aggregates_across_all_cases(self, monkeypatch):
        cases = [
            {"name": "case-a", "url": "https://a.example.com",
             "expected_findings": [
                 {"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"},
                 {"category": "On-Page Content", "keywords": ["meta", "description"], "min_severity": "warning"},
             ]},
            {"name": "case-b", "url": "https://b.example.com",
             "expected_findings": [
                 {"category": "Web Security", "keywords": ["ssl", "certificate", "expired"], "min_severity": "critical"},
             ]},
        ]

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            return {
                "url": url, "review_status": "approved",
                "categories": [
                    {"name": "Web Security", "score": 50, "weight": 0.5, "findings": [
                        {"severity": "critical", "issue": "SSL certificate has expired.", "recommendation": "Renew."},
                    ]},
                    {"name": "On-Page Content", "score": 70, "weight": 0.5, "findings": []},  # meta description miss
                ],
            }

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(mode="quick", cases=cases)
        assert summary["total_expected"] == 3
        assert summary["total_caught"] == 2  # both SSL findings caught, meta description missed
        assert summary["recall"] == round(2 / 3, 3)
        assert summary["cases_fully_passed"] == 1  # only case-b (single expectation, fully met)

    def test_default_mode_is_quick_for_cost_reasons(self, monkeypatch):
        captured_modes = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            captured_modes.append(mode)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        eval_harness.run_eval(cases=self._fake_case())
        assert captured_modes == ["quick"]

    def test_use_memory_defaults_to_false(self, monkeypatch):
        """Benchmark runs shouldn't pollute or depend on real audit history."""
        captured = {}

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            captured["use_memory"] = use_memory
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        eval_harness.run_eval(cases=self._fake_case())
        assert captured["use_memory"] is False


class TestBenchmarkCasesShape:
    """Sanity checks on the static benchmark data itself, not the grading
    logic -- catches a malformed case definition before it's ever run."""

    def test_every_case_has_required_fields(self):
        for case in eval_harness.BENCHMARK_CASES:
            assert "name" in case
            assert "url" in case
            assert case["url"].startswith(("http://", "https://"))
            assert "expected_findings" in case
            assert len(case["expected_findings"]) >= 1

    def test_every_expected_finding_has_required_fields(self):
        for case in eval_harness.BENCHMARK_CASES:
            for expected in case["expected_findings"]:
                assert "category" in expected
                assert "keywords" in expected
                assert len(expected["keywords"]) >= 1

    def test_case_names_are_unique(self):
        names = [c["name"] for c in eval_harness.BENCHMARK_CASES]
        assert len(names) == len(set(names))


class TestSamplingAndKeyRotation:
    def _pool(self, n):
        return [
            {"name": f"case-{i}", "url": f"https://site{i}.example.com",
             "expected_findings": [{"category": "Web Security", "keywords": ["ssl"], "min_severity": "warning"}]}
            for i in range(n)
        ]

    def test_default_sample_size_is_four(self):
        assert eval_harness.DEFAULT_SAMPLE_SIZE == 4

    def test_samples_fewer_cases_than_full_pool(self, monkeypatch):
        seen_urls = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_urls.append(url)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(cases=self._pool(10), sample_size=3, seed=42)
        assert len(seen_urls) == 3
        assert summary["sample_size"] == 3
        assert summary["pool_size"] == 10

    def test_same_seed_produces_the_same_sample(self, monkeypatch):
        seen_urls_runs = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_urls_runs[-1].append(url)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        pool = self._pool(10)

        seen_urls_runs.append([])
        eval_harness.run_eval(cases=pool, sample_size=3, seed=123)
        seen_urls_runs.append([])
        eval_harness.run_eval(cases=pool, sample_size=3, seed=123)

        assert seen_urls_runs[0] == seen_urls_runs[1]

    def test_different_seeds_can_produce_different_samples(self, monkeypatch):
        """Not a strict guarantee for every possible pair, but with a pool
        of 10 and a sample of 3, two different seeds producing the exact
        same sample would be a coincidence worth being suspicious of."""
        seen_urls_runs = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_urls_runs[-1].append(url)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        pool = self._pool(10)

        seen_urls_runs.append([])
        eval_harness.run_eval(cases=pool, sample_size=3, seed=1)
        seen_urls_runs.append([])
        eval_harness.run_eval(cases=pool, sample_size=3, seed=2)

        assert seen_urls_runs[0] != seen_urls_runs[1]

    def test_sample_size_capped_at_pool_size(self, monkeypatch):
        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(cases=self._pool(3), sample_size=100)
        assert summary["sample_size"] == 3
        assert summary["case_count"] == 3

    def test_sampling_the_full_pool_preserves_order_no_shuffle(self, monkeypatch):
        seen_urls = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_urls.append(url)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        pool = self._pool(4)
        eval_harness.run_eval(cases=pool, sample_size=4)
        assert seen_urls == [c["url"] for c in pool]

    def test_a_random_seed_is_generated_when_none_given_and_reported(self, monkeypatch):
        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(cases=self._pool(10), sample_size=3)
        assert isinstance(summary["seed"], int)

    def test_each_case_gets_a_different_starting_key_index_round_robin(self, monkeypatch):
        monkeypatch.setattr(eval_harness, "GROQ_API_KEYS", ["key0", "key1", "key2"])
        seen_key_indices = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_key_indices.append(starting_key_index)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        eval_harness.run_eval(cases=self._pool(5), sample_size=5)
        assert seen_key_indices == [0, 1, 2, 0, 1]  # round-robin over 3 keys, 5 cases

    def test_no_key_rotation_when_no_keys_configured(self, monkeypatch):
        monkeypatch.setattr(eval_harness, "GROQ_API_KEYS", [])
        seen_key_indices = []

        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            seen_key_indices.append(starting_key_index)
            return _report_with_category("Web Security", [])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        eval_harness.run_eval(cases=self._pool(3), sample_size=3)
        assert seen_key_indices == [0, 0, 0]

    def test_recall_is_computed_only_over_the_sampled_cases(self, monkeypatch):
        def fake_audit(url, use_memory=False, mode="quick", log_fn=None, starting_key_index=0):
            return _report_with_category("Web Security", [
                {"severity": "warning", "issue": "SSL note.", "recommendation": ""},
            ])

        monkeypatch.setattr(eval_harness, "run_full_audit", fake_audit)
        summary = eval_harness.run_eval(cases=self._pool(10), sample_size=3, seed=7)
        assert summary["total_expected"] == 3  # not 10 -- only the sampled cases count


class TestExpandedBenchmarkPool:
    def test_pool_has_at_least_seven_cases(self):
        assert len(eval_harness.BENCHMARK_CASES) >= 7

    def test_retired_http_only_case_is_not_in_the_active_pool(self):
        """http-only-no-tls was removed after a real run falsified its
        premise (see the comment above BENCHMARK_CASES) -- confirms it
        doesn't silently reappear."""
        names = [c["name"] for c in eval_harness.BENCHMARK_CASES]
        assert "http-only-no-tls" not in names

    def test_all_case_names_still_unique_after_expansion(self):
        names = [c["name"] for c in eval_harness.BENCHMARK_CASES]
        assert len(names) == len(set(names))

    def test_new_ssl_failure_cases_have_forbidden_valid_claim_guard(self):
        for case in eval_harness.BENCHMARK_CASES:
            if "ssl" in case["name"] or "cipher" in case["name"]:
                for expected in case["expected_findings"]:
                    if expected["category"] == "Web Security":
                        assert "must_not_contain" in expected, f"{case['name']} should guard against a false 'is valid' claim"