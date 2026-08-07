from agent import memory


class TestDomainOf:
    def test_extracts_netloc_from_full_url(self):
        assert memory.domain_of("https://example.com/path?x=1") == "example.com"

    def test_adds_scheme_when_missing(self):
        assert memory.domain_of("example.com") == "example.com"

    def test_preserves_subdomain(self):
        assert memory.domain_of("https://blog.example.com") == "blog.example.com"


class TestSaveAndRetrieveAudits:
    def test_save_then_get_last_audit_round_trips(self, tmp_db_path):
        report = {"overall_score": 88, "grade": "B", "url": "https://example.com", "categories": []}
        row_id = memory.save_audit("https://example.com", report)
        assert row_id == 1

        fetched = memory.get_last_audit("https://example.com")
        assert fetched["overall_score"] == 88
        assert fetched["grade"] == "B"
        assert "_timestamp" in fetched

    def test_get_last_audit_returns_none_for_unknown_domain(self, tmp_db_path):
        assert memory.get_last_audit("https://never-audited.example.com") is None

    def test_get_last_audit_returns_most_recent(self, tmp_db_path):
        memory.save_audit("https://example.com", {"overall_score": 60, "grade": "D"})
        memory.save_audit("https://example.com", {"overall_score": 75, "grade": "C"})
        fetched = memory.get_last_audit("https://example.com")
        assert fetched["overall_score"] == 75

    def test_history_scoped_by_domain(self, tmp_db_path):
        memory.save_audit("https://a.com", {"overall_score": 50, "grade": "F"})
        memory.save_audit("https://b.com", {"overall_score": 90, "grade": "A"})
        history_a = memory.get_history("https://a.com")
        assert len(history_a) == 1
        assert history_a[0]["overall_score"] == 50

    def test_history_respects_limit_and_ordering(self, tmp_db_path):
        for score in [60, 65, 70, 75, 80]:
            memory.save_audit("https://example.com", {"overall_score": score, "grade": "C"})
        history = memory.get_history("https://example.com", limit=2)
        assert len(history) == 2
        # Most recent first.
        assert history[0]["overall_score"] == 80
        assert history[1]["overall_score"] == 75

    def test_history_empty_for_unknown_domain(self, tmp_db_path):
        assert memory.get_history("https://nothing-here.example.com") == []

    def test_www_and_non_www_are_different_domains(self, tmp_db_path):
        memory.save_audit("https://www.example.com", {"overall_score": 70, "grade": "C"})
        assert memory.get_last_audit("https://example.com") is None
        assert memory.get_last_audit("https://www.example.com") is not None
