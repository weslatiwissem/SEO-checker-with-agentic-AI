from agent.schemas import validate_report, ValidationError


def _valid_report(**overrides):
    base = {
        "url": "https://example.com",
        "overall_score": 82.0,
        "grade": "B",
        "summary": "Decent site overall.",
        "categories": [
            {"name": "Technical SEO", "score": 90.0, "weight": 0.5, "findings": [
                {"severity": "good", "issue": "All good.", "recommendation": ""},
            ]},
            {"name": "Web Security", "score": 74.0, "weight": 0.5, "findings": []},
        ],
        "quick_wins": ["Add alt text."],
        "data_limitations": "",
    }
    base.update(overrides)
    return base


class TestValidateReport:
    def test_valid_report_passes_and_normalizes(self):
        result = validate_report(_valid_report())
        assert result["overall_score"] == 82.0
        assert result["grade"] == "B"
        assert len(result["categories"]) == 2

    def test_missing_required_field_raises(self):
        bad = _valid_report()
        del bad["url"]
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_invalid_grade_letter_raises(self):
        bad = _valid_report(grade="Z")
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_score_out_of_range_raises(self):
        bad = _valid_report(overall_score=150.0)
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_negative_score_raises(self):
        bad = _valid_report(overall_score=-5.0)
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_category_weight_out_of_range_raises(self):
        bad = _valid_report()
        bad["categories"][0]["weight"] = 1.5
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_invalid_finding_severity_raises(self):
        bad = _valid_report()
        bad["categories"][0]["findings"] = [{"severity": "meh", "issue": "x", "recommendation": ""}]
        try:
            validate_report(bad)
            assert False, "expected ValidationError"
        except ValidationError:
            pass

    def test_finding_recommendation_defaults_to_empty_string(self):
        report = _valid_report()
        report["categories"][0]["findings"] = [{"severity": "warning", "issue": "x"}]
        result = validate_report(report)
        assert result["categories"][0]["findings"][0]["recommendation"] == ""

    def test_empty_categories_list_is_allowed(self):
        report = _valid_report(categories=[])
        result = validate_report(report)
        assert result["categories"] == []

    def test_trend_defaults_to_none(self):
        result = validate_report(_valid_report())
        assert result["trend"] is None

    def test_trend_passthrough_when_present(self):
        report = _valid_report()
        report["trend"] = {"previous_score": 75, "score_delta": 7.0, "previous_timestamp": "2026-01-01"}
        result = validate_report(report)
        assert result["trend"]["score_delta"] == 7.0
