"""Tests for agent/tools.py. All network calls (requests.get/head, sockets)
are mocked -- these tests never touch the real network."""
from __future__ import annotations

import ssl
import socket
import types
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent import tools


SAMPLE_HTML = """
<html>
<head>
  <title>Example Page Title</title>
  <meta name="description" content="A short description of the example page.">
  <link rel="canonical" href="https://example.com/">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="application/ld+json">{"@type": "Organization", "name": "Example"}</script>
</head>
<body>
  <h1>Main Heading</h1>
  <h2>Sub heading</h2>
  <img src="/logo.png" alt="Logo">
  <img src="/banner.png">
  <a href="/about">About</a>
  <a href="https://external.com/page">External</a>
  <a href="#section">Anchor</a>
  <p>Some body text goes here for word count purposes.</p>
  <meta property="og:title" content="Example Page">
</body>
</html>
"""


def _fake_response(status_code=200, text="", headers=None, url="https://example.com"):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    resp.content = text.encode()
    resp.headers = headers or {"Content-Type": "text/html"}
    resp.url = url
    resp.history = []
    return resp


class TestNormalizeUrl:
    def test_adds_https_when_missing(self):
        assert tools.normalize_url("example.com") == "https://example.com"

    def test_leaves_existing_scheme(self):
        assert tools.normalize_url("http://example.com") == "http://example.com"
        assert tools.normalize_url("https://example.com") == "https://example.com"


class TestFetchPage:
    def setup_method(self):
        tools._page_cache.clear()

    def test_successful_fetch_caches_html_and_returns_metadata(self):
        resp = _fake_response(200, text="<html>hi</html>")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert result["ok"] is True
        assert result["status_code"] == 200
        assert "likely_blocked" not in result
        assert tools._page_cache["https://example.com"] == "<html>hi</html>"

    def test_flags_likely_blocked_on_403(self):
        resp = _fake_response(403, text="Access Denied")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert result["likely_blocked"] is True
        assert "block" in result["block_warning"].lower()

    def test_flags_likely_blocked_on_503_and_429_and_401(self):
        for code in (401, 429, 503):
            resp = _fake_response(code, text="blocked")
            with patch.object(tools.requests, "get", return_value=resp):
                result = tools.fetch_page("example.com")
            assert result["likely_blocked"] is True, f"status {code} should be flagged"

    def test_does_not_flag_normal_4xx_like_404(self):
        resp = _fake_response(404, text="not found")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_page("example.com")
        assert "likely_blocked" not in result

    def test_network_error_returns_not_ok(self):
        with patch.object(tools.requests, "get", side_effect=requests.exceptions.ConnectionError("refused")):
            result = tools.fetch_page("example.com")
        assert result["ok"] is False
        assert "refused" in result["error"]


class TestParseSeoElements:
    def setup_method(self):
        tools._page_cache.clear()

    def test_extracts_all_signals_from_sample_html(self):
        tools._page_cache["https://example.com"] = SAMPLE_HTML
        result = tools.parse_seo_elements("example.com")
        assert result["title"] == "Example Page Title"
        assert result["meta_description"].startswith("A short description")
        assert result["canonical_url"] == "https://example.com/"
        assert result["has_viewport_meta"] is True
        assert result["heading_counts"]["h1"] == 1
        assert result["heading_counts"]["h2"] == 1
        assert result["h1_texts"] == ["Main Heading"]
        assert result["total_images"] == 2
        assert result["images_missing_alt_count"] == 1
        assert result["internal_link_count"] == 1
        assert result["external_link_count"] == 1
        assert "Organization" in result["structured_data_types_found"]
        assert result["open_graph_tags"]["og:title"] == "Example Page"
        assert result["word_count"] > 0

    def test_returns_error_when_html_not_cached_and_fetch_fails(self):
        with patch.object(tools.requests, "get", side_effect=requests.exceptions.ConnectionError("nope")):
            result = tools.parse_seo_elements("neverfetched.example.com")
        assert result["ok"] is False


class TestAnalyzeSecurityHeaders:
    def test_rejects_malformed_tool_call_shaped_input(self):
        result = tools.analyze_security_headers({"function": "foo", "args": {}})
        assert result["ok"] is False

    def test_rejects_dict_containing_url_key(self):
        result = tools.analyze_security_headers({"url": "https://example.com"})
        assert result["ok"] is False

    def test_rejects_empty_or_non_dict(self):
        assert tools.analyze_security_headers({})["ok"] is False
        assert tools.analyze_security_headers(None)["ok"] is False

    def test_identifies_present_and_missing_headers(self):
        headers = {
            "Strict-Transport-Security": "max-age=63072000",
            "Content-Type": "text/html",
        }
        result = tools.analyze_security_headers(headers)
        assert "strict-transport-security" in result["present_headers"]
        missing_names = {m["header"] for m in result["missing_headers"]}
        assert "content-security-policy" in missing_names
        assert "x-frame-options" in missing_names

    def test_case_insensitive_header_matching(self):
        headers = {"STRICT-TRANSPORT-SECURITY": "max-age=1"}
        result = tools.analyze_security_headers(headers)
        assert "strict-transport-security" in result["present_headers"]


class TestCheckLinksStatus:
    def test_flags_broken_links(self):
        ok_resp = _fake_response(200)
        broken_resp = _fake_response(404)
        with patch.object(tools.requests, "head", side_effect=[ok_resp, broken_resp]):
            result = tools.check_links_status(["https://example.com/a", "https://example.com/b"])
        assert result["checked"] == 2
        assert result["broken_count"] == 1

    def test_high_failure_rate_warning_triggers_at_threshold(self):
        broken = _fake_response(500)
        with patch.object(tools.requests, "head", return_value=broken):
            result = tools.check_links_status(["https://a.com", "https://b.com", "https://c.com"])
        assert result["broken_count"] == 3
        assert result["high_failure_rate_warning"] is not None
        assert "blocking" in result["high_failure_rate_warning"].lower()

    def test_no_warning_when_most_links_are_fine(self):
        ok = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=ok):
            result = tools.check_links_status(["https://a.com", "https://b.com", "https://c.com"])
        assert result["high_failure_rate_warning"] is None

    def test_falls_back_to_get_when_head_is_4xx(self):
        head_resp = _fake_response(405)  # method not allowed
        get_resp = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=head_resp), \
             patch.object(tools.requests, "get", return_value=get_resp):
            result = tools.check_links_status(["https://example.com/a"])
        assert result["results"][0]["status_code"] == 200
        assert result["results"][0]["broken"] is False

    def test_request_exception_counts_as_broken(self):
        with patch.object(tools.requests, "head", side_effect=requests.exceptions.Timeout("slow")):
            result = tools.check_links_status(["https://example.com/a"])
        assert result["results"][0]["broken"] is True
        assert result["results"][0]["status_code"] is None

    def test_caps_at_ten_urls(self):
        urls = [f"https://example.com/{i}" for i in range(15)]
        ok = _fake_response(200)
        with patch.object(tools.requests, "head", return_value=ok):
            result = tools.check_links_status(urls)
        assert result["checked"] == 10


class TestCheckSslCertificate:
    def _fake_ssl_context(self, not_after: str):
        cert = {
            "notAfter": not_after,
            "issuer": [[("organizationName", "Fake CA")]],
            "subject": [[("commonName", "example.com")]],
        }
        fake_ssock = MagicMock()
        fake_ssock.getpeercert.return_value = cert
        fake_ssock.__enter__.return_value = fake_ssock
        fake_ssock.__exit__.return_value = False

        fake_ctx = MagicMock()
        fake_ctx.wrap_socket.return_value = fake_ssock
        return fake_ctx

    def test_valid_unexpired_certificate(self):
        future = "Jan  1 00:00:00 2099 GMT"
        fake_ctx = self._fake_ssl_context(future)
        fake_sock = MagicMock()
        fake_sock.__enter__.return_value = fake_sock
        fake_sock.__exit__.return_value = False

        with patch.object(tools.ssl, "create_default_context", return_value=fake_ctx), \
             patch.object(tools.socket, "create_connection", return_value=fake_sock):
            result = tools.check_ssl_certificate("example.com")

        assert result["ok"] is True
        assert result["has_valid_ssl"] is True
        assert result["is_expired"] is False
        assert "VALID" in result["ssl_status_summary"]

    def test_expired_certificate(self):
        past = "Jan  1 00:00:00 2020 GMT"
        fake_ctx = self._fake_ssl_context(past)
        fake_sock = MagicMock()
        fake_sock.__enter__.return_value = fake_sock
        fake_sock.__exit__.return_value = False

        with patch.object(tools.ssl, "create_default_context", return_value=fake_ctx), \
             patch.object(tools.socket, "create_connection", return_value=fake_sock):
            result = tools.check_ssl_certificate("example.com")

        assert result["is_expired"] is True
        assert "EXPIRED" in result["ssl_status_summary"]

    def test_connection_failure_returns_has_valid_ssl_false(self):
        with patch.object(tools.socket, "create_connection", side_effect=OSError("refused")):
            result = tools.check_ssl_certificate("example.com")
        assert result["ok"] is True
        assert result["has_valid_ssl"] is False
        assert "refused" in result["error"]


class TestCheckCoreWebVitals:
    LIGHTHOUSE_PAYLOAD = {
        "lighthouseResult": {
            "categories": {"performance": {"score": 0.42}},
            "audits": {
                "largest-contentful-paint": {"numericValue": 4200},
                "cumulative-layout-shift": {"numericValue": 0.25},
                "total-blocking-time": {"numericValue": 300},
                "speed-index": {"numericValue": 5000},
                "first-contentful-paint": {"numericValue": 1800},
                "interactive": {"numericValue": 6000},
            },
        },
        "loadingExperience": {},
    }

    def test_successful_response_parses_lab_data(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is True
        assert result["lab_data"]["performance_score_0_100"] == 42
        assert result["lab_data"]["lcp_ms"] == 4200
        assert result["field_data"] is None  # no loadingExperience.metrics in payload

    def test_retries_on_transient_500_then_succeeds(self):
        bad = MagicMock(status_code=500, text="server error")
        good = MagicMock(status_code=200)
        good.json.return_value = self.LIGHTHOUSE_PAYLOAD
        with patch.object(tools.requests, "get", side_effect=[bad, good]), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is True

    def test_gives_up_after_max_attempts(self):
        bad = MagicMock(status_code=500, text="server error")
        with patch.object(tools.requests, "get", return_value=bad), \
             patch.object(tools.time, "sleep", return_value=None):
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is False
        assert "fall back" in result["note"].lower()

    def test_non_retryable_status_fails_immediately(self):
        bad = MagicMock(status_code=400, text="bad request")
        with patch.object(tools.requests, "get", return_value=bad) as mock_get:
            result = tools.check_core_web_vitals("example.com")
        assert result["ok"] is False
        assert mock_get.call_count == 1  # no retry on a non-5xx failure


class TestFetchRobotsAndSitemap:
    def test_fetch_robots_txt_detects_sitemap_reference(self):
        resp = _fake_response(200, text="User-agent: *\nSitemap: https://example.com/sitemap.xml")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_robots_txt("example.com")
        assert result["exists"] is True
        assert result["mentions_sitemap"] is True

    def test_fetch_robots_txt_missing(self):
        resp = _fake_response(404, text="")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_robots_txt("example.com")
        assert result["exists"] is False

    def test_fetch_sitemap_found(self):
        xml = '<?xml version="1.0"?><urlset><url><loc>a</loc></url><url><loc>b</loc></url></urlset>'
        resp = _fake_response(200, text=xml, headers={"Content-Type": "application/xml"})
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_sitemap("example.com")
        assert result["exists"] is True
        assert result["entry_count_estimate"] == 2

    def test_fetch_sitemap_not_found(self):
        resp = _fake_response(404, text="")
        with patch.object(tools.requests, "get", return_value=resp):
            result = tools.fetch_sitemap("example.com")
        assert result["exists"] is False
