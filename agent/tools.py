"""
Concrete implementations of every tool an agent can call. Pure, testable
functions -- each returns a JSON-serializable dict so it can be dropped
straight into a tool-result message for the agent loop.
"""
from __future__ import annotations

import ssl
import socket
import time
import json
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

USER_AGENT = "SEOHealthAgent/2.0 (+https://example.com/bot)"
DEFAULT_TIMEOUT = 10

# Server-side cache: url -> raw HTML. This exists so fetched HTML never has to
# be sent through the model's context (which was previously blowing through
# the free-tier token budget in one or two tool calls). Tools that need the
# HTML look it up here instead of receiving it as a function argument.
_page_cache: dict[str, str] = {}


def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def fetch_page(url: str) -> dict:
    """Fetch a page, cache its HTML server-side, and return only lightweight
    metadata to the model (status, timing, headers) -- NOT the HTML itself.
    Call parse_seo_elements with the same url to extract SEO signals."""
    url = normalize_url(url)
    try:
        start = time.time()
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=DEFAULT_TIMEOUT,
            allow_redirects=True,
        )
        elapsed_ms = round((time.time() - start) * 1000, 1)
        _page_cache[url] = resp.text
        _page_cache[resp.url] = resp.text
        return {
            "ok": True,
            "requested_url": url,
            "final_url": resp.url,
            "redirected": resp.url != url,
            "redirect_chain_length": len(resp.history),
            "status_code": resp.status_code,
            "response_time_ms": elapsed_ms,
            "content_length_bytes": len(resp.content),
            "headers": dict(resp.headers),
            "note": "HTML fetched and cached server-side. Call parse_seo_elements(url=...) "
                    "with this same url to extract SEO signals -- do not request raw HTML.",
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "requested_url": url, "error": str(e)}


def _get_cached_html(url: str) -> str | None:
    """Look up previously-fetched HTML, fetching fresh if not cached."""
    url_n = normalize_url(url)
    if url_n in _page_cache:
        return _page_cache[url_n]
    result = fetch_page(url_n)
    if result.get("ok"):
        return _page_cache.get(result["final_url"]) or _page_cache.get(url_n)
    return None


def parse_seo_elements(url: str) -> dict:
    """Extract on-page SEO signals for a previously-fetched url. Looks up the
    server-side HTML cache populated by fetch_page (fetches fresh if needed)."""
    html = _get_cached_html(url)
    if html is None:
        return {"ok": False, "error": f"Could not retrieve HTML for {url}. Try fetch_page first."}
    base_url = normalize_url(url)

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    canonical = soup.find("link", attrs={"rel": "canonical"})
    viewport = soup.find("meta", attrs={"name": "viewport"})
    charset = soup.find("meta", attrs={"charset": True})

    headings = {f"h{i}": len(soup.find_all(f"h{i}")) for i in range(1, 7)}
    h1_texts = [h.get_text(strip=True) for h in soup.find_all("h1")][:5]

    images = soup.find_all("img")
    images_missing_alt = [img.get("src", "")[:100] for img in images if not img.get("alt")]

    links = soup.find_all("a", href=True)
    domain = urlparse(base_url).netloc
    internal_links, external_links = [], []
    for a in links:
        href = a["href"]
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full = urljoin(base_url, href)
        (internal_links if urlparse(full).netloc == domain else external_links).append(full)

    json_ld_blocks = soup.find_all("script", attrs={"type": "application/ld+json"})
    structured_data_types = []
    for block in json_ld_blocks:
        try:
            data = json.loads(block.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "@type" in item:
                    structured_data_types.append(item["@type"])
        except Exception:
            continue

    text = soup.get_text(separator=" ", strip=True)
    word_count = len(text.split())

    og_tags = {
        m.get("property"): m.get("content")
        for m in soup.find_all("meta", property=lambda p: p and p.startswith("og:"))
    }

    return {
        "title": title_tag.get_text(strip=True) if title_tag else None,
        "title_length": len(title_tag.get_text(strip=True)) if title_tag else 0,
        "meta_description": meta_desc.get("content") if meta_desc else None,
        "meta_description_length": len(meta_desc.get("content")) if meta_desc and meta_desc.get("content") else 0,
        "meta_robots": meta_robots.get("content") if meta_robots else None,
        "canonical_url": canonical.get("href") if canonical else None,
        "has_viewport_meta": viewport is not None,
        "viewport_content": viewport.get("content") if viewport else None,
        "has_charset_meta": charset is not None,
        "heading_counts": headings,
        "h1_texts": h1_texts,
        "total_images": len(images),
        "images_missing_alt_count": len(images_missing_alt),
        "images_missing_alt_sample": images_missing_alt[:10],
        "internal_link_count": len(internal_links),
        "external_link_count": len(external_links),
        "internal_links_sample": internal_links[:15],
        "external_links_sample": external_links[:10],
        "structured_data_types_found": structured_data_types,
        "open_graph_tags": og_tags,
        "word_count": word_count,
        "script_tag_count": len(soup.find_all("script")),
        "stylesheet_count": len(soup.find_all("link", rel="stylesheet")),
    }


def fetch_robots_txt(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        resp = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
        return {
            "ok": True,
            "url": robots_url,
            "status_code": resp.status_code,
            "exists": resp.status_code == 200,
            "content": resp.text[:5000] if resp.status_code == 200 else None,
            "mentions_sitemap": "sitemap:" in resp.text.lower() if resp.status_code == 200 else False,
        }
    except requests.exceptions.RequestException as e:
        return {"ok": False, "url": robots_url, "error": str(e)}


def fetch_sitemap(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    candidates = [
        f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
    ]
    for sitemap_url in candidates:
        try:
            resp = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 200 and "xml" in resp.headers.get("Content-Type", "").lower():
                url_count = resp.text.count("<url>") + resp.text.count("<sitemap>")
                return {
                    "ok": True,
                    "url": sitemap_url,
                    "exists": True,
                    "status_code": resp.status_code,
                    "entry_count_estimate": url_count,
                    "sample_content": resp.text[:2000],
                }
        except requests.exceptions.RequestException:
            continue
    return {"ok": True, "exists": False, "checked_urls": candidates}


def check_ssl_certificate(domain_or_url: str) -> dict:
    parsed = urlparse(normalize_url(domain_or_url))
    hostname = parsed.netloc.split(":")[0]
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=DEFAULT_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()

        expires_raw = cert.get("notAfter")
        is_expired = None
        days_until_expiry = None
        summary = "SSL status could not be determined."
        if expires_raw:
            # e.g. "Aug 18 12:44:57 2026 GMT" -- parse and compare ourselves so
            # the model never has to reason about date arithmetic (it gets
            # this wrong, especially smaller models -- compute it here instead).
            expires_dt = datetime.strptime(expires_raw, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            is_expired = expires_dt < now
            days_until_expiry = (expires_dt - now).days
            summary = (
                f"CERTIFICATE IS EXPIRED (expired {-days_until_expiry} day(s) ago)."
                if is_expired else
                f"Certificate is VALID and NOT expired ({days_until_expiry} day(s) remaining "
                f"until it expires on {expires_raw})."
            )

        return {
            "ok": True,
            "has_valid_ssl": True,
            "issuer": dict(x[0] for x in cert.get("issuer", [])),
            "expires": expires_raw,
            "is_expired": is_expired,
            "days_until_expiry": days_until_expiry,
            "ssl_status_summary": summary,  # READ THIS FIELD -- it's the authoritative, pre-computed answer
            "subject": dict(x[0] for x in cert.get("subject", [])),
        }
    except Exception as e:
        return {"ok": True, "has_valid_ssl": False, "error": str(e)}


def analyze_security_headers(headers: dict) -> dict:
    """Evaluate HTTP response headers (from fetch_page) for security best practices."""
    looks_malformed = (
        not isinstance(headers, dict)
        or not headers
        # a tool-call-shaped wrapper mistakenly passed instead of real headers
        or {"function", "args"}.issubset(headers.keys())
        or {"function", "arguments"}.issubset(headers.keys())
        or "url" in headers
        # real HTTP headers are always string -> string
        or any(not isinstance(v, str) for v in headers.values())
    )
    if looks_malformed:
        return {
            "ok": False,
            "error": (
                "Invalid 'headers' argument. Pass the exact 'headers' object returned "
                "by fetch_page (a flat dict of header-name -> header-value strings), "
                "e.g. {\"Content-Type\": \"text/html\", \"Strict-Transport-Security\": \"...\"}. "
                "Call fetch_page first if you haven't already, then copy its result.headers verbatim."
            ),
        }

    lower_headers = {k.lower(): v for k, v in headers.items()}

    checks = {
        "strict-transport-security": "HSTS forces browsers to use HTTPS.",
        "content-security-policy": "CSP mitigates XSS and data-injection attacks.",
        "x-content-type-options": "Prevents MIME-sniffing (should be 'nosniff').",
        "x-frame-options": "Mitigates clickjacking (should be 'DENY' or 'SAMEORIGIN').",
        "referrer-policy": "Controls how much referrer info is leaked cross-origin.",
        "permissions-policy": "Restricts access to browser features/APIs.",
    }

    present, missing = {}, []
    for header, explanation in checks.items():
        if header in lower_headers:
            present[header] = lower_headers[header]
        else:
            missing.append({"header": header, "why_it_matters": explanation})

    return {
        "present_headers": present,
        "missing_headers": missing,
        "server_banner": lower_headers.get("server"),
        "powered_by_banner": lower_headers.get("x-powered-by"),  # leaking tech stack is itself a minor risk
    }


def check_links_status(urls: list[str]) -> dict:
    """HEAD (fallback GET) a small sample of links to find broken ones."""
    results = []
    for url in urls[:10]:
        try:
            resp = requests.head(url, headers={"User-Agent": USER_AGENT}, timeout=6, allow_redirects=True)
            if resp.status_code >= 400:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=6, allow_redirects=True)
            results.append({"url": url, "status_code": resp.status_code, "broken": resp.status_code >= 400})
        except requests.exceptions.RequestException as e:
            results.append({"url": url, "status_code": None, "broken": True, "error": str(e)})
    broken_count = sum(1 for r in results if r["broken"])
    return {"checked": len(results), "broken_count": broken_count, "results": results}