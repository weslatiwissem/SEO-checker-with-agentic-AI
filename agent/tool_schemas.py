"""Groq (OpenAI-compatible) tool-use schemas, organized into reusable groups.

Each entry follows the standard `{"type": "function", "function": {...}}`
shape expected by Groq's chat.completions.create(tools=...) parameter.
"""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


FETCH_PAGE = _tool(
    "fetch_page",
    (
        "Fetch a web page over HTTP(S). Returns status code, response time, "
        "headers, content length, and the raw HTML (truncated). Always call "
        "this first on the target URL before anything else."
    ),
    {"url": {"type": "string", "description": "Full URL or domain to fetch"}},
    ["url"],
)

PARSE_SEO_ELEMENTS = _tool(
    "parse_seo_elements",
    (
        "Extract on-page SEO signals for a URL you already fetched with fetch_page: "
        "title, meta description, headings, images missing alt text, internal/external "
        "links, structured data, Open Graph tags, word count. Just pass the same url -- "
        "the HTML is cached server-side, do NOT try to pass HTML content yourself."
    ),
    {"url": {"type": "string", "description": "The same URL previously fetched with fetch_page"}},
    ["url"],
)

FETCH_ROBOTS_TXT = _tool(
    "fetch_robots_txt",
    "Fetch and check the site's robots.txt file for existence and sitemap references.",
    {"domain_or_url": {"type": "string"}},
    ["domain_or_url"],
)

FETCH_SITEMAP = _tool(
    "fetch_sitemap",
    "Check for the existence of sitemap.xml / sitemap_index.xml and estimate its size.",
    {"domain_or_url": {"type": "string"}},
    ["domain_or_url"],
)

CHECK_SSL_CERTIFICATE = _tool(
    "check_ssl_certificate",
    "Check whether the site has a valid HTTPS/SSL certificate and when it expires.",
    {"domain_or_url": {"type": "string"}},
    ["domain_or_url"],
)

ANALYZE_SECURITY_HEADERS = _tool(
    "analyze_security_headers",
    (
        "Evaluate HTTP response headers against security best practices: HSTS, CSP, "
        "X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy. "
        "The 'headers' argument must be EXACTLY the 'headers' object from a prior "
        "fetch_page call's result -- copy that dict verbatim, e.g. "
        '{"headers": {"Content-Type": "text/html", "Strict-Transport-Security": "max-age=63072000"}}. '
        "Do not pass a URL, a tool-call description, or anything other than that headers dict."
    ),
    {
        "headers": {
            "type": "object",
            "description": "The exact headers dict copied from fetch_page's result.headers field",
        }
    },
    ["headers"],
)

CHECK_LINKS_STATUS = _tool(
    "check_links_status",
    (
        "Check a list of URLs (e.g. internal or external links found on the page) "
        "for broken links (4xx/5xx status codes). Pass at most ~8 URLs."
    ),
    {"urls": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
    ["urls"],
)

CHECK_CORE_WEB_VITALS = _tool(
    "check_core_web_vitals",
    (
        "Run a REAL Lighthouse audit via Google's PageSpeed Insights API -- not a proxy "
        "signal. Returns actual LCP, CLS, Total Blocking Time (INP proxy), Speed Index, and "
        "a 0-100 performance score, plus real-world Chrome user data (CrUX) when available. "
        "Takes 10-30+ seconds since Google runs a real audit server-side. If it fails (ok: "
        "false), fall back to proxy signals from fetch_page/parse_seo_elements instead."
    ),
    {
        "url": {"type": "string", "description": "The URL to audit"},
        "strategy": {"type": "string", "description": "\"mobile\" or \"desktop\" (default mobile)"},
    },
    ["url"],
)

CHECK_ACCESSIBILITY_AND_BEST_PRACTICES = _tool(
    "check_accessibility_and_best_practices",
    (
        "Run a REAL Lighthouse accessibility audit via Google's PageSpeed Insights API -- "
        "same underlying data source as check_core_web_vitals, so calling both for the same "
        "URL is cheap. Returns a 0-100 accessibility score plus the specific WCAG-adjacent "
        "checks that failed (color contrast, missing alt text, unlabeled form fields, "
        "missing ARIA attributes, etc.) with each audit's id/title/description. This is "
        "real automated-audit data, but automated tools only catch roughly 30-50% of real "
        "WCAG issues -- treat a clean result as a floor, not proof of full compliance."
    ),
    {
        "url": {"type": "string", "description": "The URL to audit"},
        "strategy": {"type": "string", "description": "\"mobile\" or \"desktop\" (default mobile)"},
    },
    ["url"],
)

CHECK_BEST_PRACTICES = _tool(
    "check_best_practices",
    (
        "Run a REAL Lighthouse Best Practices audit via Google's PageSpeed Insights API -- "
        "same underlying data source as check_core_web_vitals, so calling it alongside that "
        "or check_accessibility_and_best_practices for the same URL is cheap. Returns a "
        "0-100 score plus the specific checks that failed: known-vulnerable JS libraries, "
        "deprecated browser APIs, missing charset/doctype, unsafe third-party patterns, "
        "console errors, etc. HTTPS/SSL checks are deliberately excluded -- that's the "
        "Security specialist's domain."
    ),
    {
        "url": {"type": "string", "description": "The URL to audit"},
        "strategy": {"type": "string", "description": "\"mobile\" or \"desktop\" (default mobile)"},
    },
    ["url"],
)

# Named groups handed to each specialist agent. The "competitive" specialist
# intentionally has no client tools -- it runs on Groq's Compound system,
# which performs web search server-side and does not support custom tools
# being mixed in alongside its built-in ones.
TOOL_GROUPS = {
    "technical_seo": [FETCH_PAGE, FETCH_ROBOTS_TXT, FETCH_SITEMAP, CHECK_SSL_CERTIFICATE, PARSE_SEO_ELEMENTS],
    "content": [FETCH_PAGE, PARSE_SEO_ELEMENTS],
    "performance": [FETCH_PAGE, PARSE_SEO_ELEMENTS, CHECK_CORE_WEB_VITALS],
    "security": [FETCH_PAGE, CHECK_SSL_CERTIFICATE, ANALYZE_SECURITY_HEADERS],
    "links": [FETCH_PAGE, PARSE_SEO_ELEMENTS, CHECK_LINKS_STATUS],
    "accessibility": [FETCH_PAGE, PARSE_SEO_ELEMENTS, CHECK_ACCESSIBILITY_AND_BEST_PRACTICES],
    "best_practices": [FETCH_PAGE, CHECK_BEST_PRACTICES],
    "competitive": [],
}