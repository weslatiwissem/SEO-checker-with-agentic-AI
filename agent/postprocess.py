"""
Deterministic ground-truth reconciliation.

LLMs -- especially smaller fallback ones -- sometimes misreport facts a tool
already computed for them with total certainty, or restate numbers in prose
that contradict deterministic data computed elsewhere in the same report.
Prompt instructions reduce this but can't eliminate it, so this module fixes
specific, repeatedly-observed failure classes in code instead of trusting
the model to self-correct.
"""
from __future__ import annotations

import re

_SSL_KEYWORDS = ("certificate", "ssl", "https/ssl", "tls")
_SSL_VALIDITY_KEYWORDS = ("expir", "valid", "not set")


def _is_ssl_finding(finding: dict) -> bool:
    text = (finding.get("issue", "") + " " + finding.get("recommendation", "")).lower()
    return any(k in text for k in _SSL_KEYWORDS) and any(k in text for k in _SSL_VALIDITY_KEYWORDS)


def _latest_ssl_tool_result(tool_call_log: list[dict]) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == "check_ssl_certificate":
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def reconcile_ssl_findings(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """Replace any SSL-expiry/validity findings in specialist_result with a
    single, deterministically-correct one built from the actual tool output.
    No-op if the specialist never called check_ssl_certificate."""
    ssl_result = _latest_ssl_tool_result(tool_call_log)
    if ssl_result is None:
        return specialist_result

    findings = specialist_result.get("findings", [])
    kept = [f for f in findings if not _is_ssl_finding(f)]
    removed_count = len(findings) - len(kept)

    if ssl_result.get("has_valid_ssl") is False:
        canonical = {
            "severity": "critical",
            "issue": f"Could not verify an SSL certificate: {ssl_result.get('error', 'connection failed')}.",
            "recommendation": "Investigate why the HTTPS/SSL handshake is failing.",
        }
    elif ssl_result.get("is_expired") is True:
        canonical = {
            "severity": "critical",
            "issue": f"SSL certificate has expired. {ssl_result.get('ssl_status_summary', '')}".strip(),
            "recommendation": "Renew the SSL certificate immediately.",
        }
    else:
        canonical = {
            "severity": "good",
            "issue": f"SSL certificate is valid. {ssl_result.get('ssl_status_summary', '')}".strip(),
            "recommendation": "No action needed; renew before it approaches expiry.",
        }

    kept.append(canonical)
    specialist_result["findings"] = kept

    if removed_count and log_fn:
        log_fn(
            f"  -> Corrected {removed_count} inaccurate SSL finding(s) in "
            f"'{specialist_result.get('category', '?')}' using verified tool data "
            f"(is_expired={ssl_result.get('is_expired')})"
        )

    return specialist_result


# --- Competitive specialist / on-page overlap ---------------------------

_ONPAGE_OVERLAP_TOPIC_KEYWORDS = ("title tag", "title length", "title is", "meta description")
_ONPAGE_OVERLAP_CONTEXT_KEYWORDS = ("character", "length", "within", "recommend", "missing", "lacks", "omission")


def _is_onpage_overlap_finding(finding: dict) -> bool:
    text = (finding.get("issue", "") + " " + finding.get("recommendation", "")).lower()
    mentions_topic = any(k in text for k in _ONPAGE_OVERLAP_TOPIC_KEYWORDS)
    mentions_context = any(k in text for k in _ONPAGE_OVERLAP_CONTEXT_KEYWORDS)
    return mentions_topic and mentions_context


def strip_competitive_onpage_overlap(specialist_result: dict, log_fn=None) -> dict:
    """Remove any competitive-specialist finding that re-judges title tag or
    meta description length/presence -- that's the Content specialist's job,
    and the competitive specialist doesn't share its ground-truth data."""
    findings = specialist_result.get("findings", [])
    kept = [f for f in findings if not _is_onpage_overlap_finding(f)]
    removed_count = len(findings) - len(kept)

    specialist_result["findings"] = kept

    if removed_count and log_fn:
        log_fn(
            f"  -> Removed {removed_count} duplicate/conflicting on-page finding(s) from "
            f"'{specialist_result.get('category', '?')}' -- title/meta description length is "
            f"the Content specialist's domain, not competitive's."
        )

    return specialist_result


# --- Summary text vs. real trend data ------------------------------------
#
# The synthesizer sometimes writes a specific point-delta claim in its prose
# summary ("up 3 points", "improved by 8%") that contradicts the actual,
# deterministically-computed trend sitting right next to it in the same
# report -- sometimes getting the *direction* backwards, and sometimes
# agreeing on direction but stating the wrong *magnitude* (e.g. "increased
# by 1.5 points" when the real delta is +5). Rather than trying to
# surgically rewrite the model's prose (fragile), append a clear, correct
# note so the real number is never far from the wrong one.

_UP_WORDS = ("improved", "increase", "up ", "risen", "grew", "gained")
_DOWN_WORDS = ("decreased", "decline", "down ", "dropped", "fell", "worsened", "regressed")
_POINT_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:point|points|%|percent)", re.IGNORECASE)


def fix_summary_trend_mismatch(report: dict, log_fn=None) -> None:
    """If the summary text's claimed trend direction OR magnitude contradicts
    the actual computed score_delta, append a correcting note rather than
    trusting the model's restated arithmetic."""
    trend = report.get("trend")
    summary = report.get("summary", "")
    if not trend or not summary:
        return

    delta = trend.get("score_delta")
    if delta is None:
        return

    summary_lower = summary.lower()
    claims_up = any(w in summary_lower for w in _UP_WORDS)
    claims_down = any(w in summary_lower for w in _DOWN_WORDS)

    direction_mismatch = (claims_up and delta < 0) or (claims_down and delta > 0)

    magnitude_mismatch = False
    if (claims_up or claims_down) and not direction_mismatch:
        match = _POINT_NUMBER_RE.search(summary_lower)
        if match:
            claimed_magnitude = float(match.group(1))
            real_magnitude = abs(delta)
            # Tolerate small rounding differences; flag anything meaningfully off.
            tolerance = max(1.0, real_magnitude * 0.25)
            if abs(claimed_magnitude - real_magnitude) > tolerance:
                magnitude_mismatch = True

    if direction_mismatch or magnitude_mismatch:
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        reason = "direction" if direction_mismatch else "magnitude"
        if log_fn:
            log_fn(f"  -> Summary text's trend claim has a {reason} mismatch vs actual "
                   f"score_delta ({delta:+g}); appending a correction.")
        report["summary"] = (
            summary.rstrip()
            + f" (Note: the summary above may misstate the change -- the actual verified "
              f"change since the last audit is {direction} {abs(delta):g} points.)"
        )


# --- Fabricated "previous audit" comparisons -----------------------------

_PREVIOUS_AUDIT_PHRASES = (
    "previous audit", "last audit", "prior audit", "earlier audit",
    "since the last", "compared to the previous", "compared to the last",
)


def fix_fabricated_trend_claim(report: dict, log_fn=None) -> None:
    """If no real trend data exists (no prior audit for this domain), but the
    summary still claims a comparison to a previous audit, append a
    correction rather than leaving a fabricated data point unchallenged."""
    if report.get("trend"):
        return  # a real previous audit exists -- fix_summary_trend_mismatch handles that case

    summary = report.get("summary", "")
    if not summary:
        return

    summary_lower = summary.lower()
    if any(phrase in summary_lower for phrase in _PREVIOUS_AUDIT_PHRASES):
        if log_fn:
            log_fn("  -> Summary claims a comparison to a previous audit, but no prior audit "
                   "exists for this domain; appending a correction.")
        report["summary"] = (
            summary.rstrip()
            + " (Note: this is the first recorded audit for this domain -- there is no actual "
              "previous audit to compare against, so any such comparison above is fabricated "
              "and should be disregarded.)"
        )


# --- Likely bot-blocked fetches --------------------------------------------
#
# When a site's anti-bot/WAF protection returns 401/403/429/503 instead of
# the real page, that response gets cached and every specialist that reuses
# it ends up scoring a block/challenge page as if it were the real site --
# "no headings", "no images", "missing meta description" etc. are typical
# symptoms of this, not real content quality problems. Prompting specialists
# to notice fetch_page's likely_blocked flag isn't reliable enough on its
# own, so this deterministically overrides the specialist's output instead
# of trusting the model to correctly discount a blocked fetch.

def _latest_fetch_page_result(tool_call_log: list[dict]) -> dict | None:
    for entry in reversed(tool_call_log):
        if entry.get("name") == "fetch_page":
            result = entry.get("result") or {}
            if result.get("ok"):
                return result
    return None


def reconcile_likely_blocked(specialist_result: dict, tool_call_log: list[dict], log_fn=None) -> dict:
    """If this specialist's fetch_page result was flagged likely_blocked,
    override its score/findings with a single, honest 'could not verify'
    finding rather than trusting whatever it inferred from a block page."""
    fetch_result = _latest_fetch_page_result(tool_call_log)
    if not fetch_result or not fetch_result.get("likely_blocked"):
        return specialist_result

    status = fetch_result.get("status_code")
    category = specialist_result.get("category", "?")

    if log_fn:
        log_fn(
            f"  -> '{category}' fetch returned status {status} (likely bot-blocking) -- "
            f"overriding its findings with a single block-warning finding rather than "
            f"trusting scores/findings derived from what is probably a block page, not "
            f"the real site."
        )

    specialist_result["score"] = None
    specialist_result["findings"] = [{
        "severity": "critical",
        "issue": (
            f"Could not reliably audit real page content for this category -- the request "
            f"returned HTTP {status}, which commonly indicates bot/WAF blocking rather than "
            f"a genuine site issue."
        ),
        "recommendation": (
            "Manually verify the page loads correctly in a real browser. This category's "
            "automated findings are not reliable until access is confirmed; it has been "
            "excluded from the overall score rather than scored against a likely block page."
        ),
    }]
    specialist_result["raw_evidence_notes"] = (
        f"fetch_page returned status {status} and was flagged likely_blocked; "
        f"findings suppressed and replaced with this notice."
    )
    return specialist_result