"""
Deterministic ground-truth reconciliation.

LLMs -- especially smaller fallback ones -- sometimes misreport facts a tool
already computed for them with total certainty (e.g. claiming an SSL
certificate is "expired" when the tool's own is_expired field says False).
Prompt instructions reduce this but can't eliminate it, and once a
specialist's free-text finding is wrong, nothing downstream (synthesizer,
critic) can catch it -- they only ever see the specialist's word for it, not
the raw tool data.

This module fixes that gap for high-stakes, easily-verified facts: after a
specialist finishes, we look at what its tools *actually* returned and
force-correct any contradicting findings, rather than trusting the model's
prose. This is intentionally narrow (SSL expiry today) rather than a general
fact-checker -- it targets the specific, repeatedly-observed failure mode.
"""
from __future__ import annotations

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
#
# The competitive specialist was explicitly instructed not to re-judge title
# tag / meta description length (that's the Content specialist's exclusive
# domain), since it doesn't have access to that specialist's actual measured
# data and independently re-deriving the same fact risks contradicting it
# (e.g. claiming a 9-character title is "well within 50-60 characters", or
# claiming a meta description tag is missing when Content already confirmed
# it exists). That prompt instruction alone did not reliably stop it, so this
# strips out any such finding deterministically rather than trusting the model.

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
# report -- occasionally even getting the *direction* backwards. Rather than
# trying to surgically rewrite the model's prose (fragile), append a clear,
# correct note so the real number is never far from the wrong one.

_UP_WORDS = ("improved", "increase", "up ", "risen", "grew", "gained")
_DOWN_WORDS = ("decreased", "decline", "down ", "dropped", "fell", "worsened", "regressed")


def fix_summary_trend_mismatch(report: dict, log_fn=None) -> None:
    """If the summary text's claimed trend direction contradicts the actual
    computed score_delta, append a correcting note rather than trusting the
    model's restated arithmetic."""
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

    mismatch = (claims_up and delta < 0) or (claims_down and delta > 0)
    if mismatch:
        direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        if log_fn:
            log_fn(f"  -> Summary text's trend claim contradicts the actual score_delta "
                   f"({delta:+g}); appending a correction.")
        report["summary"] = (
            summary.rstrip()
            + f" (Note: the summary above may have the trend direction wrong -- the actual "
              f"verified change since the last audit is {direction} {abs(delta):g} points.)"
        )


# --- Fabricated "previous audit" comparisons -----------------------------
#
# A distinct, worse failure mode than a mismatched delta: when there is NO
# real previous audit for a domain at all (first-ever run), the model can
# still confidently write "compared to the previous audit, score increased
# by 1 point" -- inventing a comparison out of nothing rather than getting
# an existing number wrong. fix_summary_trend_mismatch only checks cases
# where real trend data exists, so this needs its own check.

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