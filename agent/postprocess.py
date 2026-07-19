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