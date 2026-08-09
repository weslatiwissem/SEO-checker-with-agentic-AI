"""
Proactive payload compaction for the synthesizer and critic.

base_agent.py's _shrink_largest_message already reacts to a 413 "request too
large" error after the fact -- effective (it recovers), but wasteful: every
time it fires, that's a failed API call plus retry latency the run didn't
need to pay, right in the middle of a live audit. Observed in practice on
the Competitive specialist's payload on at least one real run.

This module estimates the outgoing payload size BEFORE the first send and,
only if it's large enough to be at real risk, deterministically trims the
least load-bearing content -- so the first request has a real chance of
succeeding outright instead of needing a round-trip failure to find out it
was too big.

What gets trimmed, in order, and why that order: low-priority ("good")
findings are dropped before any critical/warning finding ever is, since a
report can least afford to lose the high-severity findings that justify its
scores. Overlong free-text fields (raw_evidence_notes, individual
issue/recommendation strings) are truncated next. Nothing is trimmed at all
if the estimated payload is already under the threshold -- this only
activates when there's a real, size-driven reason to.

Uses a simple characters-per-token heuristic rather than an exact tokenizer,
since Groq doesn't publish one per model and an approximate trigger is all
a "is this at risk of a 413" check needs -- getting within ~20% of the real
count is enough to meaningfully cut 413s without the complexity of an exact
count.
"""
from __future__ import annotations

import copy
import json

from .config import (
    COMPACTION_TOKEN_THRESHOLD, COMPACTION_MAX_FINDINGS_PER_CATEGORY,
    COMPACTION_MAX_EVIDENCE_NOTES_CHARS, COMPACTION_MAX_FINDING_TEXT_CHARS,
)

_CHARS_PER_TOKEN_ESTIMATE = 4  # rough, model-agnostic heuristic
_SEVERITY_PRIORITY = {"critical": 0, "warning": 1, "good": 2}


def estimate_tokens(payload) -> int:
    """Rough token-count estimate for a JSON-serializable payload -- used
    only to decide whether compaction is worth applying, not for exact
    accounting or billing."""
    return len(json.dumps(payload)) // _CHARS_PER_TOKEN_ESTIMATE


def _truncate(text, max_chars: int):
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated for payload size]"


def _compact_findings(findings, max_findings: int) -> tuple[list, int]:
    """Truncate overlong finding text, then -- only if still too many --
    drop the lowest-severity findings first. Returns (findings, dropped_count)."""
    if not isinstance(findings, list):
        return findings, 0

    shortened = [
        {
            **f,
            "issue": _truncate(f.get("issue", ""), COMPACTION_MAX_FINDING_TEXT_CHARS),
            "recommendation": _truncate(f.get("recommendation", ""), COMPACTION_MAX_FINDING_TEXT_CHARS),
        }
        if isinstance(f, dict) else f
        for f in findings
    ]

    if len(shortened) <= max_findings:
        return shortened, 0

    # Stable-sort by severity priority (critical first), keeping original
    # relative order within the same severity, then keep only the top N.
    ranked = sorted(range(len(shortened)), key=lambda i: _SEVERITY_PRIORITY.get(shortened[i].get("severity") if isinstance(shortened[i], dict) else None, 3))
    kept_indices = sorted(ranked[:max_findings])
    kept = [shortened[i] for i in kept_indices]
    dropped = len(shortened) - len(kept)
    return kept, dropped


def compact_report(report: dict) -> dict:
    """Compact a report-shaped dict (has a "categories" list, each with a
    "findings" list) -- used for both a draft report and a previous draft
    re-embedded in critic feedback. Non-mutating: returns a new dict."""
    if not isinstance(report, dict) or not report.get("categories"):
        return report

    compacted = copy.deepcopy(report)
    for cat in compacted.get("categories", []):
        if not isinstance(cat, dict):
            continue
        findings, dropped = _compact_findings(cat.get("findings", []), COMPACTION_MAX_FINDINGS_PER_CATEGORY)
        cat["findings"] = findings
        if dropped:
            cat["_compaction_note"] = f"{dropped} additional lower-priority finding(s) omitted to fit request size."
    return compacted


def _compact_specialist_entry(entry: dict) -> dict:
    if not isinstance(entry, dict):
        return entry
    compacted = dict(entry)
    if "raw_evidence_notes" in compacted:
        compacted["raw_evidence_notes"] = _truncate(compacted["raw_evidence_notes"], COMPACTION_MAX_EVIDENCE_NOTES_CHARS)
    if "findings" in compacted:
        findings, dropped = _compact_findings(compacted["findings"], COMPACTION_MAX_FINDINGS_PER_CATEGORY)
        compacted["findings"] = findings
        if dropped:
            compacted["_compaction_note"] = f"{dropped} additional lower-priority finding(s) omitted to fit request size."
    return compacted


def compact_specialist_reports(specialist_reports: dict) -> dict:
    """Compact the specialist_reports dict passed to the synthesizer/critic.
    Handles the special "_critic_feedback" key (which embeds a full
    previous draft, not a normal specialist report) separately from
    ordinary per-specialist entries. Non-mutating: returns a new dict."""
    if not isinstance(specialist_reports, dict):
        return specialist_reports

    compacted = {}
    for key, value in specialist_reports.items():
        if key == "_critic_feedback" and isinstance(value, dict):
            feedback = dict(value)
            if "previous_draft" in feedback:
                feedback["previous_draft"] = compact_report(feedback["previous_draft"])
            if "instructions" in feedback:
                feedback["instructions"] = _truncate(feedback["instructions"], COMPACTION_MAX_EVIDENCE_NOTES_CHARS)
            compacted[key] = feedback
        else:
            compacted[key] = _compact_specialist_entry(value)
    return compacted


def maybe_compact_specialist_reports(specialist_reports: dict, threshold: int | None = None, log_fn=None) -> dict:
    """No-op (returns the same object, no copy) if the estimated payload
    size is already under threshold. Otherwise returns a compacted copy.
    This is the function synthesizer.py/critic.py actually call."""
    threshold = threshold if threshold is not None else COMPACTION_TOKEN_THRESHOLD
    before = estimate_tokens(specialist_reports)
    if before <= threshold:
        return specialist_reports

    compacted = compact_specialist_reports(specialist_reports)
    after = estimate_tokens(compacted)
    if log_fn:
        log_fn(f"  -> Proactively compacted specialist_reports payload: "
               f"~{before} -> ~{after} estimated tokens (threshold {threshold}).")
    return compacted


def maybe_compact_report(report: dict, threshold: int | None = None, log_fn=None) -> dict:
    """Same idea as maybe_compact_specialist_reports, for a standalone
    report-shaped dict (e.g. the critic's draft_report)."""
    threshold = threshold if threshold is not None else COMPACTION_TOKEN_THRESHOLD
    before = estimate_tokens(report)
    if before <= threshold:
        return report

    compacted = compact_report(report)
    after = estimate_tokens(compacted)
    if log_fn:
        log_fn(f"  -> Proactively compacted draft report payload: "
               f"~{before} -> ~{after} estimated tokens (threshold {threshold}).")
    return compacted