"""
Orchestrator: the top-level pipeline.

    Planner -> [Specialists run concurrently] -> Synthesizer -> Critic (reflection loop) -> Memory

This is the "multi-agent orchestration" piece: distinct agents, each with
narrow tool access and a narrow job, coordinated by a controller that
dispatches work in parallel and reconciles the results.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
import time

from . import memory
from .planner import run_planner
from .specialists import build_specialist, SPECIALIST_DEFINITIONS
from .critic import reflect_and_revise
from .schemas import validate_report, ValidationError
from .postprocess import (
    reconcile_ssl_findings, strip_competitive_onpage_overlap,
    fix_summary_trend_mismatch, fix_fabricated_trend_claim,
    reconcile_likely_blocked, reconcile_core_web_vitals,
    fix_stale_cwv_data_limitations, reconcile_accessibility_data,
    reconcile_best_practices_data,
)
from .config import (
    MAX_PARALLEL_SPECIALISTS, SPECIALIST_DISPATCH_STAGGER_SECONDS,
    DEFAULT_MODEL, FALLBACK_MODEL, COMPETITIVE_MODEL, PLANNER_MODEL, CRITIC_MODEL, GROQ_API_KEYS,
)

# Three ways to run an audit, trading speed/cost against quality:
# - "quick": everything runs on the small, fast, always-available fallback
#   model. No automatic model-switching needed since it's already the
#   cheapest tier. Fastest and least likely to hit any rate limit.
# - "deep": everything runs on the strong primary model, with automatic
#   fallback to the smaller model DISABLED -- if the primary's quota is
#   exhausted, agents wait or fail rather than silently using a weaker
#   model. Slowest but most consistent quality.
# - "auto" (default): current behavior -- try the primary model, silently
#   fall back to the smaller one if/when its quota runs out.
MODE_CONFIGS = {
    "quick": {"primary": FALLBACK_MODEL, "planner": FALLBACK_MODEL, "critic": FALLBACK_MODEL,
              "competitive": FALLBACK_MODEL, "fallback": None},
    "deep": {"primary": DEFAULT_MODEL, "planner": PLANNER_MODEL, "critic": CRITIC_MODEL,
             "competitive": COMPETITIVE_MODEL, "fallback": None},
    "auto": {"primary": DEFAULT_MODEL, "planner": PLANNER_MODEL, "critic": CRITIC_MODEL,
             "competitive": COMPETITIVE_MODEL, "fallback": FALLBACK_MODEL},
}


# Fixed, canonical category names -- the model's own "category" field in its
# JSON output is overridden with these rather than trusted, since the
# competitive specialist in particular renames itself differently almost
# every run ("SEO Technical Health", "SEO Health Assessment", "SEO Audit"...),
# which is a major source of the critic's repeated "category doesn't match
# between specialist report and draft" complaints.
CANONICAL_CATEGORY_NAMES = {
    "technical_seo": "Technical SEO",
    "content": "On-Page Content",
    "performance": "Page Speed",
    "security": "Web Security",
    "links": "Link Health",
    "accessibility": "Accessibility",
    "best_practices": "Best Practices",
    "competitive": "Competitive & Industry Benchmarking",
}


def _run_one_specialist(key: str, url: str, competitor_url: str | None, cfg: dict, key_index: int, log_fn) -> tuple[str, dict]:
    model = cfg["competitive"] if key == "competitive" else cfg["primary"]
    agent = build_specialist(key, log_fn=log_fn, model=model, fallback_model=cfg["fallback"], key_index=key_index)
    task = f"Target URL: {url}"
    if key == "competitive" and competitor_url:
        task += f"\nCompetitor URL to compare against: {competitor_url}"
    result = agent.run(task)
    result = reconcile_likely_blocked(result, agent.tool_call_log, log_fn=log_fn)
    result = reconcile_ssl_findings(result, agent.tool_call_log, log_fn=log_fn)
    result = reconcile_core_web_vitals(result, agent.tool_call_log, log_fn=log_fn)  # <-- new
    result = reconcile_accessibility_data(result, agent.tool_call_log, log_fn=log_fn)
    result = reconcile_best_practices_data(result, agent.tool_call_log, log_fn=log_fn)
    if key == "competitive":
        result = strip_competitive_onpage_overlap(result, log_fn=log_fn)
    result["category"] = CANONICAL_CATEGORY_NAMES.get(key, result.get("category", key))
    return key, result


def _reconcile_overall_score(report: dict, log_fn) -> None:
    """LLMs (especially smaller ones) are unreliable at weighted-average
    arithmetic -- the critic repeatedly catches "overall_score doesn't match
    the weighted average" but that alone doesn't fix it. Recompute it
    deterministically here rather than trusting the model's own math.
    Categories with a null/missing score (e.g. a specialist that failed
    outright) are excluded rather than treated as 0, and their weight is
    excluded from the total so they don't silently drag the score down."""
    categories = report.get("categories") or []
    if not categories:
        return

    usable = [
        c for c in categories
        if isinstance(c.get("score"), (int, float)) and isinstance(c.get("weight"), (int, float))
    ]
    if not usable:
        return

    total_weight = sum(c["weight"] for c in usable)
    if total_weight <= 0:
        return

    weighted_sum = sum(c["score"] * c["weight"] for c in usable)
    computed_score = round(weighted_sum / total_weight, 1)

    reported_score = report.get("overall_score")
    if reported_score is None or abs(reported_score - computed_score) > 0.05:
        log_fn(f"  -> Correcting overall_score: model said {reported_score}, "
               f"actual weighted average is {computed_score}")
        report["overall_score"] = computed_score
        report["grade"] = (
            "A" if computed_score >= 90 else
            "B" if computed_score >= 80 else
            "C" if computed_score >= 70 else
            "D" if computed_score >= 60 else "F"
        )

    # The score above is already correct regardless of whether weights sum to
    # 1.0 (we divide by the actual total), but the displayed per-category
    # weights are still misleading to a reader if they don't sum to 1.0 --
    # the critic flags this almost every run and it's never actually fixed
    # upstream, so normalize it here deterministically.
    if abs(total_weight - 1.0) > 0.02:
        log_fn(f"  -> Normalizing category weights: they summed to {round(total_weight, 3)}, not 1.0")
        for c in usable:
            c["weight"] = round(c["weight"] / total_weight, 3)

def _recover_or_drop_empty_categories(report: dict, specialist_reports: dict, log_fn) -> None:
    """A category with zero findings in the synthesized draft doesn't always
    mean the specialist itself failed -- the synthesizer can silently lose a
    specialist's real findings while merging (observed: Link Health's
    specialist successfully checked links and returned findings, but the
    draft category ended up empty anyway). Try to recover the specialist's
    original findings first; only drop the category if there's genuinely
    nothing to show (e.g. the specialist itself failed, like a 413 error).
    Must run BEFORE _reconcile_overall_score so a truly-empty, dropped
    category never counts toward the weighted average.

    Also validates/recovers `score`, independently of findings. Observed in
    the wild: a category can have non-empty findings (sometimes fabricated
    by the synthesizer from a failed specialist's raw error text -- see the
    raw_evidence_notes sanitization above) while its score is None/invalid,
    which fails schema validation and would otherwise ship a category
    showing a blank/"None" score in the final report. If the specialist's
    own score is a valid number, recover that; if the specialist itself has
    no valid score either (it genuinely failed), the whole category is
    dropped even if it has findings, since those findings aren't backed by
    anything trustworthy."""
    categories = report.get("categories") or []
    reverse_canonical = {v: k for k, v in CANONICAL_CATEGORY_NAMES.items()}
    kept = []
    for c in categories:
        has_findings = bool(c.get("findings"))
        has_valid_score = isinstance(c.get("score"), (int, float))

        spec_key = reverse_canonical.get(c.get("name"))
        source = specialist_reports.get(spec_key) if spec_key else None
        source_has_findings = bool(source and source.get("findings"))
        source_has_valid_score = bool(source and isinstance(source.get("score"), (int, float)))

        if not has_findings and source_has_findings:
            log_fn(
                f"  -> '{c.get('name')}' had no findings in the synthesized draft but the "
                f"original specialist report did -- recovering its real findings instead of dropping."
            )
            c["findings"] = source["findings"]
            has_findings = True

        if not has_valid_score and source_has_valid_score:
            log_fn(
                f"  -> '{c.get('name')}' had an invalid score ({c.get('score')!r}) in the "
                f"synthesized draft -- recovering the specialist's real score ({source['score']})."
            )
            c["score"] = source["score"]
            has_valid_score = True

        if has_findings and has_valid_score:
            kept.append(c)
        else:
            log_fn(f"  -> Dropping incomplete category '{c.get('name', '?')}' -- no valid "
                   f"findings+score available in the draft or the original specialist report "
                   f"(the specialist itself likely failed).")

    report["categories"] = kept


def _drop_issues_resolved_by_reconciliation(issues: list[str]) -> list[str]:
    """The critic's review runs BEFORE _reconcile_overall_score, so one
    specific class of its complaint -- category weights not summing to
    ~1.0 -- is unconditionally fixed by that deterministic normalization
    step on every single run (see _reconcile_overall_score above).
    Surfacing that complaint as still-"unresolved" in the final report is
    stale and actively misleading: the report the user is looking at has
    already had its weights normalized to sum to 1.0 by the time they read
    this list. Drop only that narrow, mechanically-guaranteed-resolved
    class of complaint; leave substantive judgment calls (e.g. "the score
    still seems too high given the findings") alone, since those aren't
    something a deterministic step fixes."""
    kept = []
    for issue in issues:
        lowered = issue.lower()
        if "weight" in lowered and "sum" in lowered:
            continue
        kept.append(issue)
    return kept


def run_full_audit(
    url: str,
    competitor_url: str | None = None,
    use_memory: bool = True,
    mode: str = "auto",
    log_fn: Callable[[str], None] | None = None,
    starting_key_index: int = 0,
) -> dict:
    log_fn = log_fn or (lambda msg: None)
    cfg = MODE_CONFIGS.get(mode, MODE_CONFIGS["auto"])
    if mode not in MODE_CONFIGS:
        log_fn(f"  -> Unknown mode '{mode}', defaulting to 'auto'.")

    previous_audit = memory.get_last_audit(url) if use_memory else None

    log_fn(f"Stage 1/4: Planning audit scope... (mode: {mode})")
    # starting_key_index lets a caller running many audits back-to-back in
    # one process (e.g. the eval harness) start each audit's key rotation
    # on a different key -- without this, every single audit always began
    # on key 0 regardless of how many prior audits already ran in this
    # process, so multiple configured keys never actually got used across
    # a sequence of audits, only within one.
    plan = run_planner(url, competitor_url, has_history=previous_audit is not None,
                        model=cfg["planner"], fallback_model=cfg["fallback"],
                        key_index=starting_key_index, log_fn=log_fn)
    specialist_keys = [k for k in plan.get("specialists", []) if k in SPECIALIST_DEFINITIONS]
    if not specialist_keys:
        specialist_keys = ["technical_seo", "content", "performance", "security", "links", "accessibility", "best_practices"]
    log_fn(f"  -> Plan: {specialist_keys} ({plan.get('reasoning', '')})")

    log_fn(f"Stage 2/4: Dispatching {len(specialist_keys)} specialist agents "
           f"(max {MAX_PARALLEL_SPECIALISTS} concurrent, staggered"
           + (f", spread across {len(GROQ_API_KEYS)} API keys)..." if len(GROQ_API_KEYS) > 1 else ")..."))
    specialist_reports: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_SPECIALISTS, len(specialist_keys))) as pool:
        futures = {}
        for i, key in enumerate(specialist_keys):
            if i > 0:
                time.sleep(SPECIALIST_DISPATCH_STAGGER_SECONDS)
            # Proactively spread specialists across available keys (round-robin)
            # rather than only reactively rotating after one key's exhausted --
            # with multiple keys this avoids ever hitting the limit in the
            # first place for most runs. Offset by starting_key_index so a
            # caller running several audits in sequence doesn't have every
            # single one begin on the same key.
            key_index = (i + starting_key_index) % len(GROQ_API_KEYS) if GROQ_API_KEYS else 0
            futures[pool.submit(_run_one_specialist, key, url, competitor_url, cfg, key_index, log_fn)] = key

        for future in as_completed(futures):
            key = futures[future]
            try:
                _, result = future.result()
                specialist_reports[key] = result
                log_fn(f"  -> {key} specialist done (score: {result.get('score')})")
            except Exception as e:
                log_fn(f"  -> {key} specialist FAILED: {e}")
                # Deliberately NOT including str(e) verbatim here. When a
                # specialist fails because the model's JSON was truncated/
                # invalid, the raised error message embeds that raw, never-
                # successfully-parsed text (it's useful for a human reading
                # the log). Observed in the wild: passing that same text
                # into raw_evidence_notes let the synthesizer AND critic
                # both treat quoted "findings" inside the failed attempt as
                # if they were real, validated specialist data -- the
                # critic cited specific findings/scores that existed only
                # inside a JSON blob that was never actually parsed. A
                # short, explicitly-non-data message avoids handing the
                # next LLM stage something structured-looking to
                # over-trust.
                specialist_reports[key] = {
                    "category": key,
                    "score": None,
                    "findings": [],
                    "raw_evidence_notes": (
                        "This specialist failed to complete due to a technical/formatting "
                        "error. No reliable findings are available for this category -- do "
                        "not infer, reconstruct, or cite any specific finding, score, or "
                        "number for it."
                    ),
                }
    had_real_cwv = any(r.get("_real_cwv_available") for r in specialist_reports.values())
    
    log_fn("Stage 3/4: Synthesizing + critiquing report (reflection loop)...")
    # Continue the same key rotation sequence right after the specialists,
    # rather than resetting back to key 0 -- synthesizer/critic run multiple
    # times in this stage and previously always hit whichever key specialist
    # #0 used, every single time.
    stage3_start_index = (len(specialist_keys) + starting_key_index) % len(GROQ_API_KEYS) if GROQ_API_KEYS else 0
    draft, reflection_log = reflect_and_revise(
        url, specialist_reports, previous_audit,
        synthesizer_model=cfg["primary"], critic_model=cfg["critic"],
        fallback_model=cfg["fallback"], starting_key_index=stage3_start_index, log_fn=log_fn,
    )

    try:
        final_report = validate_report(draft)
    except ValidationError as e:
        log_fn(f"  -> WARNING: final report failed schema validation: {e}")
        final_report = draft  # surface the raw draft rather than crashing the whole run

    _recover_or_drop_empty_categories(final_report, specialist_reports, log_fn)   # <-- was _drop_empty_categories(final_report, log_fn)
    _reconcile_overall_score(final_report, log_fn)


    was_approved = bool(reflection_log) and reflection_log[-1].get("review", {}).get("approved")
    if not was_approved:
        unresolved = reflection_log[-1].get("review", {}).get("issues", []) if reflection_log else []
        unresolved = _drop_issues_resolved_by_reconciliation(unresolved)
        final_report["review_status"] = "not_approved"
        final_report["unresolved_review_issues"] = unresolved
        log_fn(f"  -> WARNING: report was NOT approved by the critic after "
               f"{len(reflection_log)} round(s); treat findings with extra scrutiny.")
    else:
        final_report["review_status"] = "approved"

    final_report["_specialist_reports"] = specialist_reports
    final_report["_reflection_log"] = reflection_log

    if previous_audit:
        final_report["trend"] = {
            "previous_score": previous_audit.get("overall_score"),
            "previous_timestamp": previous_audit.get("_timestamp"),
            "score_delta": round(final_report.get("overall_score", 0) - previous_audit.get("overall_score", 0), 1),
        }
        fix_summary_trend_mismatch(final_report, log_fn)
    else:
        fix_fabricated_trend_claim(final_report, log_fn)

    fix_stale_cwv_data_limitations(final_report, had_real_cwv, log_fn)

    log_fn("Stage 4/4: Saving to persistent memory...")
    if use_memory:
        memory.save_audit(url, final_report)

    return final_report