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
from .postprocess import reconcile_ssl_findings, strip_competitive_onpage_overlap, fix_summary_trend_mismatch
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
    "competitive": "Competitive & Industry Benchmarking",
}


def _run_one_specialist(key: str, url: str, competitor_url: str | None, cfg: dict, key_index: int, log_fn) -> tuple[str, dict]:
    model = cfg["competitive"] if key == "competitive" else cfg["primary"]
    agent = build_specialist(key, log_fn=log_fn, model=model, fallback_model=cfg["fallback"], key_index=key_index)
    task = f"Target URL: {url}"
    if key == "competitive" and competitor_url:
        task += f"\nCompetitor URL to compare against: {competitor_url}"
    result = agent.run(task)
    result = reconcile_ssl_findings(result, agent.tool_call_log, log_fn=log_fn)
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
    if reported_score is None or abs(reported_score - computed_score) > 2:
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


def run_full_audit(
    url: str,
    competitor_url: str | None = None,
    use_memory: bool = True,
    mode: str = "auto",
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    log_fn = log_fn or (lambda msg: None)
    cfg = MODE_CONFIGS.get(mode, MODE_CONFIGS["auto"])
    if mode not in MODE_CONFIGS:
        log_fn(f"  -> Unknown mode '{mode}', defaulting to 'auto'.")

    previous_audit = memory.get_last_audit(url) if use_memory else None

    log_fn(f"Stage 1/4: Planning audit scope... (mode: {mode})")
    # Planner only runs once per audit, so it staying on key 0 has low impact
    # compared to synthesizer/critic, which run repeatedly in Stage 3.
    plan = run_planner(url, competitor_url, has_history=previous_audit is not None,
                        model=cfg["planner"], fallback_model=cfg["fallback"], log_fn=log_fn)
    specialist_keys = [k for k in plan.get("specialists", []) if k in SPECIALIST_DEFINITIONS]
    if not specialist_keys:
        specialist_keys = ["technical_seo", "content", "performance", "security", "links"]
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
            # first place for most runs.
            key_index = i % len(GROQ_API_KEYS) if GROQ_API_KEYS else 0
            futures[pool.submit(_run_one_specialist, key, url, competitor_url, cfg, key_index, log_fn)] = key

        for future in as_completed(futures):
            key = futures[future]
            try:
                _, result = future.result()
                specialist_reports[key] = result
                log_fn(f"  -> {key} specialist done (score: {result.get('score')})")
            except Exception as e:
                log_fn(f"  -> {key} specialist FAILED: {e}")
                specialist_reports[key] = {
                    "category": key,
                    "score": None,
                    "findings": [],
                    "raw_evidence_notes": f"Specialist failed to complete: {e}",
                }

    log_fn("Stage 3/4: Synthesizing + critiquing report (reflection loop)...")
    # Continue the same key rotation sequence right after the specialists,
    # rather than resetting back to key 0 -- synthesizer/critic run multiple
    # times in this stage and previously always hit whichever key specialist
    # #0 used, every single time.
    stage3_start_index = len(specialist_keys) % len(GROQ_API_KEYS) if GROQ_API_KEYS else 0
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

    _reconcile_overall_score(final_report, log_fn)

    was_approved = bool(reflection_log) and reflection_log[-1].get("review", {}).get("approved")
    if not was_approved:
        unresolved = reflection_log[-1].get("review", {}).get("issues", []) if reflection_log else []
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

    log_fn("Stage 4/4: Saving to persistent memory...")
    if use_memory:
        memory.save_audit(url, final_report)

    return final_report