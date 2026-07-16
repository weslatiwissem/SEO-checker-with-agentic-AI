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
from .config import MAX_PARALLEL_SPECIALISTS, SPECIALIST_DISPATCH_STAGGER_SECONDS


def _run_one_specialist(key: str, url: str, competitor_url: str | None, log_fn) -> tuple[str, dict]:
    agent = build_specialist(key, log_fn=log_fn)
    task = f"Target URL: {url}"
    if key == "competitive" and competitor_url:
        task += f"\nCompetitor URL to compare against: {competitor_url}"
    result = agent.run(task)
    return key, result


def run_full_audit(
    url: str,
    competitor_url: str | None = None,
    use_memory: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    log_fn = log_fn or (lambda msg: None)

    previous_audit = memory.get_last_audit(url) if use_memory else None

    log_fn("Stage 1/4: Planning audit scope...")
    plan = run_planner(url, competitor_url, has_history=previous_audit is not None, log_fn=log_fn)
    specialist_keys = [k for k in plan.get("specialists", []) if k in SPECIALIST_DEFINITIONS]
    if not specialist_keys:
        specialist_keys = ["technical_seo", "content", "performance", "security", "links"]
    log_fn(f"  -> Plan: {specialist_keys} ({plan.get('reasoning', '')})")

    log_fn(f"Stage 2/4: Dispatching {len(specialist_keys)} specialist agents "
           f"(max {MAX_PARALLEL_SPECIALISTS} concurrent, staggered)...")
    specialist_reports: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_SPECIALISTS, len(specialist_keys))) as pool:
        futures = {}
        for i, key in enumerate(specialist_keys):
            if i > 0:
                time.sleep(SPECIALIST_DISPATCH_STAGGER_SECONDS)
            futures[pool.submit(_run_one_specialist, key, url, competitor_url, log_fn)] = key

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
    draft, reflection_log = reflect_and_revise(url, specialist_reports, previous_audit, log_fn=log_fn)

    try:
        final_report = validate_report(draft)
    except ValidationError as e:
        log_fn(f"  -> WARNING: final report failed schema validation: {e}")
        final_report = draft  # surface the raw draft rather than crashing the whole run

    final_report["_specialist_reports"] = specialist_reports
    final_report["_reflection_log"] = reflection_log

    if previous_audit:
        final_report["trend"] = {
            "previous_score": previous_audit.get("overall_score"),
            "previous_timestamp": previous_audit.get("_timestamp"),
            "score_delta": round(final_report.get("overall_score", 0) - previous_audit.get("overall_score", 0), 1),
        }

    log_fn("Stage 4/4: Saving to persistent memory...")
    if use_memory:
        memory.save_audit(url, final_report)

    return final_report
