"""
Critic agent: implements a reflection loop. Rather than trusting the
synthesizer's first draft, a separate critical pass checks it for
hallucinated claims, miscalibrated scores, and internal inconsistency, and
either approves it or sends it back with concrete revision instructions.
This catches errors a single-pass pipeline would silently ship.
"""
from __future__ import annotations

import json

from .base_agent import ToolAgent
from .synthesizer import run_synthesizer
from .config import CRITIC_MODEL, MAX_REFLECTION_ROUNDS

CRITIC_SYSTEM_PROMPT = """You are the critic agent in a multi-agent SEO audit system. You review
a draft report produced by a synthesizer agent against the raw specialist findings it was built
from. Check for:
- Findings or numbers in the draft that are NOT supported by the underlying specialist reports
  (hallucination)
- Scores that seem miscalibrated given the findings (e.g. a "critical" finding present but the
  category score is still 95)
- Internal inconsistency (e.g. weights don't sum to ~1.0, overall_score doesn't roughly match
  the weighted average of category scores)
- Missing categories that a specialist actually reported on
- Vague, non-actionable recommendations

Respond with ONLY a JSON object (no prose, no markdown fences):
{
  "approved": boolean,
  "issues": [string],   // empty list if approved
  "instructions_for_revision": string  // empty string if approved; otherwise concrete, specific fixes
}
Be a genuinely skeptical reviewer -- approving a flawed report defeats the point of this step.
"""


def critique(draft: dict, specialist_reports: dict, log_fn=None) -> dict:
    agent = ToolAgent(name="Critic", system_prompt=CRITIC_SYSTEM_PROMPT, model=CRITIC_MODEL, log_fn=log_fn)
    payload = {"draft_report": draft, "specialist_reports": specialist_reports}
    return agent.run(json.dumps(payload, indent=2))


def reflect_and_revise(
    url: str,
    specialist_reports: dict[str, dict],
    previous_audit: dict | None,
    log_fn=None,
) -> tuple[dict, list[dict]]:
    """Run synthesizer -> critic -> (revise if needed) up to MAX_REFLECTION_ROUNDS times.
    Returns (final_report, reflection_log)."""
    reflection_log = []

    draft = run_synthesizer(url, specialist_reports, previous_audit, log_fn=log_fn)

    for round_num in range(1, MAX_REFLECTION_ROUNDS + 1):
        review = critique(draft, specialist_reports, log_fn=log_fn)
        reflection_log.append({"round": round_num, "review": review})

        if review.get("approved"):
            break

        if log_fn:
            log_fn(f"[Critic] round {round_num}: revision requested -- {review.get('issues')}")

        # Feed critic's instructions back into a re-run of the synthesizer
        revised_reports = dict(specialist_reports)
        revised_reports["_critic_feedback"] = {
            "previous_draft": draft,
            "issues": review.get("issues"),
            "instructions": review.get("instructions_for_revision"),
        }
        draft = run_synthesizer(url, revised_reports, previous_audit, log_fn=log_fn)
    else:
        if log_fn:
            log_fn("[Critic] max reflection rounds reached; proceeding with latest draft.")

    return draft, reflection_log
