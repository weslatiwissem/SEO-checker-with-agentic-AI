"""Planner agent: the entry point of the pipeline. Decides which specialists
are worth running for this particular audit, rather than always blindly
running all of them (e.g. skip the competitive specialist if no competitor
context makes sense, or reprioritize security if the URL is on http://)."""
from __future__ import annotations

from .base_agent import ToolAgent
from .specialists import SPECIALIST_DEFINITIONS
from .config import PLANNER_MODEL, FALLBACK_MODEL

_SPECIALIST_LIST = "\n".join(
    f"- {key}: {d['display_name']}" for key, d in SPECIALIST_DEFINITIONS.items()
)

PLANNER_SYSTEM_PROMPT = f"""You are the planning agent for a multi-agent SEO audit system.
Given a target URL (and optionally a competitor URL, and prior audit history), decide which
specialist agents should run for this audit.

Available specialists:
{_SPECIALIST_LIST}

Normally run all of technical_seo, content, performance, security, and links -- they are cheap
and each covers a distinct category. Only include "competitive" if it would add real value
(e.g. a competitor URL was provided, or the site appears to be commercial/public-facing where
industry benchmarking is meaningful). Skip it for things like localhost/staging/internal URLs.

Respond with ONLY a JSON object (no prose, no markdown fences):
{{
  "specialists": [list of specialist keys to run],
  "reasoning": "1-2 sentence justification"
}}
"""


def run_planner(url: str, competitor_url: str | None, has_history: bool, model: str = PLANNER_MODEL,
                 fallback_model: str | None = FALLBACK_MODEL, key_index: int = 0, log_fn=None) -> dict:
    agent = ToolAgent(
        name="Planner",
        system_prompt=PLANNER_SYSTEM_PROMPT,
        model=model,
        fallback_model=fallback_model,
        starting_key_index=key_index,
        log_fn=log_fn,
    )
    context = f"Target URL: {url}"
    if competitor_url:
        context += f"\nCompetitor URL provided: {competitor_url}"
    if has_history:
        context += "\nNote: a previous audit of this domain exists in memory."
    return agent.run(context)