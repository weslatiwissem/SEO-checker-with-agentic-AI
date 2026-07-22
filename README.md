# SEO Health Agent — Multi-Agent Edition (Groq-powered)

An agentic SEO & website-health auditing system. A planner decides what to
investigate, specialist agents each independently research one category
using real tool calls, a synthesizer merges their findings into one weighted
report, a critic agent reflects on that report and can send it back for
revision, and everything is persisted so later runs can reason about trends
over time.

Runs entirely on **Groq's free-tier API** (OpenAI-compatible, very fast,
open-weight models) — no paid model provider required. Built to survive
Groq's free-tier limits gracefully: automatic model fallback, multi-key
rotation, and a `--mode` flag to trade speed for quality on demand.

No frontend — CLI + importable library. Wrap it in an API later if you want.

## Architecture

```
                         ┌─────────────┐
                         │   Planner   │  decides which specialists to run
                         └──────┬──────┘
                                │
        ┌───────────┬──────────┼──────────┬───────────┐
        ▼           ▼          ▼           ▼           ▼
   Technical    Content    Performance  Security     Links       (Competitive*)
     SEO                                                          *Groq Compound
   Specialist  Specialist  Specialist  Specialist   Specialist   (built-in search)
        │           │          │           │           │              │
        └───────────┴──────────┴─────┬─────┴───────────┴──────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │  Deterministic cleanup  │  SSL fact-check, on-page
                          │     (postprocess.py)    │  overlap removal
                          └───────────┬─────────────┘
                                      ▼
                              ┌───────────────┐
                              │  Synthesizer  │  merges + weights + scores
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                    ┌────────►│    Critic     │  reflection / self-critique
                    │         └───────┬───────┘
                    │ revise if       ▼
                    │ not approved  approved?
                    └────────────── no ── yes ──► Draft Report
                                                        │
                                                        ▼
                                        Deterministic score/weight/trend
                                          reconciliation (orchestrator.py)
                                                        │
                                                        ▼
                                                  Final Report
                                                        │
                                                        ▼
                                             SQLite memory (trend tracking)
```

## How this project evolved

This started as a single-model script and became a hardened multi-agent
pipeline through real, repeated testing against live sites (Wikipedia,
YouTube, and others) on Groq's free tier — which turned out to be an
excellent stress test, since free-tier rate limits and a smaller fallback
model exposed failure modes a happy-path demo never would have. Most of the
system's current design exists specifically because of bugs caught this way:

- **LLMs are unreliable at arithmetic.** The critic kept correctly flagging
  "overall score doesn't match the weighted average," but asking the model
  to fix its own math never reliably worked. The fix: recompute the score,
  grade, and category weights deterministically in Python
  (`orchestrator.py::_reconcile_overall_score`) and simply overwrite the
  model's number, no matter how small the discrepancy.
- **LLMs restate facts they already computed correctly elsewhere, and get
  it wrong.** An SSL certificate's expiry date, once computed, is a plain
  boolean fact — but a smaller model would still sometimes hallucinate
  "certificate expired" from a perfectly valid cert. Fix: compute
  `is_expired` and a plain-English `ssl_status_summary` in the tool itself
  (`tools.py`), then deterministically reconcile any specialist finding that
  contradicts it (`postprocess.py::reconcile_ssl_findings`) rather than
  trusting the model to read the field correctly.
- **Specialists don't share ground truth with each other.** The competitive
  specialist repeatedly re-judged the same title tag / meta description the
  Content specialist already measured, and sometimes flatly contradicted it
  (e.g. calling a 9-character title "well within the 50-60 character
  recommendation"). Prompting it not to didn't hold up. Fix: deterministically
  strip any competitive finding that overlaps that territory
  (`postprocess.py::strip_competitive_onpage_overlap`).
- **The model's prose can contradict its own structured data.** A summary
  claiming "improved by 3 points" while the actual computed trend was -3.3.
  Fix: detect direction mismatches and append a correcting note
  (`postprocess.py::fix_summary_trend_mismatch`).
- **Free-tier quota exhaustion needed a real strategy, not just retries.**
  Early on, a single 429 could stall or crash a run. The current system:
  parses Groq's exact wait time (handling `"11s"`, `"6m53s"`, and
  `"1h4m12s"` formats), prefers an instant model-fallback over waiting,
  proactively round-robins specialists *and* the synthesizer/critic's
  repeated calls across every configured API key, and only fails fast (with
  a clear message) when a wait is genuinely too long to be worth it.
- **A rejected draft was still being shipped as final, silently.** The
  critic could reject a report twice and the system would just serve the
  last draft anyway with no indication. Now a `review_status` field and a
  visible warning surface exactly which issues were never resolved, and
  the pipeline no longer wastes an extra, unreviewed synthesis call after
  the last round.

The net effect: the LLM layer is now treated as a fallible reasoning engine
that proposes findings, while anything that can be verified or computed
outright (arithmetic, certificate facts, cross-category consistency) is
enforced deterministically in code around it. Where prompting alone wasn't
enough, actual guardrails were added instead.

## Agentic AI features

1. **Multi-agent orchestration** — a planner agent and up to six specialist
   agents (technical SEO, content, performance, security, link health,
   competitive/benchmarking), each with its own system prompt and restricted
   tool set, run **concurrently** via a thread pool and are reconciled by a
   synthesizer agent.
2. **Reflection / self-critique loop** — a critic agent reviews the
   synthesizer's draft against the raw specialist evidence, checking for
   hallucinated claims, miscalibrated scores, and internal inconsistency. If
   it isn't satisfied, it sends revision instructions back to the
   synthesizer (bounded to `SEO_AGENT_MAX_REFLECTION_ROUNDS`, default 2). If
   still unresolved after the last round, the report is marked
   `review_status: "not_approved"` with the specific unresolved issues
   attached, rather than silently presenting it as clean.
3. **Tool use / function calling** — every specialist decides which of its
   tools to call and in what order (fetch pages, parse HTML, check
   robots.txt/sitemap, verify SSL, inspect security headers, sample links).
4. **Live web-search-augmented research** — the competitive/benchmarking
   specialist runs on Groq's **Compound** system (`groq/compound-mini`),
   which performs web search server-side automatically, so it grounds
   "current best practice" claims in live results instead of frozen
   training data.
5. **Persistent long-term memory** — every completed audit is written to
   SQLite (`agent/memory.py`). Future audits of the same domain
   automatically pull the last result and the synthesizer weaves a trend
   observation into the report.
6. **Schema-validated structured output** — every final report is validated
   against a Pydantic schema before being trusted downstream (PDF export,
   CLI rendering, DB storage).
7. **Deterministic fact-checking of the LLM layer** (`postprocess.py`) — SSL
   expiry reconciliation, cross-specialist on-page overlap removal, and
   summary/trend contradiction detection, all applied in code rather than
   left to the model to get right.
8. **Multi-key, multi-model resilience** (`base_agent.py`, `config.py`) —
   automatic fallback to a smaller model on rate/quota limits, automatic
   rotation across multiple configured API keys, automatic payload-shrinking
   on request-too-large errors, and a JSON self-repair retry if a response
   comes back truncated.

## Setup

```bash
cd seo_agent
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your GROQ_API_KEY
```

Get a **free** API key at https://console.groq.com/keys

## Usage

```bash
# Full multi-agent audit (auto mode: strong model, falls back if needed)
python main.py audit https://example.com

# Fast, always-available mode -- skips the 70B model entirely
python main.py audit https://example.com --mode quick

# Best-quality mode -- only the strong model, fails clearly rather than
# silently downgrading if its quota is exhausted
python main.py audit https://example.com --mode deep

# Also benchmark against a competitor (feeds the competitive specialist)
python main.py audit https://example.com --competitor https://competitor.com

# Save JSON + a polished PDF report
python main.py audit https://example.com --out report.json --pdf report.pdf

# See score history for a domain (populated automatically after each audit)
python main.py history https://example.com

# Suppress the live agent activity log
python main.py audit https://example.com --quiet
```

Or use it as a library:

```python
from agent import run_full_audit

report = run_full_audit("https://example.com", competitor_url=None, mode="auto")
print(report["overall_score"], report["grade"], report["review_status"])
```

## Project layout

```
seo_agent/
├── main.py                  CLI entry point (audit / history subcommands, --mode flag)
├── requirements.txt
├── .env.example
├── README.md
└── agent/
    ├── __init__.py           exposes run_full_audit
    ├── config.py             model names, API keys, & tunables, all overridable via env vars
    ├── tools.py              low-level tool implementations (fetch, parse, ssl, headers, links)
    ├── tool_schemas.py        Groq/OpenAI-format tool-use schemas, grouped per specialist
    ├── base_agent.py          generic reusable ToolAgent -- the agentic loop runtime, including
    │                          Groq rate-limit/quota handling, model fallback, API-key rotation,
    │                          request-too-large payload shrinking, and JSON self-repair
    ├── specialists.py         specialist system prompts + tool assignments
    ├── planner.py             planning agent
    ├── synthesizer.py         synthesizer agent
    ├── critic.py              critic agent + reflection loop controller
    ├── postprocess.py         deterministic fact-checking: SSL reconciliation, competitive
    │                          on-page overlap removal, summary/trend mismatch correction
    ├── orchestrator.py        top-level pipeline: mode configs, canonical category names,
    │                          deterministic score/weight reconciliation, wiring it all together
    ├── memory.py              SQLite persistence + trend lookups
    ├── schemas.py             Pydantic validation of the final report contract
    ├── report_pdf.py          reportlab-based PDF export of a completed audit
    └── compaction.py          NOT YET WIRED IN -- built to proactively shrink synthesizer/critic
                               payloads (drop verbose fields, cap finding counts) before sending,
                               rather than reactively shrinking only after a 413 error. A good
                               next step if "request too large" retries are still frequent.
```

## Report shape

```json
{
  "url": "...",
  "overall_score": 78,
  "grade": "B",
  "summary": "...",
  "review_status": "approved",
  "unresolved_review_issues": [],
  "categories": [
    {
      "name": "Technical SEO",
      "score": 85,
      "weight": 0.25,
      "findings": [
        {"severity": "warning", "issue": "...", "recommendation": "..."}
      ]
    }
  ],
  "quick_wins": ["..."],
  "data_limitations": "...",
  "trend": {"previous_score": 70, "previous_timestamp": "...", "score_delta": 8}
}
```

`review_status` is `"not_approved"` (with `unresolved_review_issues` populated)
if the critic never signed off after the max reflection rounds — treat those
reports with extra scrutiny.

## Configuration

All overridable via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required unless `GROQ_API_KEYS` set) | API auth |
| `GROQ_API_KEYS` | — (optional) | Comma-separated list of multiple keys, each with its own daily quota. Specialists round-robin across them proactively; synthesizer/critic continue the same rotation sequence rather than resetting to key 0 |
| `SEO_AGENT_MODEL` | `llama-3.3-70b-versatile` | Primary model used by specialists, synthesizer |
| `SEO_AGENT_PLANNER_MODEL` | same as above | Model for the planner agent |
| `SEO_AGENT_CRITIC_MODEL` | same as above | Model for the critic agent |
| `SEO_AGENT_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Used automatically when the primary model hits a rate/quota limit (separate quota pool). Set to empty to disable fallback entirely |
| `SEO_AGENT_COMPETITIVE_MODEL` | `groq/compound-mini` | Groq's built-in web-search system, used only by the competitive specialist |
| `SEO_AGENT_MAX_ITER` | 10 | Max tool-call iterations per agent |
| `SEO_AGENT_MAX_REFLECTION_ROUNDS` | 2 | Max critic revision rounds |
| `SEO_AGENT_MAX_WORKERS` | 2 | Max concurrent specialist agents (kept low by default for free-tier TPM limits) |
| `SEO_AGENT_DISPATCH_STAGGER` | 2.0 | Seconds between dispatching each specialist, to avoid an instant burst of requests |
| `SEO_AGENT_RATE_LIMIT_RETRIES` | 4 | Max retry attempts on a rate-limited call before giving up |
| `SEO_AGENT_DB_PATH` | `./data/audit_history.db` | SQLite history location |

CLI-only: `--mode {quick,deep,auto}` (see Usage above).

## Honest limitations

- No JavaScript rendering — client-side-rendered content/SEO tags are
  invisible, the same blind spot classic crawlers have.
- Performance signals (response time, page weight, resource counts) are
  proxies, not a real Core Web Vitals / Lighthouse measurement.
- Link checking samples a handful of links, not a full-site crawl. If a high
  fraction fail at once, the tool flags this as likely bot-blocking rather
  than confidently reporting a broken-links crisis — but this still needs
  manual spot-checking to be sure.
- Groq's free tier has real daily quota limits per model *and* per
  organization — multiple API keys only help if they're genuinely separate
  accounts/orgs, not just multiple keys within the same one.
- Even with all the deterministic guardrails in `postprocess.py`, the
  smaller fallback model can still produce lower-quality or occasionally
  inaccurate prose in findings/recommendations that aren't covered by an
  existing reconciliation rule. `--mode deep` avoids the fallback model
  entirely for the highest-confidence results.
- The system reports its known limitations in `data_limitations` on every run.

## Possible next steps

- Wire up `compaction.py` (see Project Layout) to proactively shrink
  synthesizer/critic payloads instead of only reacting to 413s after the fact.
- Wrap `run_full_audit()` in a FastAPI endpoint for a real backend/API.
- Add a `crawl` mode that runs the pipeline across multiple pages of a site
  and aggregates a site-wide score.
- Extend `postprocess.py`'s deterministic-reconciliation pattern to other
  recurring hallucination classes as they're discovered (e.g. broken-link
  claims contradicting `check_links_status`'s actual results).
- Add an "auto-fix" agent that drafts corrected meta tags / alt text for
  low-effort findings.