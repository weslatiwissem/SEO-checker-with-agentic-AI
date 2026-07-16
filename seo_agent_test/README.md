# SEO Health Agent — Multi-Agent Edition (Groq-powered)

An agentic SEO & website-health auditing system. A planner decides what to
investigate, specialist agents each independently research one category
using real tool calls, a synthesizer merges their findings into one weighted
report, a critic agent reflects on that report and can send it back for
revision, and everything is persisted so later runs can reason about trends
over time.

Runs entirely on **Groq's free-tier API** (OpenAI-compatible, very fast,
open-weight models) — no paid model provider required.

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
                              ┌───────────────┐
                              │  Synthesizer  │  merges + weights + scores
                              └───────┬───────┘
                                      ▼
                              ┌───────────────┐
                    ┌────────►│    Critic     │  reflection / self-critique
                    │         └───────┬───────┘
                    │ revise if       ▼
                    │ not approved  approved?
                    └────────────── no ── yes ──► Final Report
                                                        │
                                                        ▼
                                             SQLite memory (trend tracking)
```

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
   synthesizer (bounded to `SEO_AGENT_MAX_REFLECTION_ROUNDS`, default 2).
3. **Tool use / function calling** — every specialist decides which of its
   tools to call and in what order (fetch pages, parse HTML, check
   robots.txt/sitemap, verify SSL, inspect security headers, sample links).
4. **Live web-search-augmented research** — the competitive/benchmarking
   specialist runs on Groq's **Compound** system (`groq/compound-mini`),
   which performs web search server-side automatically, so it grounds
   "current best practice" claims in live results instead of frozen
   training data. (Compound doesn't support mixing in custom tools, so this
   one specialist has no client-side tool list — it relies entirely on its
   built-in search.)
5. **Persistent long-term memory** — every completed audit is written to
   SQLite (`agent/memory.py`). Future audits of the same domain
   automatically pull the last result and the synthesizer weaves a trend
   observation into the report.
6. **Schema-validated structured output** — every final report is validated
   against a Pydantic schema before being trusted downstream (PDF export,
   CLI rendering, DB storage).

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
# Full multi-agent audit
python main.py audit https://example.com

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

report = run_full_audit("https://example.com", competitor_url=None)
print(report["overall_score"], report["grade"])
```

## Project layout

```
seo_agent/
├── main.py                  CLI entry point (audit / history subcommands)
├── requirements.txt
├── .env.example
├── README.md
└── agent/
    ├── __init__.py          exposes run_full_audit
    ├── config.py            model names & tunables, all overridable via env vars
    ├── tools.py             low-level tool implementations (fetch, parse, ssl, etc.)
    ├── tool_schemas.py       Groq/OpenAI-format tool-use schemas, grouped per specialist
    ├── base_agent.py        generic reusable ToolAgent (the agentic loop runtime, on Groq)
    ├── specialists.py        specialist system prompts + tool assignments
    ├── planner.py            planning agent
    ├── synthesizer.py        synthesizer agent
    ├── critic.py             critic agent + reflection loop controller
    ├── orchestrator.py        top-level pipeline wiring it all together
    ├── memory.py             SQLite persistence + trend lookups
    ├── schemas.py             Pydantic validation of the final report contract
    └── report_pdf.py          reportlab-based PDF export of a completed audit
```

## Report shape

```json
{
  "url": "...",
  "overall_score": 78,
  "grade": "B",
  "summary": "...",
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

## Configuration

All overridable via environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — (required) | API auth |
| `SEO_AGENT_MODEL` | `llama-3.3-70b-versatile` | Model used by specialists, planner, synthesizer, critic |
| `SEO_AGENT_PLANNER_MODEL` | same as above | Model for the planner agent |
| `SEO_AGENT_CRITIC_MODEL` | same as above | Model for the critic agent |
| `SEO_AGENT_COMPETITIVE_MODEL` | `groq/compound-mini` | Groq's built-in web-search system, used only by the competitive specialist |
| `SEO_AGENT_MAX_ITER` | 10 | Max tool-call iterations per agent |
| `SEO_AGENT_MAX_REFLECTION_ROUNDS` | 2 | Max critic revision rounds |
| `SEO_AGENT_MAX_WORKERS` | 5 | Max concurrent specialist agents |
| `SEO_AGENT_DB_PATH` | `./data/audit_history.db` | SQLite history location |

## Honest limitations

- No JavaScript rendering — client-side-rendered content/SEO tags are
  invisible, the same blind spot classic crawlers have.
- Performance signals (response time, page weight, resource counts) are
  proxies, not a real Core Web Vitals / Lighthouse measurement.
- Link checking samples a handful of links, not a full-site crawl.
- Groq's free tier has rate limits; a multi-agent audit fires several model
  calls per run, so heavy concurrent usage may need the paid tier.
- The system reports these limitations itself in `data_limitations` on every run.

## Possible next steps

- Wrap `run_full_audit()` in a FastAPI endpoint for a real backend/API.
- Add a `crawl` mode that runs the pipeline across multiple pages of a site
  and aggregates a site-wide score.
- Cache tool results per domain to avoid re-fetching within a short window.
- Add an "auto-fix" agent that drafts corrected meta tags / alt text for
  low-effort findings.
