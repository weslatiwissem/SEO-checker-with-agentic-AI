# SEO Health Agent — Multi-Agent Edition (Groq-powered)

An agentic SEO & website-health auditing system. A planner decides what to
investigate, specialist agents each independently research one category
using real tool calls, a synthesizer merges their findings into one weighted
report, a critic agent reflects on that report and can send it back for
revision, and everything is persisted so later runs can reason about trends
over time.

Runs entirely on **Groq's free-tier API** (OpenAI-compatible, very fast,
open-weight models) — no paid model provider required. Hardened through
extensive real-world testing against live sites (Wikipedia, YouTube, Apple,
YouTube Music, and others) to survive both Groq's free-tier limits and the
kinds of mistakes LLMs make when asked to self-report facts and do arithmetic.

No frontend in this document — CLI + importable library. (A Next.js web UI
exists as a separate, in-progress add-on; see the `webapp/` project if you're
looking for that.)

## Architecture

```
                         ┌─────────────┐
                         │   Planner   │  decides which specialists to run
                         └──────┬──────┘
                                │
        ┌───────────┬──────────┼──────────┬───────────┐
        ▼           ▼          ▼           ▼           ▼
   Technical    Content    Performance  Security     Links       (Competitive*)
     SEO                    +Lighthouse                          *Groq Compound
   Specialist  Specialist  Specialist  Specialist   Specialist   (built-in search)
        │           │          │           │           │              │
        └───────────┴──────────┴─────┬─────┴───────────┴──────────────┘
                                      ▼
                          ┌────────────────────────┐
                          │  Deterministic cleanup  │  SSL fact-check, CWV fact-
                          │     (postprocess.py)    │  check, on-page/technical
                          │                         │  overlap removal
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
                                   Category recovery/cleanup, deterministic
                                   score/weight/trend reconciliation
                                                        │
                                                        ▼
                                                  Final Report
                                                        │
                                                        ▼
                                             SQLite memory (trend tracking)
```

## How this project evolved

This started as a single-model script and became a hardened multi-agent
pipeline through real, repeated testing on Groq's free tier — an excellent
stress test, since free-tier rate limits and a smaller fallback model exposed
failure modes a happy-path demo never would have. Nearly every part of the
current design exists because of a specific bug caught this way:

- **LLMs are unreliable at arithmetic.** The critic kept correctly flagging
  "overall score doesn't match the weighted average," but asking the model to
  fix its own math never reliably worked. Fix: recompute the score, grade,
  and category weights deterministically in Python
  (`orchestrator.py::_reconcile_overall_score`), with near-zero tolerance —
  even a 1-2 point discrepancy gets corrected, not just large ones.
- **LLMs restate facts they already computed correctly elsewhere, and get it
  wrong.** An SSL certificate's expiry, once computed, is a plain boolean —
  but a smaller model would still sometimes hallucinate "certificate
  expired" from a perfectly valid cert. Fix: compute `is_expired` and a
  plain-English `ssl_status_summary` in the tool itself, then
  deterministically overwrite any specialist finding that contradicts it
  (`postprocess.py::reconcile_ssl_findings`).
- **The same failure mode showed up for real performance data too.** After
  wiring in genuine Google Lighthouse data via the PageSpeed Insights API,
  specialists sometimes wrote vague findings ("page load time is higher than
  expected") instead of citing the actual measured numbers. Fix:
  `postprocess.py::reconcile_core_web_vitals` injects a canonical finding
  built from the real LCP/CLS/performance-score data if the model's own
  findings don't already cite it — and a companion fix corrects `data_
  limitations` text that goes stale (claiming "no real Core Web Vitals data"
  when it was, in fact, obtained that run).
- **Specialists don't share ground truth with each other.** The competitive
  specialist repeatedly re-judged the same title tag, meta description, or
  canonical tag another specialist already measured, sometimes flatly
  contradicting it (e.g. calling a 9-character title "well within the 50-60
  character recommendation," or claiming a canonical tag was missing when
  Technical SEO had just confirmed it existed). Prompting it not to didn't
  hold up reliably. Fix: deterministically strip any competitive finding
  that overlaps that territory
  (`postprocess.py::strip_competitive_onpage_overlap`).
- **The model's prose can contradict its own structured data — in two
  different ways.** A summary claiming "improved by 3 points" while the real
  computed trend was -3.3 (wrong direction), or "improved by 1.5 points"
  when the real delta was +5 (right direction, wrong magnitude) — and
  separately, a summary confidently comparing to a "previous audit" that
  never existed (the domain's first-ever run). Three distinct deterministic
  checks now catch these: `fix_summary_trend_mismatch` (direction and
  magnitude), and `fix_fabricated_trend_claim` (fabricated comparisons).
- **A category can end up empty, or worse, get real data silently
  discarded.** Sometimes a specialist genuinely fails (e.g. a 413 "request
  too large" error) and produces nothing — that's fine, and should be
  dropped rather than shown as a confusing blank report section. But it was
  also observed that a specialist could succeed, produce real findings, and
  have the *synthesizer* lose them during merging anyway. Fix:
  `_recover_or_drop_empty_categories` first tries to recover the original
  specialist's findings before giving up and dropping the category — and
  critically, this must run *before* score reconciliation, or a
  failed-but-scored-zero category unfairly drags down the whole report.
- **A site's own anti-bot protection can silently corrupt an entire audit.**
  A 403/429/503 response gets cached and reused by every specialist that
  shares that fetch — meaning "no headings," "no images," "missing meta
  description" findings can all just be symptoms of a block page, not real
  content problems. Fix: `fetch_page` flags likely-blocked responses, and
  `reconcile_likely_blocked` overrides a specialist's score/findings with a
  single honest "could not verify, likely blocked" finding instead of
  scoring the block page as if it were genuine content.
- **A rejected draft was still being shipped as final, silently — and via a
  wasted extra API call.** The critic could reject a report on its last
  allowed round, and the pipeline would still run one more, never-reviewed
  synthesis pass before giving up — occasionally producing a broken,
  degenerate draft that nothing ever checked. Fixed by stopping at the
  last-reviewed draft instead, and a `review_status` field now surfaces
  exactly which issues were never resolved rather than hiding them.
- **Free-tier quota exhaustion needed a real strategy, not just retries.**
  The system now: parses Groq's exact wait time across all three formats it
  uses (`"11s"`, `"6m53s"`, `"1h4m12s"`), prefers an instant model-fallback
  over waiting, proactively round-robins specialists *and* the
  synthesizer/critic's repeated calls across every configured API key
  (continuing one unbroken rotation sequence across the whole run, not
  resetting per stage), automatically shrinks oversized payloads on 413
  errors, and only fails fast when a wait is genuinely too long to be worth it.
- **Some "broken" results were actually false positives from looking like a
  bot.** A default `SEOHealthAgent/2.0` User-Agent got flagged and
  soft-blocked by some sites' WAFs, producing false "broken links" or
  "blocked page" results. Fixed by using a standard browser User-Agent, plus
  a statistical caution flag when an unusually high fraction of sampled
  links fail at once (likely blocking, not real breakage).
- **A stale OS certificate trust store produced a false "certificate
  expired" for a perfectly valid site.** `requests` (used for page fetches)
  bundles its own up-to-date root CA list via `certifi`; the raw `ssl`
  module used for the certificate check did not, and could diverge from it
  on some systems. Fixed by using the same `certifi` bundle for both.

The net effect: the LLM layer is treated as a fallible reasoning engine that
proposes findings, while anything that can be verified or computed outright
(arithmetic, certificate facts, real performance data, cross-category
consistency, bot-blocking detection) is enforced deterministically in code
around it. Where prompting alone wasn't enough — which was often — actual
guardrails were added instead of just asking more firmly.

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
   synthesizer (bounded to `SEO_AGENT_MAX_REFLECTION_ROUNDS`, default 2). On
   the last round, it stops at the last-reviewed draft rather than running
   one further, unreviewed revision. If still unresolved, the report is
   marked `review_status: "not_approved"` with the specific unresolved
   issues attached.
3. **Tool use / function calling** — every specialist decides which of its
   tools to call and in what order (fetch pages, parse HTML, check
   robots.txt/sitemap, verify SSL, inspect security headers, sample links,
   run a real Lighthouse audit).
4. **Real performance data via Google PageSpeed Insights** — the performance
   specialist calls a genuine Lighthouse audit (not a proxy signal): actual
   LCP, CLS, Total Blocking Time, Speed Index, a 0-100 performance score,
   and real-world Chrome User Experience Report field data when available.
   Automatically retries on Google's known-common transient 500 errors.
5. **Live web-search-augmented research** — the competitive/benchmarking
   specialist runs on Groq's **Compound** system (`groq/compound-mini`),
   which performs web search server-side automatically.
6. **Persistent long-term memory** — every completed audit is written to
   SQLite (`agent/memory.py`), enabling trend tracking across repeated runs.
7. **Schema-validated structured output** — every final report is validated
   against a Pydantic schema before being trusted downstream.
8. **A deterministic fact-checking layer** (`postprocess.py`) that catches
   an entire class of LLM self-reporting failures in code rather than
   prompting: SSL reconciliation, real Core Web Vitals injection, stale
   data-limitations correction, cross-specialist on-page/technical overlap
   removal, summary/trend contradiction and fabrication detection, bot-block
   detection, and empty-category recovery.
9. **Multi-key, multi-model resilience** — automatic fallback to a smaller
   model on rate/quota limits, proactive round-robin rotation across
   multiple configured API keys spanning the *entire* run (not just
   specialists), automatic payload-shrinking on request-too-large errors,
   and a JSON self-repair retry if a response comes back truncated.

## Setup

```bash
cd seo_agent
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your GROQ_API_KEY
```

Get a **free** API key at https://console.groq.com/keys. Optionally get a
free Google PageSpeed Insights key at
https://developers.google.com/speed/docs/insights/v5/get-started for a
higher rate limit on real Lighthouse audits (works without one too, just at
a lower rate limit).

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
    ├── tools.py              low-level tool implementations: fetch (with bot-block detection),
    │                         parse, ssl (certifi-based), headers, links (with bot-block
    │                         detection), and real Core Web Vitals via PageSpeed Insights
    │                         (with retry-on-transient-500)
    ├── tool_schemas.py        Groq/OpenAI-format tool-use schemas, grouped per specialist
    ├── base_agent.py          generic reusable ToolAgent -- the agentic loop runtime, including
    │                          Groq rate-limit/quota handling, model fallback, API-key rotation,
    │                          request-too-large payload shrinking, and JSON self-repair
    ├── specialists.py         specialist system prompts + tool assignments
    ├── planner.py             planning agent
    ├── synthesizer.py         synthesizer agent
    ├── critic.py              critic agent + reflection loop controller (stops at the last-
    │                          reviewed draft rather than running an unreviewed final revision)
    ├── postprocess.py         the deterministic fact-checking layer -- see "How this project
    │                          evolved" above for what each function catches and why
    ├── orchestrator.py        top-level pipeline: mode configs, canonical category names,
    │                          category recovery/cleanup, deterministic score/weight
    │                          reconciliation, wiring it all together
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
| `GROQ_API_KEYS` | — (optional) | Comma-separated list of multiple keys, each with its own daily quota. Specialists *and* the synthesizer/critic's repeated calls round-robin across them proactively, continuing one rotation sequence across the whole run |
| `GOOGLE_PAGESPEED_API_KEY` | — (optional) | Raises the rate limit on real Lighthouse audits via PageSpeed Insights; works without one at a lower limit |
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
  invisible to the HTML parser, the same blind spot classic crawlers have
  (though real Core Web Vitals now come from a genuine browser-based
  Lighthouse audit, which does render JS).
- Link checking samples a handful of links, not a full-site crawl. If a high
  fraction fail at once, the tool flags this as likely bot-blocking rather
  than confidently reporting a broken-links crisis — but this still needs
  manual spot-checking to be sure.
- Groq's free tier has real daily quota limits per model *and* per
  organization — multiple API keys only help if they're genuinely separate
  accounts/orgs, not just multiple keys within the same one.
- Google's PageSpeed Insights API occasionally has transient outages
  ("Unable to process request, please wait a while and try again") on
  heavier/complex sites; retried automatically, but can still fail on a
  particularly bad day for Google's servers.
- Even with all the deterministic guardrails in `postprocess.py`, the
  smaller fallback model can still produce lower-quality or occasionally
  inaccurate prose in findings/recommendations that aren't covered by an
  existing reconciliation rule. `--mode deep` avoids the fallback model
  entirely for the highest-confidence results.
- The system reports its known limitations in `data_limitations` on every run.

## Possible next steps

- Wire up `compaction.py` (see Project Layout) to proactively shrink
  synthesizer/critic payloads instead of only reacting to 413s after the fact.
- Wrap `run_full_audit()` in a FastAPI endpoint for a real backend/API (a
  first pass at this exists in the separate `webapp/` project).
- Add a `crawl` mode that runs the pipeline across multiple pages of a site
  and aggregates a site-wide score.
- Extend `postprocess.py`'s deterministic-reconciliation pattern to other
  recurring hallucination classes as they're discovered.
- Add an "auto-fix" agent that drafts corrected meta tags / alt text for
  low-effort findings.