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
Samsung, Internet Archive's blog, YouTube Music, and others) to survive both
Groq's free-tier limits and the kinds of mistakes LLMs make when asked to
self-report facts and do arithmetic. Backed by a 241-test automated `pytest`
suite and a self-grading eval harness that runs the real pipeline against
known-issue benchmark sites (see Testing / Eval harness below).

No frontend in this document — CLI + importable library. (A Next.js web UI
exists as a separate, in-progress add-on; see the `webapp/` project if you're
looking for that.)

## Architecture

```
                         ┌─────────────┐
                         │   Planner   │  decides which specialists to run
                         └──────┬──────┘
                                │
        ┌───────────────────────┴────────────────────────┐
        │       Up to 8 specialists run concurrently      │
        │  Technical SEO · Content · Performance+LH ·     │
        │  Security · Links · Accessibility ·             │
        │  Best Practices · Competitive*                  │
        │               (*Groq Compound, built-in search) │
        └───────────────────────┬────────────────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │  Deterministic cleanup  │  SSL / CWV / accessibility /
                     │     (postprocess.py)    │  best-practices fact-check,
                     │                         │  on-page overlap removal
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
                       score/weight/trend reconciliation, stale-issue
                                        filtering
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
- **The exact same "vague finding instead of citing real data" bug showed
  up again for accessibility and best-practices data — and the first fix
  for it wasn't robust enough, twice.** After wiring in real Lighthouse
  accessibility/best-practices audits, the "did the model already cite the
  real data" check first used exact-substring matching, which produced
  false "not covered" results (and duplicate injected findings) on
  perfectly good writing: plurals ("child" vs. the audit's "children"),
  word order ("list items" vs. the audit id `listitem`), and simple
  omission of the literal score digit. Fixed with a stemmed,
  majority-keyword-overlap matcher (`postprocess.py::keyword_topic_covered`)
  shared by both categories instead of two independent copies — reusing one
  fixed implementation rather than risking the same bug drifting back in
  twice. A related false positive also showed up during testing: a
  coincidental digit match (a score of 50 "matching" the unrelated phrase
  "30-50% of real WCAG issues" elsewhere in the text) wrongly suppressed a
  real injection — fixed with a boundary-aware regex that won't match a
  number embedded inside another number or a percentage.
- **A resolved complaint can still get shown to the user as "unresolved."**
  The critic's review runs *before* `_reconcile_overall_score`'s automatic
  weight normalization — so "category weights don't sum to 1.0" was a
  complaint the critic could raise on its last round, get silently fixed by
  the very next step, and then still show up in the final printed
  `unresolved_review_issues`, telling the user to distrust something that
  was already correct in the report they were looking at. Fixed by
  `orchestrator.py::_drop_issues_resolved_by_reconciliation`, which strips
  only that specific, mechanically-guaranteed-resolved complaint class —
  substantive judgment calls (e.g. "the score still seems too high") are
  left alone, since those aren't something a deterministic step fixes.
- **The reactive 413-shrinking mechanism was recovering fine, but wastefully.**
  Real runs occasionally hit a "request too large" error on a specialist's
  payload — the existing reactive shrink-and-retry logic always recovered,
  but each occurrence cost a failed API call and retry latency mid-run.
  `agent/compaction.py` now estimates the synthesizer/critic payload size
  *before* the first send and, only above a threshold, trims lowest-priority
  findings and overlong free-text fields first — so the initial request has
  a real shot at succeeding instead of needing a round-trip failure to find
  out it was too big.

- **A finding could say "OK" and "Failing checks include: X" in the same
  sentence.** Lighthouse can genuinely report a perfect 100/100 category
  score while still flagging one zero-weight/informational audit as
  "failing" (e.g. a missing JS source map that doesn't count toward the
  score) — both facts are accurate, but the reconciliation layer's own
  message-building logic labeled that combination "good" purely from the
  score, producing self-contradictory text. The critic actually caught this
  exact case on a real run ("a perfect score despite the presence of
  issues...") but that complaint didn't survive to the final report. Fixed
  by never labeling a finding "good" while it's also naming a failing
  check, regardless of score.
- **Every audit always started its API-key rotation from the same key,**
  even when several were run back-to-back in one process (as the eval
  harness does) — so configuring more `GROQ_API_KEYS` didn't actually
  spread a *sequence* of audits across them, only the specialists *within*
  one audit. `run_full_audit()` now accepts a `starting_key_index`, and the
  eval harness passes a different one per sampled case (round-robin) — so
  more configured keys now directly buys headroom to sample more benchmark
  sites per run without concentrating load on a single key.
- **A failed specialist's raw, never-successfully-parsed JSON was leaking
  into the synthesizer/critic as if it were real data.** When a specialist's
  JSON repair attempt still failed, the resulting error message embeds the
  model's raw, truncated JSON text (useful for a human reading the log) —
  but that same text was also getting copied verbatim into the failure
  placeholder's `raw_evidence_notes`. Both the synthesizer and the critic
  then treated quoted "findings" inside that never-parsed blob as if they
  were validated specialist data: on one real run, the critic cited a
  specific score and finding count for a category whose specialist had
  *completely failed*, because that exact text happened to appear inside
  the failure message. Fixed by replacing the raw error text with a short,
  explicitly-non-data message when building the failure placeholder.
- **A category could ship with real-looking findings but a `null` score.**
  Downstream of the bug above: a category built partly from a failed
  specialist's leaked text could have findings but no valid score, which
  failed Pydantic schema validation and would otherwise show a category
  with a blank score in the final report.
  `orchestrator.py::_recover_or_drop_empty_categories` previously only
  checked for *empty findings*; it now also validates `score`
  independently, recovering the specialist's real score when available and
  dropping the whole category (even if it has findings) when the specialist
  itself never provided a valid score either.
- **A benchmark case's own premise turned out to be wrong.** The
  `http-only-no-tls` eval harness case assumed "this site never redirects
  http:// to https://" meant "has no valid SSL/TLS available" — but a real
  run showed `check_ssl_certificate` completing a genuinely valid TLS
  handshake against the domain, an accurate tool result that simply didn't
  match the site's actual (different, more specific) behavior. Removed
  rather than left in as a permanent false-negative generator — see the
  comment above `BENCHMARK_CASES` in `eval_harness.py`.

These bugs (and the compaction/key-rotation improvements) were all found by
actually running the pipeline against real live sites (blog.archive.org,
samsung.com, apple.com, pathe.tn, and the eval harness's own benchmark
sites) after each change and reading the output closely — not just by
adding a feature and assuming a clean first run meant it worked.

The net effect: the LLM layer is treated as a fallible reasoning engine that
proposes findings, while anything that can be verified or computed outright
(arithmetic, certificate facts, real performance data, cross-category
consistency, bot-blocking detection) is enforced deterministically in code
around it. Where prompting alone wasn't enough — which was often — actual
guardrails were added instead of just asking more firmly.

## Agentic AI features

1. **Multi-agent orchestration** — a planner agent and up to seven specialist
   agents (technical SEO, content, performance, security, link health,
   accessibility, best practices, plus competitive/benchmarking when
   relevant), each with its own system prompt and restricted tool set, run
   **concurrently** via a thread pool and are reconciled by a synthesizer
   agent.
2. **Reflection / self-critique loop** — a critic agent reviews the
   synthesizer's draft against the raw specialist evidence, checking for
   hallucinated claims, miscalibrated scores, and internal inconsistency. If
   it isn't satisfied, it sends revision instructions back to the
   synthesizer (bounded to `SEO_AGENT_MAX_REFLECTION_ROUNDS`, default 2). On
   the last round, it stops at the last-reviewed draft rather than running
   one further, unreviewed revision. If still unresolved, the report is
   marked `review_status: "not_approved"` with the specific unresolved
   issues attached — filtered to drop complaints the deterministic
   reconciliation step has since mechanically resolved (e.g. weight-sum
   normalization), so the list stays honest.
3. **Tool use / function calling** — every specialist decides which of its
   tools to call and in what order (fetch pages, parse HTML, check
   robots.txt/sitemap, verify SSL, inspect security headers, sample links,
   run a real Lighthouse audit).
4. **Real performance, accessibility, and best-practices data via Google
   PageSpeed Insights** — the performance, accessibility, and best-practices
   specialists each pull from a genuine Lighthouse audit (not a proxy
   signal): actual LCP/CLS/TBT/Speed Index and a performance score, real
   WCAG-adjacent accessibility check failures, and real best-practices audit
   failures (vulnerable JS libraries, deprecated APIs, console errors,
   etc.). All three share one cached PageSpeed Insights call per
   `(url, strategy)` pair, so requesting the extra categories is genuinely
   near-free rather than tripling the API cost. Best Practices
   deterministically excludes HTTPS/SSL-related audit results (Security's
   domain already covers that, more authoritatively) rather than relying on
   a prompt instruction to avoid the overlap. Automatically retries on
   Google's known-common transient 500 errors.
5. **Live web-search-augmented research** — the competitive/benchmarking
   specialist runs on Groq's **Compound** system (`groq/compound-mini`),
   which performs web search server-side automatically.
6. **Persistent long-term memory** — every completed audit is written to
   SQLite (`agent/memory.py`), enabling trend tracking across repeated runs.
7. **Schema-validated structured output** — every final report is validated
   against a Pydantic schema before being trusted downstream.
8. **A deterministic fact-checking layer** (`postprocess.py`) that catches
   an entire class of LLM self-reporting failures in code rather than
   prompting: SSL reconciliation, real Core Web Vitals/accessibility/
   best-practices injection (via a shared, stemmed keyword-overlap matcher,
   not exact-substring matching), stale data-limitations correction,
   cross-specialist on-page/technical overlap removal, summary/trend
   contradiction and fabrication detection, bot-block detection, empty-
   category recovery, and stale-critic-complaint filtering.
9. **Multi-key, multi-model resilience** — automatic fallback to a smaller
   model on rate/quota limits, proactive round-robin rotation across
   multiple configured API keys spanning the *entire* run (not just
   specialists), automatic payload-shrinking on request-too-large errors,
   and a JSON self-repair retry if a response comes back truncated.
   `agent/compaction.py` handles the synthesizer/critic side of this
   *proactively* rather than only reactively: it estimates request size
   before sending and, only above a size threshold, trims lowest-priority
   findings and overlong free-text fields — so the first request has a real
   chance of succeeding instead of needing a 413 round-trip to find out it
   was too big.
10. **Automated regression test suite** — 241 `pytest` tests covering every
    deterministic reconciliation function, every tool implementation
    (network fully mocked), the retry/rate-limit engine, the reflection
    loop (including a dedicated regression test for a specific,
    previously-reintroduced bug), and end-to-end pipeline wiring. See
    Testing below.
11. **Self-grading eval harness** — runs the real pipeline against a
    randomly-sampled subset of a curated pool of benchmark sites with
    known, stable, verifiably-true issues (expired/self-signed/mismatched/
    untrusted certificates, an HTTP-only host, pages missing a meta
    description), grading the result in pure Python — no LLM call is spent
    on grading. Each sampled case starts on a different configured API key,
    reproducible via a logged seed. See Eval harness below.

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

# Run the self-grading eval harness against known-issue benchmark sites
python main.py eval

# Eval harness with the strong model instead of the default cheap/fast mode
python main.py eval --mode auto --out eval_results.json
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
├── main.py                  CLI entry point (audit / history / eval subcommands, --mode flag)
├── pytest.ini                pytest config -- `pytest` just works from the project root
├── requirements.txt
├── .env.example
├── README.md
├── agent/
│   ├── __init__.py           exposes run_full_audit
│   ├── config.py             model names, API keys, & tunables, all overridable via env vars
│   ├── tools.py              low-level tool implementations: fetch (with bot-block detection),
│   │                         parse, ssl (certifi-based), headers, links (with bot-block
│   │                         detection), and real Lighthouse data via PageSpeed Insights --
│   │                         Core Web Vitals, accessibility, and best-practices audits all
│   │                         share one cached API call per (url, strategy) (with retry-on-
│   │                         transient-500)
│   ├── tool_schemas.py        Groq/OpenAI-format tool-use schemas, grouped per specialist
│   ├── base_agent.py          generic reusable ToolAgent -- the agentic loop runtime, including
│   │                          Groq rate-limit/quota handling, model fallback, API-key rotation,
│   │                          request-too-large payload shrinking, and JSON self-repair
│   ├── specialists.py         specialist system prompts + tool assignments (7 specialists +
│   │                          competitive)
│   ├── planner.py             planning agent
│   ├── synthesizer.py         synthesizer agent
│   ├── critic.py              critic agent + reflection loop controller (stops at the last-
│   │                          reviewed draft rather than running an unreviewed final revision)
│   ├── postprocess.py         the deterministic fact-checking layer -- see "How this project
│   │                          evolved" above for what each function catches and why. Includes
│   │                          the shared keyword_topic_covered matcher used both internally
│   │                          (accessibility/best-practices dedup) and by the eval harness
│   ├── orchestrator.py        top-level pipeline: mode configs, canonical category names,
│   │                          category recovery/cleanup, deterministic score/weight
│   │                          reconciliation, stale-critic-complaint filtering, wiring it all
│   │                          together
│   ├── memory.py              SQLite persistence + trend lookups
│   ├── schemas.py             Pydantic validation of the final report contract
│   ├── report_pdf.py          reportlab-based PDF export of a completed audit
│   ├── eval_harness.py        self-grading eval harness -- see "Eval harness" below
│   └── compaction.py          proactive payload compaction for the synthesizer/critic --
│                               estimates request size BEFORE sending and, only above a
│                               token-estimate threshold, drops lowest-priority ("good")
│                               findings first and truncates overlong free-text fields, so
│                               the first request has a real chance of succeeding instead of
│                               relying on base_agent.py's reactive 413-triggered shrinking
└── tests/                     241 pytest tests -- see "Testing" below
    ├── conftest.py             shared fixtures: fake Groq client/errors, sample report data
    ├── test_base_agent.py      retry/backoff/rate-limit engine, the tool-call loop
    ├── test_compaction.py      proactive payload trimming, incl. the actual synthesizer/
    │                           critic wiring (not just the compaction functions in isolation)
    ├── test_critic.py          reflection loop, incl. a dedicated regression test
    ├── test_eval_harness.py    eval harness grading logic (run_full_audit fully mocked)
    ├── test_memory.py          SQLite persistence
    ├── test_orchestrator.py    score/weight reconciliation, category recovery, pipeline wiring
    ├── test_postprocess.py     every deterministic reconciliation function
    ├── test_schemas.py         the Pydantic report contract
    └── test_tools.py           every tool implementation, network fully mocked
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
| `SEO_AGENT_COMPACTION_TOKEN_THRESHOLD` | 4000 | Estimated-token threshold above which the synthesizer/critic payload is proactively compacted before sending (see `agent/compaction.py`) |
| `SEO_AGENT_COMPACTION_MAX_FINDINGS` | 12 | Max findings kept per category when compacting (lowest-severity dropped first) |
| `SEO_AGENT_COMPACTION_MAX_EVIDENCE_CHARS` | 600 | Max length for `raw_evidence_notes`/critic instructions when compacting |
| `SEO_AGENT_COMPACTION_MAX_FINDING_CHARS` | 400 | Max length for an individual finding's issue/recommendation text when compacting |

CLI-only: `--mode {quick,deep,auto}` on `audit` and `eval` (see Usage above).

## Testing

```bash
pip install pytest   # already in requirements.txt
pytest                # runs all 241 tests, ~10-25s, zero network/API calls
```

Every test mocks the Groq client and any network calls (`requests.get`,
`socket`, etc.) — the suite runs with no API key and no internet access.
Coverage by file is in the Project layout table above; a few worth calling
out specifically:

- `test_critic.py` includes a test that reintroduces the exact reflection-
  loop regression described in "How this project evolved" (the critic
  rejecting on the final round used to trigger one more, unreviewed
  synthesis pass) to confirm it's genuinely caught, not just present by
  coincidence — verified by temporarily reverting the fix and watching the
  test fail before restoring it.
- `test_postprocess.py` includes regression tests for the accessibility/
  best-practices duplicate-finding bugs and the stale-weight-complaint bug
  described above, each reproducing the exact real-world finding text that
  triggered them.
- `test_eval_harness.py` includes a test that feeds the grader a synthetic
  hallucinated "certificate is valid" report against the real
  `expired-ssl-certificate` benchmark case, confirming the harness would
  actually catch that regression if `reconcile_ssl_findings` ever broke.

## Eval harness

```bash
python main.py eval                                        # random 4-of-7 sample (default)
python main.py eval --sample-size 7                         # run the whole pool
python main.py eval --sample-size 3 --seed 123               # reproduce an exact past sample
python main.py eval --mode auto --out results.json           # more thorough, more expensive
```

Runs the **real** pipeline (real API calls, real network requests — not
mocked) against a random sample of a curated pool of benchmark sites, each
chosen for having a known, stable, verifiably-true issue rather than being
a "representative" site that could change tomorrow:

| Case | Site | Known-true issue |
|---|---|---|
| `expired-ssl-certificate` | `expired.badssl.com` | permanently expired cert |
| `self-signed-ssl-certificate` | `self-signed.badssl.com` | fails cert verification |
| `wrong-hostname-ssl-certificate` | `wrong.host.badssl.com` | cert issued for a different domain |
| `untrusted-root-ssl-certificate` | `untrusted-root.badssl.com` | signed by an untrusted CA |
| `no-encryption-null-cipher` | `null.badssl.com` | refuses real encryption |
| `minimal-page-missing-meta-description` | `example.com` | no meta description tag |
| `minimal-historical-page` | `info.cern.ch` | no meta description tag |

An earlier `http-only-no-tls` case (`neverssl.com`) was **removed** after a
real eval run falsified its premise: the case assumed "never redirects
http:// to https://" meant "has no valid SSL/TLS," but a real run showed
`check_ssl_certificate` completing a genuinely valid TLS handshake against
the domain — an accurate tool result that just didn't match what the case
expected. No current specialist tool actually tests "does this URL
auto-upgrade to https," so there was no honest `expected_findings` to write
for the site's real behavior. See the comment above `BENCHMARK_CASES` in
`eval_harness.py` for the full reasoning if a similar case gets proposed
again.

By default, each run randomly samples 4 of these 7 (`DEFAULT_SAMPLE_SIZE`
in `eval_harness.py`) rather than always running the full pool — this
exercises different combinations across repeated runs (catching a fix that
happens to work for one site's exact phrasing but not another's) while
keeping any single run's cost bounded. The exact sample is logged with a
seed (`--seed 123` reproduces it exactly, e.g. to debug a specific
failure); requesting a sample size at or above the pool size runs
everything, in the pool's original order, no shuffling.

**Each sampled case now starts its audit on a different configured
`GROQ_API_KEYS` entry** (round-robin by sample position via
`run_full_audit`'s `starting_key_index`), rather than every case starting
on key 0 the way an earlier version of this harness did — that earlier
behavior meant configuring more keys didn't actually spread a sequence of
audits across them. With this fixed, more configured keys directly buys
headroom to raise `--sample-size` without concentrating load on a single
key's daily quota — a sample size up to N spreads one key per case with N
keys configured.

Grading is **100% deterministic Python** — `postprocess.py::keyword_topic_covered`
(the same stemmed/majority-keyword matcher used internally for
accessibility/best-practices dedup) checks whether each site's known issue
shows up in the right report category at the right severity. No LLM call is
spent on grading; only the audits themselves consume API quota. Every SSL-
failure case also asserts the report must *not* contain a phrase like "is
valid" — a direct regression guard for the exact SSL-hallucination bug
`reconcile_ssl_findings` exists to prevent.

**Honest framing:** this measures *recall of a small set of known-true
issues*, not full precision/recall in the ML sense — there's no ground truth
for "every issue a page does or doesn't have," so false-positive rate isn't
computable this way. What it does verify: "did the pipeline still catch the
specific things we know for certain are true," which is exactly what a
regression test needs even though it's not a complete quality benchmark.

**Cost:** each sampled case is one full `run_full_audit()` call — the same
token cost as a real user audit (planner + specialists + synthesizer + up
to 2 critic rounds, each re-running the synthesizer). Defaults to
`--mode quick` (small fallback model) specifically to keep repeated/CI runs
affordable; a default run costs roughly 4x one normal quick-mode audit, not
more, since grading itself is free. `quick` mode's model has a separate
daily quota pool from `auto`/`deep` mode's primary model even on the same
API key, so running `eval` regularly doesn't eat into everyday audit quota
unless everyday usage is *also* run in `--mode quick`.

New pool entries added later should be spot-checked with a real
`python main.py eval` run before being trusted, the same as any existing
one — none of these were (or can be) verified from a sandboxed environment
without live internet access; they were chosen based on badssl.com's and
info.cern.ch's well-documented, deliberately-permanent stability, not by
directly confirming their current HTML/response against this codebase.

## Honest limitations

- No JavaScript rendering — client-side-rendered content/SEO tags are
  invisible to the HTML parser, the same blind spot classic crawlers have
  (though real Core Web Vitals now come from a genuine browser-based
  Lighthouse audit, which does render JS).
- Link checking samples a handful of links, not a full-site crawl. If a high
  fraction fail at once, the tool flags this as likely bot-blocking rather
  than confidently reporting a broken-links crisis — but this still needs
  manual spot-checking to be sure.
- Automated accessibility and best-practices audits (via Lighthouse) only
  catch roughly 30-50% of real-world WCAG issues — a clean automated result
  is a floor, not proof of full accessibility compliance. The report says
  this explicitly rather than implying otherwise.
- The eval harness measures recall of a small set of known-true issues, not
  full precision — see "Eval harness" above for why that's a meaningfully
  different (and more honest) claim than "measures accuracy."
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

- Wrap `run_full_audit()` in a FastAPI endpoint for a real backend/API (a
  first pass at this exists in the separate `webapp/` project).
- Add a `crawl` mode that runs the pipeline across multiple pages of a site
  and aggregates a site-wide score.
- Extend `postprocess.py`'s deterministic-reconciliation pattern to other
  recurring hallucination classes as they're discovered.
- Add an "auto-fix" agent that drafts corrected meta tags / alt text for
  low-effort findings.
- Keep growing the eval harness's benchmark pool (now 8, randomly sampled
  4-at-a-time) as more known-issue-with-a-stable-ground-truth sites are
  identified — the current 8 are a floor, not a target. Spot-check any new
  entry with a real `python main.py eval` run before trusting it.
- Hook `python main.py eval` into CI so a regression in the deterministic
  layer gets caught automatically, not just when someone happens to notice
  a duplicate/stale finding in a manual run (as items in "How this project
  evolved" above were).