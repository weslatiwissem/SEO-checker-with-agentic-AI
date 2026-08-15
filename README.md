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
Samsung, Internet Archive's blog, and others) to survive both Groq's
free-tier limits and the kinds of mistakes LLMs make when asked to
self-report facts and do arithmetic. Backed by a 317-test automated
`pytest` suite and a self-grading eval harness (see Testing / Eval harness
below).

No frontend in this document — CLI + importable library. (A Next.js web UI
exists as a separate, in-progress add-on; see the `webapp/` project.)

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
pipeline through real, repeated testing on Groq's free tier. Nearly every
part of the current design exists because of a specific bug caught this
way — LLMs are treated as fallible reasoning engines that propose findings,
while anything verifiable or computable outright is enforced
deterministically in code around them instead of just prompted more firmly.

- **Bad arithmetic.** The model's own overall-score math rarely matched its
  category scores/weights. Fixed: recompute deterministically
  (`_reconcile_overall_score`), near-zero tolerance.
- **Restating already-known facts wrong.** A smaller model would sometimes
  hallucinate "certificate expired" from a perfectly valid cert. Fixed: the
  tool computes `is_expired` with certainty; any contradicting finding gets
  deterministically overwritten (`reconcile_ssl_findings`).
- **Same pattern for real performance/accessibility/best-practices data.**
  Findings sometimes stayed vague instead of citing the real Lighthouse
  numbers. Fixed: canonical findings get injected from the real data when
  missing (`reconcile_core_web_vitals`, and a shared matcher —
  `keyword_topic_covered` — for accessibility/best-practices). That matcher
  itself needed two rounds of fixing: exact-substring matching produced
  false "not covered" results on ordinary rewording (plurals, word order),
  and a coincidental digit match (score "50" inside an unrelated "30-50%"
  phrase) once wrongly suppressed a real injection — both fixed with
  stemmed, boundary-aware matching.
- **Specialists don't share ground truth.** The competitive specialist
  repeatedly re-judged title/meta/canonical facts another specialist had
  already measured, sometimes contradicting them. Fixed: stripped
  deterministically (`strip_competitive_onpage_overlap`).
- **Prose contradicting its own structured data.** Wrong trend direction,
  wrong magnitude, or a fabricated comparison to a "previous audit" that
  never existed. Fixed: `fix_summary_trend_mismatch` and
  `fix_fabricated_trend_claim`.
- **Categories silently losing real data.** A specialist could succeed and
  still have the synthesizer drop its findings during merging. Fixed:
  `_recover_or_drop_empty_categories` recovers the original findings first
  (and now the score too, independently — see below), before dropping.
- **Anti-bot blocking corrupting an audit.** A block page's "no headings,
  no images" got scored as real content. Fixed: `fetch_page` flags
  likely-blocked responses; a standard browser User-Agent and a
  high-failure-rate caution flag in link checking help too.
- **A rejected draft shipping anyway, via a wasted extra call.** The critic
  could reject on its final round and the pipeline would still run one
  more, never-reviewed pass. Fixed: stop at the last-reviewed draft;
  `review_status` surfaces exactly what's unresolved.
- **Free-tier quota exhaustion.** Now handled with real strategy: parses
  Groq's wait-time formats, prefers instant model-fallback over waiting,
  round-robins every configured API key continuously across the whole run
  (not per-stage), shrinks oversized payloads reactively on 413s *and*
  proactively before sending (`agent/compaction.py`), and only fails fast
  when a wait is genuinely too long.
- **A stale cert trust store.** `requests` and the raw `ssl` module used
  different root CA lists, occasionally producing a false "expired."
  Fixed: both now use the same `certifi` bundle.
- **A resolved complaint still shown as "unresolved."** The critic reviews
  *before* weight normalization runs, so "weights don't sum to 1.0" could
  survive into the final report as an active complaint about something
  already fixed. Fixed: `_drop_issues_resolved_by_reconciliation` strips
  only that mechanically-resolved class, leaving genuine judgment calls alone.
- **A finding saying "OK" and "Failing checks include: X" at once.**
  Lighthouse can have a 100/100 score with one zero-weight informational
  audit still "failing" — both facts true, but self-contradictory to read.
  Fixed: never label a finding "good" while it's also naming a failing check.
- **Every audit starting key rotation from the same key.** Running several
  audits back-to-back (as the eval harness does) never actually spread
  them across multiple configured keys. Fixed: `starting_key_index` on
  `run_full_audit()`, rotated per case.
- **A failed specialist's raw JSON leaking into the synthesizer/critic as
  trusted data.** The error message embedding a truncated JSON attempt got
  copied verbatim into `raw_evidence_notes` — both downstream agents then
  cited "findings" that were never actually parsed. Fixed: a short,
  explicitly-non-data placeholder message instead.
- **A category shipping with real-looking findings but a `null` score.**
  Downstream of the bug above. Fixed: `_recover_or_drop_empty_categories`
  now validates `score` independently of findings, dropping the whole
  category if the specialist itself never had a valid one either.
- **A benchmark case's own premise being wrong.** The eval harness assumed
  a site "never redirecting to https" meant "no valid TLS" — a real run
  showed a genuinely valid handshake, falsifying the premise. Removed
  rather than left as a permanent false-negative generator.
- **Correlation ranking rewarding small-sample noise.** In Score Analytics,
  sorting by raw `|r|` let an 8-sample correlation (nonsensical sign) rank
  above a 51-sample one (sensible sign). Fixed: reliable results (n≥15)
  always rank first, unreliable ones are flagged, not hidden.

All of the above were found by actually running the pipeline against real
sites and real accumulated data — not by adding a feature and assuming a
clean first run meant it worked.

## Agentic AI features

1. **Multi-agent orchestration** — a planner and up to 7 specialists
   (technical SEO, content, performance, security, links, accessibility,
   best practices, plus competitive when relevant) run **concurrently**,
   reconciled by a synthesizer.
2. **Reflection / self-critique loop** — a critic reviews the draft against
   raw evidence and can send it back for revision (max 2 rounds by
   default), stopping at the last-reviewed draft rather than shipping an
   unreviewed final pass. `review_status`/`unresolved_review_issues`
   surface what's still unresolved, filtered to exclude complaints already
   mechanically fixed.
3. **Tool use / function calling** — each specialist chooses its own tool
   calls (fetch, parse, SSL check, headers, links, real Lighthouse audit).
4. **Real performance/accessibility/best-practices data** via Google
   PageSpeed Insights — genuine Lighthouse data, not a proxy signal, shared
   across one cached API call per `(url, strategy)` so the extra categories
   are near-free. Best Practices deterministically excludes HTTPS/SSL
   audits (Security's domain) rather than relying on a prompt to avoid the
   overlap.
5. **Live web-search-augmented research** — the competitive specialist runs
   on Groq's **Compound** system (`groq/compound-mini`), web search
   server-side.
6. **Persistent long-term memory** — every audit written to SQLite
   (`agent/memory.py`), enabling trend tracking and the Score Analytics
   dataset below.
7. **Schema-validated structured output** — every final report validated
   against a Pydantic schema.
8. **A deterministic fact-checking layer** (`postprocess.py`) — see "How
   this project evolved" for the full list of what it catches and why.
9. **Multi-key, multi-model resilience** — automatic model fallback,
   proactive round-robin across all configured keys for the entire run,
   proactive *and* reactive payload shrinking, JSON self-repair retries.
10. **317-test automated regression suite** — see Testing below.
11. **Self-grading eval harness** — runs the real pipeline against a
    randomly-sampled pool of benchmark sites with known, verifiable
    issues, graded in pure Python (no LLM call spent on grading). See Eval
    harness below.
12. **Score Analytics** — classical ML / statistics on top of the SQLite
    audit history: feature engineering, Pearson correlation, and
    cross-validated comparison of 4 scikit-learn regressors predicting
    `overall_score`. See Score Analytics below.
13. **Findings Similarity Search** — vector search (TF-IDF by default;
    optional real sentence embeddings) over a seed knowledge base plus real
    audit history, so a past finding/fix can be looked up by meaning rather
    than exact wording. See Findings Similarity Search below.

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
higher rate limit on real Lighthouse audits (works without one too, at a
lower rate limit).

## Usage

```bash
# Full multi-agent audit (auto mode: strong model, falls back if needed)
python main.py audit https://example.com

# Fast, always-available mode -- skips the 70B model entirely
python main.py audit https://example.com --mode quick

# Best-quality mode -- only the strong model, fails clearly rather than
# silently downgrading if its quota is exhausted
python main.py audit https://example.com --mode deep

# Also benchmark against a competitor
python main.py audit https://example.com --competitor https://competitor.com

# Save JSON + a polished PDF report
python main.py audit https://example.com --out report.json --pdf report.pdf

# Score history for a domain
python main.py history https://example.com

# Self-grading eval harness against known-issue benchmark sites
python main.py eval

# Classical ML / statistical analysis of your audit history
python main.py analyze

# Search past findings for ones similar to a new issue
python main.py similar "page missing meta description"
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
├── main.py                  CLI entry point (audit / history / eval / analyze, --mode flag)
├── pytest.ini                pytest config -- `pytest` just works from the project root
├── requirements.txt
├── .env.example
├── README.md
├── agent/
│   ├── __init__.py           exposes run_full_audit
│   ├── config.py             model names, API keys, & tunables, all overridable via env vars
│   ├── tools.py              fetch/parse/ssl/headers/links + real Lighthouse data via
│   │                         PageSpeed Insights (CWV, accessibility, best-practices share
│   │                         one cached API call per (url, strategy))
│   ├── tool_schemas.py        Groq/OpenAI-format tool-use schemas, grouped per specialist
│   ├── base_agent.py          the agentic loop runtime: rate-limit/quota handling, model
│   │                          fallback, API-key rotation, payload shrinking, JSON self-repair
│   ├── specialists.py         specialist system prompts + tool assignments
│   ├── planner.py             planning agent
│   ├── synthesizer.py         synthesizer agent
│   ├── critic.py              critic agent + reflection loop controller
│   ├── postprocess.py         the deterministic fact-checking layer -- see "How this project
│   │                          evolved" above
│   ├── orchestrator.py        top-level pipeline: mode configs, category recovery, score/
│   │                          weight reconciliation, stale-complaint filtering
│   ├── memory.py              SQLite persistence, trend lookups, full-history dataset access
│   ├── schemas.py             Pydantic validation of the final report contract
│   ├── report_pdf.py          reportlab-based PDF export
│   ├── eval_harness.py        self-grading eval harness -- see "Eval harness" below
│   ├── analytics.py           Score Analytics -- see "Score Analytics" below
│   ├── similarity_search.py   Findings Similarity Search -- see below
│   └── compaction.py          proactive payload compaction (see "How this project evolved")
└── tests/                     317 pytest tests -- see "Testing" below
    ├── conftest.py             shared fixtures: fake Groq client/errors, sample report data
    ├── test_analytics.py       feature engineering, correlations, model training/comparison
    ├── test_base_agent.py      retry/backoff/rate-limit engine, the tool-call loop
    ├── test_compaction.py      proactive payload trimming, incl. actual wiring
    ├── test_critic.py          reflection loop, incl. a dedicated regression test
    ├── test_eval_harness.py    eval harness grading logic (run_full_audit fully mocked)
    ├── test_memory.py          SQLite persistence
    ├── test_orchestrator.py    score/weight reconciliation, category recovery, pipeline wiring
    ├── test_postprocess.py     every deterministic reconciliation function
    ├── test_schemas.py         the Pydantic report contract
    ├── test_similarity_search.py  corpus building, TF-IDF search, embedding fallback
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
| `GROQ_API_KEYS` | — (optional) | Comma-separated list of multiple keys; specialists and synthesizer/critic calls round-robin across them for the whole run |
| `GOOGLE_PAGESPEED_API_KEY` | — (optional) | Raises the rate limit on real Lighthouse audits; works without one at a lower limit |
| `SEO_AGENT_MODEL` | `llama-3.3-70b-versatile` | Primary model used by specialists, synthesizer |
| `SEO_AGENT_PLANNER_MODEL` | same as above | Model for the planner agent |
| `SEO_AGENT_CRITIC_MODEL` | same as above | Model for the critic agent |
| `SEO_AGENT_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Used automatically on a rate/quota limit (separate quota pool); empty disables fallback |
| `SEO_AGENT_COMPETITIVE_MODEL` | `groq/compound-mini` | Groq's built-in web-search system, competitive specialist only |
| `SEO_AGENT_MAX_ITER` | 10 | Max tool-call iterations per agent |
| `SEO_AGENT_MAX_REFLECTION_ROUNDS` | 2 | Max critic revision rounds |
| `SEO_AGENT_MAX_WORKERS` | 2 | Max concurrent specialist agents |
| `SEO_AGENT_DISPATCH_STAGGER` | 2.0 | Seconds between dispatching each specialist |
| `SEO_AGENT_RATE_LIMIT_RETRIES` | 4 | Max retry attempts on a rate-limited call |
| `SEO_AGENT_DB_PATH` | `./data/audit_history.db` | SQLite history location |
| `SEO_AGENT_COMPACTION_TOKEN_THRESHOLD` | 4000 | Estimated-token threshold above which the synthesizer/critic payload is proactively compacted |
| `SEO_AGENT_COMPACTION_MAX_FINDINGS` | 12 | Max findings kept per category when compacting |
| `SEO_AGENT_COMPACTION_MAX_EVIDENCE_CHARS` | 600 | Max length for evidence notes/critic instructions when compacting |
| `SEO_AGENT_COMPACTION_MAX_FINDING_CHARS` | 400 | Max length for a finding's issue/recommendation text when compacting |

CLI-only: `--mode {quick,deep,auto}` on `audit` and `eval` (see Usage above).

## Testing

```bash
pip install pytest   # already in requirements.txt
pytest                # runs all 317 tests, zero network/API calls
```

Every test mocks the Groq client and any network calls — the suite runs
with no API key and no internet access. A few worth calling out:

- `test_critic.py` reintroduces the exact reflection-loop regression noted
  above to confirm it's genuinely caught, verified by temporarily
  reverting the fix and watching the test fail before restoring it.
- `test_postprocess.py` reproduces the exact real-world text that triggered
  the accessibility/best-practices dedup and stale-weight-complaint bugs.
- `test_eval_harness.py` feeds the grader a synthetic hallucinated
  "certificate is valid" report against the real
  `expired-ssl-certificate` case, confirming it would catch that
  regression if `reconcile_ssl_findings` ever broke.
- `test_analytics.py` reproduces the exact small-sample-correlation
  scenario above to confirm reliable results always outrank unreliable ones.

## Eval harness

```bash
python main.py eval                                        # random 4-of-7 sample (default)
python main.py eval --sample-size 7                         # run the whole pool
python main.py eval --sample-size 3 --seed 123               # reproduce an exact past sample
python main.py eval --mode auto --out results.json           # more thorough, more expensive
```

Runs the **real** pipeline against a random sample of a curated pool of
benchmark sites, each chosen for a known, stable, verifiably-true issue:

| Case | Site | Known-true issue |
|---|---|---|
| `expired-ssl-certificate` | `expired.badssl.com` | permanently expired cert |
| `self-signed-ssl-certificate` | `self-signed.badssl.com` | fails cert verification |
| `wrong-hostname-ssl-certificate` | `wrong.host.badssl.com` | cert issued for a different domain |
| `untrusted-root-ssl-certificate` | `untrusted-root.badssl.com` | signed by an untrusted CA |
| `no-encryption-null-cipher` | `null.badssl.com` | refuses real encryption |
| `minimal-page-missing-meta-description` | `example.com` | no meta description tag |
| `minimal-historical-page` | `info.cern.ch` | no meta description tag |

By default, each run randomly samples 4 of these 7 rather than always
running the full pool, logged with a seed (`--seed 123` reproduces it
exactly). Each sampled case starts its audit on a different configured
`GROQ_API_KEYS` entry (round-robin), so more configured keys directly buys
headroom to raise `--sample-size`.

Grading is **100% deterministic Python** — `keyword_topic_covered` (the
same matcher used internally for accessibility/best-practices dedup)
checks whether each site's known issue shows up in the right category at
the right severity. No LLM call is spent grading; only the audits
themselves consume API quota. Every SSL-failure case also asserts the
report must *not* say "is valid" — a direct regression guard.

**Honest framing:** this measures *recall of known-true issues*, not full
precision/recall — there's no ground truth for "every issue a page does or
doesn't have." What it verifies: did the pipeline still catch the specific
things we know for certain are true.

**Cost:** each sampled case is one full `run_full_audit()` call. Defaults
to `--mode quick` to keep repeated/CI runs affordable — that model has a
separate daily quota pool from `auto`/`deep` mode's primary model, so
running `eval` regularly doesn't eat into everyday audit quota unless
everyday usage is *also* run in `--mode quick`.

## Score Analytics

```bash
python main.py analyze                          # real data if >=20 audits, else synthetic
python main.py analyze --source synthetic --n-synthetic 200
python main.py analyze --source real             # force real data even if sparse
python main.py analyze --out results.json
```

Classical ML / statistics on top of your SQLite audit history:

- **Feature engineering** — category scores/weights, finding counts by
  severity, category presence, review outcome, extracted from every
  stored report (`agent/analytics.py`).
- **Statistical modeling** — Pearson correlation of every feature against
  `overall_score`, with p-values. Results are tagged `reliable` (n≥15) and
  sorted so well-supported results always outrank raw-magnitude noise —
  see "How this project evolved" for the real example that motivated this.
- **Classical ML + quantitative comparison** — `LinearRegression`, `Ridge`,
  `RandomForestRegressor`, `GradientBoostingRegressor`, compared via
  k-fold cross-validated R²/MAE/RMSE, never a single train/test split.

**Honest limitation:** real audit history starts small. `--source auto`
(default) only uses real data once you have 20+ audits; below that, it
falls back to a clearly-labeled **synthetic** dataset (built from a known,
verifiable weighted-sum generative formula) so the module is genuinely
runnable and testable before real data accumulates. Every output states
explicitly whether it used real or synthetic data — never silently.

## Findings Similarity Search

```bash
python main.py similar "page missing meta description"
python main.py similar "weak TLS cipher" --category "Web Security" --top-k 3
python main.py similar "slow page load" --backend embedding   # real semantic search, if available
```

Vector search over collected findings, so a similar past issue+fix can be
looked up by meaning instead of exact wording. Standalone (`agent/
similarity_search.py`) — **not currently wired into the live audit
pipeline**; see Possible next steps.

Two backends, both in `build_index()`:
- **TF-IDF** (default) — classical lexical vector search via scikit-learn,
  no downloads, always available.
- **Sentence embeddings** (`--backend embedding`) — genuine semantic
  similarity via `sentence-transformers`, if installed and a model can be
  downloaded. Falls back to TF-IDF automatically, with a clear log
  message, if either isn't available — never raises just because the
  optional path is missing.

The searchable corpus mixes two clearly-labeled kinds of entries: a small,
hand-authored **seed knowledge base** (~26 common findings/fixes across all
7 categories, so this is useful before much real history exists) and real
findings pulled from your stored audit history. Every search result states
which one it came from.

**Honest limitation:** TF-IDF matches shared vocabulary, not meaning — a
query like "takes forever to load" won't match a finding about "Largest
Contentful Paint" the way a real embedding model would. Install
`sentence-transformers` for genuine semantic matching.

## Honest limitations

- No JavaScript rendering for HTML parsing (though real Core Web Vitals do
  come from a genuine browser-based Lighthouse audit, which renders JS).
- Link checking samples a handful of links, not a full crawl; a high
  failure rate is flagged as likely bot-blocking rather than reported as a
  confident broken-links crisis, but still needs manual spot-checking.
- Automated accessibility/best-practices audits only catch roughly 30-50%
  of real-world WCAG issues — a clean result is a floor, not proof of
  compliance.
- The eval harness measures recall of known-true issues, not full
  precision. Score Analytics results are only as trustworthy as the
  sample size behind them (see the `reliable` flag).
- Groq's free tier has real daily quota limits per model *and*
  organization — multiple API keys only help if they're genuinely separate
  accounts, not just multiple keys on one.
- Google's PageSpeed Insights API has occasional transient outages;
  retried automatically, but can still fail on a bad day for Google.
- The smaller fallback model can still produce lower-quality prose in
  findings not covered by an existing deterministic rule. `--mode deep`
  avoids it entirely for the highest-confidence results.
- The system reports its known limitations in `data_limitations` on every run.

## Possible next steps

- Wrap `run_full_audit()` in a FastAPI endpoint for a real backend/API (a
  first pass exists in the separate `webapp/` project).
- Add a `crawl` mode that audits multiple pages of a site for a site-wide score.
- Extend `postprocess.py`'s deterministic-reconciliation pattern to other
  recurring hallucination classes as they're discovered.
- Add an "auto-fix" agent that drafts corrected meta tags / alt text.
- Keep growing the eval harness's benchmark pool (now 7) and Score
  Analytics' real dataset (currently well under the 20-row threshold for
  trustworthy real-data analysis) as more audits accumulate.
- Wire Findings Similarity Search into the live pipeline — e.g. giving the
  synthesizer retrieved similar past fixes as grounding context instead of
  writing recommendations from scratch each time.
- Hook `python main.py eval` (and `analyze`) into CI so a regression gets
  caught automatically, not just when someone notices in a manual run.