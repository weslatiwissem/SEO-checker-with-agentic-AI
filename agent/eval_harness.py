"""
Self-grading eval harness.

Runs the full audit pipeline against a small, curated set of benchmark
sites chosen specifically because they have known, stable, verifiably-true
issues (expired/self-signed certificates, plain-HTTP-only hosts, minimal
pages missing standard SEO tags) -- then deterministically checks whether
each audit's findings actually caught the issue, in the right category,
at the right severity.

IMPORTANT, HONEST FRAMING: this measures RECALL of a small set of known-true
issues, not full precision/recall in the ML sense. We don't have a ground
truth for "every issue a page does or doesn't have," so we can't compute
false-positive rate. What this DOES tell you: "did the pipeline still catch
the specific things we know for certain are true" -- which is exactly what
a regression test needs, even if it's not a complete quality benchmark.

Grading is 100% deterministic Python (keyword_topic_covered from
postprocess.py -- the same matcher already used, and already bug-fixed
twice, for Lighthouse-audit deduplication) -- no LLM call is spent on
grading itself, only on running the audits.

Cost note: each benchmark case is one full run_full_audit() call, i.e. the
same token cost as a real user audit (planner + specialists + synthesizer +
critic). Defaults to mode="quick" (small fallback model, cheapest quota
pool) to keep repeated runs affordable; pass mode="auto"/"deep" for a more
thorough but more expensive validation pass.
"""
from __future__ import annotations

import json
import time
from typing import Callable

from .orchestrator import run_full_audit
from .postprocess import keyword_topic_covered


# Each case's `url` was picked specifically for a known, stable, deterministic
# property -- not because it's a representative "real" site. Do not swap
# these for arbitrary sites; the whole point is that the expected issue is
# guaranteed true regardless of when this runs.
BENCHMARK_CASES: list[dict] = [
    {
        "name": "expired-ssl-certificate",
        "url": "https://expired.badssl.com/",
        "notes": "badssl.com's dedicated expired-certificate test host -- deliberately, "
                 "permanently serves an expired SSL certificate.",
        "expected_findings": [
            {
                "category": "Web Security",
                "keywords": ["ssl", "certificate", "expired"],
                "min_severity": "critical",
                # The exact bug reconcile_ssl_findings exists to prevent: a model
                # claiming an expired cert is valid. If this phrase shows up
                # anywhere in the report, the deterministic correction failed.
                "must_not_contain": ["is valid", "not expired"],
            },
        ],
    },
    {
        "name": "self-signed-ssl-certificate",
        "url": "https://self-signed.badssl.com/",
        "notes": "badssl.com's dedicated self-signed-certificate test host -- fails "
                 "standard certificate verification every time.",
        "expected_findings": [
            {
                "category": "Web Security",
                "keywords": ["ssl", "certificate"],
                "min_severity": "critical",
            },
        ],
    },
    {
        "name": "http-only-no-tls",
        "url": "http://neverssl.com/",
        "notes": "neverssl.com is purpose-built to never redirect to HTTPS.",
        "expected_findings": [
            {
                "category": "Web Security",
                "keywords": ["ssl", "https", "certificate"],
                "min_severity": "warning",
            },
        ],
    },
    {
        "name": "minimal-page-missing-meta-description",
        "url": "https://example.com/",
        "notes": "IANA's example.com -- a deliberately minimal, extremely stable page "
                 "with no meta description tag.",
        "expected_findings": [
            {
                "category": "On-Page Content",
                "keywords": ["meta", "description"],
                "min_severity": "warning",
            },
        ],
    },
]

_SEVERITY_RANK = {"good": 0, "warning": 1, "critical": 2}


def _category_findings_text(report: dict, category_name: str) -> str | None:
    """Return the combined issue+recommendation text for a category, or
    None if that category doesn't exist in the final report at all (e.g.
    its specialist failed and _recover_or_drop_empty_categories dropped it
    -- that's itself a miss worth surfacing distinctly from 'ran but didn't
    catch the issue')."""
    for cat in report.get("categories", []):
        if cat.get("name") == category_name:
            return " ".join(
                f"{f.get('issue', '')} {f.get('recommendation', '')}"
                for f in cat.get("findings", [])
            )
    return None


def _category_max_severity(report: dict, category_name: str, keywords: list[str]) -> str | None:
    """Among findings in this category that actually mention the topic
    keywords, return the highest severity found (so 'category has SOME
    critical finding, but not about this specific issue' doesn't count)."""
    best = None
    for cat in report.get("categories", []):
        if cat.get("name") != category_name:
            continue
        for f in cat.get("findings", []):
            text = f"{f.get('issue', '')} {f.get('recommendation', '')}"
            if keyword_topic_covered(keywords, text):
                sev = f.get("severity")
                if best is None or _SEVERITY_RANK.get(sev, -1) > _SEVERITY_RANK.get(best, -1):
                    best = sev
    return best


def grade_case(case: dict, report: dict) -> dict:
    """Deterministically grade one benchmark case's audit report against
    its expected_findings. Never raises -- a malformed/missing category is
    a graded miss, not a crash, so one bad case can't take down the whole
    harness run."""
    results = []
    for expected in case["expected_findings"]:
        category = expected["category"]
        keywords = expected["keywords"]
        min_severity = expected.get("min_severity", "warning")
        must_not_contain = expected.get("must_not_contain", [])

        findings_text = _category_findings_text(report, category)
        category_present = findings_text is not None
        findings_text_lower = (findings_text or "").lower()

        topic_caught = category_present and keyword_topic_covered(keywords, findings_text_lower)
        actual_severity = _category_max_severity(report, category, keywords) if topic_caught else None
        severity_ok = (
            actual_severity is not None
            and _SEVERITY_RANK.get(actual_severity, -1) >= _SEVERITY_RANK.get(min_severity, 1)
        )

        forbidden_hits = [phrase for phrase in must_not_contain if phrase.lower() in findings_text_lower]

        results.append({
            "category": category,
            "keywords": keywords,
            "category_present_in_report": category_present,
            "topic_caught": topic_caught,
            "min_severity_required": min_severity,
            "actual_severity": actual_severity,
            "severity_ok": severity_ok,
            "forbidden_phrase_hits": forbidden_hits,
            "passed": topic_caught and severity_ok and not forbidden_hits,
        })

    return {
        "name": case["name"],
        "url": case["url"],
        "expected_count": len(results),
        "caught_count": sum(1 for r in results if r["passed"]),
        "results": results,
        "all_passed": all(r["passed"] for r in results),
    }


def run_eval(
    mode: str = "quick",
    cases: list[dict] | None = None,
    use_memory: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Run the full pipeline against each benchmark case and grade it.
    A single case's audit raising an exception is caught and recorded as a
    failed case (with the error message) rather than aborting the whole
    run -- one flaky/unreachable benchmark site shouldn't block grading
    the others. Returns a summary dict; nothing here makes any grading
    LLM call -- only the audits themselves consume API quota."""
    log_fn = log_fn or (lambda msg: None)
    cases = cases if cases is not None else BENCHMARK_CASES

    case_results = []
    for case in cases:
        log_fn(f"[eval] Running benchmark case '{case['name']}' ({case['url']}) in mode={mode}...")
        started = time.monotonic()
        try:
            report = run_full_audit(case["url"], use_memory=use_memory, mode=mode, log_fn=log_fn)
        except Exception as e:
            log_fn(f"[eval]   -> FAILED to complete audit: {e}")
            case_results.append({
                "name": case["name"],
                "url": case["url"],
                "expected_count": len(case["expected_findings"]),
                "caught_count": 0,
                "results": [],
                "all_passed": False,
                "audit_error": str(e),
            })
            continue

        elapsed = round(time.monotonic() - started, 1)
        graded = grade_case(case, report)
        graded["elapsed_seconds"] = elapsed
        graded["review_status"] = report.get("review_status")
        case_results.append(graded)
        log_fn(f"[eval]   -> {graded['caught_count']}/{graded['expected_count']} caught "
               f"({elapsed}s, review_status={graded.get('review_status')})")

    total_expected = sum(c["expected_count"] for c in case_results)
    total_caught = sum(c["caught_count"] for c in case_results)
    recall = round(total_caught / total_expected, 3) if total_expected else None

    return {
        "mode": mode,
        "case_count": len(case_results),
        "total_expected": total_expected,
        "total_caught": total_caught,
        "recall": recall,
        "cases_fully_passed": sum(1 for c in case_results if c["all_passed"]),
        "cases": case_results,
    }


def print_eval_summary(summary: dict) -> None:
    print("\n" + "=" * 64)
    print("EVAL HARNESS SUMMARY")
    print("=" * 64)
    print(f"Mode: {summary['mode']}")
    print(f"Known-issue recall: {summary['total_caught']}/{summary['total_expected']} "
          f"({summary['recall'] * 100:.1f}%)" if summary["recall"] is not None else "N/A")
    print(f"Cases fully passed: {summary['cases_fully_passed']}/{summary['case_count']}")
    print("-" * 64)
    for case in summary["cases"]:
        status = "PASS" if case["all_passed"] else "FAIL"
        print(f"[{status}] {case['name']} ({case['url']}) "
              f"-- {case['caught_count']}/{case['expected_count']} caught")
        if case.get("audit_error"):
            print(f"       audit error: {case['audit_error']}")
            continue
        for r in case["results"]:
            if not r["passed"]:
                reason = []
                if not r["category_present_in_report"]:
                    reason.append("category missing from final report")
                elif not r["topic_caught"]:
                    reason.append("no finding mentioned the expected topic")
                elif not r["severity_ok"]:
                    reason.append(f"severity was {r['actual_severity']!r}, needed >= {r['min_severity_required']!r}")
                if r["forbidden_phrase_hits"]:
                    reason.append(f"contained forbidden phrase(s): {r['forbidden_phrase_hits']}")
                print(f"         - MISS in '{r['category']}' ({', '.join(reason)})")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the self-grading eval harness")
    parser.add_argument("--mode", choices=["quick", "auto", "deep"], default="quick")
    parser.add_argument("--out", help="Write full JSON results to this file", default=None)
    args = parser.parse_args()

    summary = run_eval(mode=args.mode, log_fn=print)
    print_eval_summary(summary)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Full JSON results written to {args.out}")