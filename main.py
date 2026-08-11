#!/usr/bin/env python3
"""
Multi-agent SEO & website health checker.

Usage:
    python main.py audit https://example.com
    python main.py audit https://example.com --competitor https://competitor.com
    python main.py audit https://example.com --out report.json --pdf report.pdf
    python main.py history https://example.com
"""
import argparse
import json
import sys

from dotenv import load_dotenv
load_dotenv()  # MUST run before importing agent.* -- agent/config.py reads
                # env vars (like GROQ_API_KEYS) at import time, not lazily.

from agent import run_full_audit
from agent import memory
from agent.report_pdf import export_report_pdf
from agent.eval_harness import run_eval, print_eval_summary


def _log(msg: str) -> None:
    print(msg)


def print_report(report: dict) -> None:
    print("\n" + "=" * 64)
    print(f"SEO HEALTH REPORT: {report.get('url')}")
    print("=" * 64)
    print(f"Overall Score: {report.get('overall_score')}/100  (Grade: {report.get('grade')})")

    if report.get("review_status") == "not_approved":
        print("\n⚠ WARNING: this report was NOT approved by the critic's review after "
              "multiple revision rounds. Treat findings with extra scrutiny -- unresolved issues:")
        for issue in report.get("unresolved_review_issues", []):
            print(f"    - {issue}")

    trend = report.get("trend")
    if trend:
        delta = trend.get("score_delta", 0)
        arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
        print(f"Trend: {arrow} {delta:+g} vs previous audit ({trend.get('previous_timestamp')})")

    print(f"\n{report.get('summary')}\n")

    for cat in report.get("categories", []):
        print(f"--- {cat['name']}  [{cat['score']}/100, weight {cat['weight']}] ---")
        for f in cat.get("findings", []):
            icon = {"good": "OK ", "warning": "!! ", "critical": "XX "}.get(f.get("severity"), "-  ")
            print(f"  {icon}{f.get('issue')}")
            if f.get("recommendation"):
                print(f"       -> {f.get('recommendation')}")
        print()

    quick_wins = report.get("quick_wins", [])
    if quick_wins:
        print("Quick wins:")
        for qw in quick_wins:
            print(f"  * {qw}")
        print()

    if report.get("data_limitations"):
        print(f"Data limitations: {report['data_limitations']}")
    print("=" * 64 + "\n")


def cmd_audit(args):
    try:
        report = run_full_audit(
            args.url,
            competitor_url=args.competitor,
            use_memory=not args.no_memory,
            mode=args.mode,
            log_fn=_log if not args.quiet else (lambda m: None),
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Full JSON report written to {args.out}")

    if args.pdf:
        export_report_pdf(report, args.pdf)
        print(f"PDF report written to {args.pdf}")


def cmd_history(args):
    rows = memory.get_history(args.url, limit=args.limit)
    if not rows:
        print(f"No audit history found for {args.url}")
        return
    print(f"\nAudit history for {memory.domain_of(args.url)}:")
    print("-" * 64)
    for row in rows:
        print(f"  {row['timestamp']}  score={row['overall_score']:<6} grade={row['grade']}")
    print("-" * 64 + "\n")


def cmd_eval(args):
    summary = run_eval(mode=args.mode, sample_size=args.sample_size, seed=args.seed,
                        log_fn=_log if not args.quiet else (lambda m: None))
    print_eval_summary(summary)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Full JSON results written to {args.out}")
    # Non-zero exit if anything failed, so this can be dropped into CI.
    if summary["cases_fully_passed"] < summary["case_count"]:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent SEO & website health checker")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Run a full multi-agent audit on a URL")
    p_audit.add_argument("url", help="URL of the site to audit, e.g. https://example.com")
    p_audit.add_argument("--mode", choices=["quick", "deep", "auto"], default="auto",
                          help="quick: fast, always uses the small model. deep: best quality, "
                               "only the strong model (waits/fails rather than downgrading). "
                               "auto (default): strong model with automatic fallback.")
    p_audit.add_argument("--competitor", help="Optional competitor URL for benchmarking", default=None)
    p_audit.add_argument("--out", help="Write the full JSON report to this file", default=None)
    p_audit.add_argument("--pdf", help="Also export a polished PDF report to this path", default=None)
    p_audit.add_argument("--no-memory", action="store_true", help="Skip reading/writing audit history")
    p_audit.add_argument("--quiet", action="store_true", help="Suppress live agent activity log")
    p_audit.set_defaults(func=cmd_audit)

    p_history = sub.add_parser("history", help="Show past audit scores for a domain")
    p_history.add_argument("url", help="URL or domain to look up")
    p_history.add_argument("--limit", type=int, default=10)
    p_history.set_defaults(func=cmd_history)

    p_eval = sub.add_parser("eval", help="Run the self-grading eval harness against known benchmark sites")
    p_eval.add_argument("--mode", choices=["quick", "deep", "auto"], default="quick",
                         help="quick (default): cheapest, always uses the small model -- "
                              "recommended for repeated/CI runs. auto/deep: more thorough, "
                              "more expensive validation pass.")
    p_eval.add_argument("--sample-size", type=int, default=None,
                         help="How many benchmark cases to randomly sample from the pool "
                              "(default 4). With N configured API keys, a sample size up to "
                              "N spreads one key per case instead of concentrating load.")
    p_eval.add_argument("--seed", type=int, default=None,
                         help="Reproduce a specific sample from a previous run's logged seed")
    p_eval.add_argument("--out", help="Write full JSON results to this file", default=None)
    p_eval.add_argument("--quiet", action="store_true", help="Suppress live agent activity log")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()