"""
Findings Similarity Search: vector search over this project's collected
findings, so future recommendations can eventually be grounded in similar
past issues+fixes instead of relying purely on the LLM every time. (This
module is standalone -- it is NOT currently wired into the live audit
pipeline; see "Possible next steps" in the README.)

Two backends:
- TF-IDF (default, always available): classical lexical vector search via
  scikit-learn's TfidfVectorizer + cosine similarity. No downloads, no
  extra dependencies beyond scikit-learn (already required by
  agent/analytics.py), works fully offline.
- Sentence embeddings (optional, requires `pip install sentence-transformers`
  and a working internet connection the first time, to download the small
  pretrained model): genuine semantic similarity, not just shared words.
  Only used if explicitly requested (backend="embedding") and available;
  falls back to TF-IDF automatically (with a clear log message) if the
  package isn't installed or the model can't be loaded -- never silently
  half-broken, never raises just because the optional path isn't available.

HONEST LABELING: the corpus mixes two kinds of entries:
1. A small, hand-authored SEED knowledge base of common SEO/security/
   accessibility/performance findings and fixes (source="seed"), included
   so this module is useful even before much real audit history exists.
2. Real findings pulled from your actual stored audit history
   (source="real_audit", tagged with the real domain/timestamp).
Every search result states which one it came from -- never presented as if
both were the same kind of evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import memory

# --- Seed knowledge base ---------------------------------------------------
# Hand-authored, common, well-established findings and fixes -- NOT real
# audit history. Chosen to mirror the kinds of findings this project's own
# specialists actually produce (see specialists.py/postprocess.py), so
# search results are genuinely representative of a real audit.
SEED_FINDINGS: list[dict] = [
    {"category": "Web Security", "severity": "critical", "issue": "SSL certificate has expired.",
     "recommendation": "Renew the SSL certificate immediately and verify auto-renewal is configured."},
    {"category": "Web Security", "severity": "critical",
     "issue": "SSL certificate is self-signed or issued by an untrusted certificate authority.",
     "recommendation": "Replace it with a certificate from a trusted CA (e.g. Let's Encrypt, DigiCert)."},
    {"category": "Web Security", "severity": "warning", "issue": "Missing Content-Security-Policy header.",
     "recommendation": "Implement a Content-Security-Policy to mitigate XSS and data-injection attacks."},
    {"category": "Web Security", "severity": "warning",
     "issue": "Missing X-Frame-Options or frame-ancestors CSP directive.",
     "recommendation": "Add X-Frame-Options: DENY or SAMEORIGIN to prevent clickjacking."},
    {"category": "Web Security", "severity": "warning",
     "issue": "Missing Strict-Transport-Security (HSTS) header.",
     "recommendation": "Add a Strict-Transport-Security header to force HTTPS on all future visits."},
    {"category": "On-Page Content", "severity": "critical", "issue": "Missing meta description.",
     "recommendation": "Add a unique meta description of 120-160 characters summarizing the page."},
    {"category": "On-Page Content", "severity": "critical", "issue": "Multiple H1 tags found on the page.",
     "recommendation": "Use exactly one H1 tag per page, with remaining headings as H2/H3 in logical order."},
    {"category": "On-Page Content", "severity": "warning",
     "issue": "Title tag is too long or too short for optimal display in search results.",
     "recommendation": "Rewrite the title tag to be 50-60 characters, front-loading the primary keyword."},
    {"category": "On-Page Content", "severity": "warning", "issue": "No Open Graph tags found for social sharing.",
     "recommendation": "Add og:title, og:description, and og:image tags for link-preview control."},
    {"category": "On-Page Content", "severity": "warning", "issue": "Thin content -- page has very few words.",
     "recommendation": "Expand the page with substantive, unique content relevant to the target keyword."},
    {"category": "Technical SEO", "severity": "critical", "issue": "No robots.txt file found.",
     "recommendation": "Add a robots.txt file specifying crawl rules and a sitemap reference."},
    {"category": "Technical SEO", "severity": "warning", "issue": "No sitemap.xml file found.",
     "recommendation": "Generate and submit an XML sitemap to search engines via Search Console."},
    {"category": "Technical SEO", "severity": "warning", "issue": "Images are missing alt text.",
     "recommendation": "Add descriptive alt text to informative images; empty alt for purely decorative ones."},
    {"category": "Technical SEO", "severity": "warning",
     "issue": "Canonical tag is missing or points to the wrong URL.",
     "recommendation": "Add a self-referencing canonical tag pointing to the correct preferred URL."},
    {"category": "Page Speed", "severity": "critical", "issue": "Largest Contentful Paint (LCP) exceeds 2500ms.",
     "recommendation": "Optimize the largest above-the-fold image/text block and improve server response time."},
    {"category": "Page Speed", "severity": "warning", "issue": "High Cumulative Layout Shift (CLS).",
     "recommendation": "Reserve explicit width/height for images and ads; avoid late-injected content shifting layout."},
    {"category": "Page Speed", "severity": "warning", "issue": "High Total Blocking Time (TBT).",
     "recommendation": "Minify and defer non-critical JavaScript, and split large bundles."},
    {"category": "Link Health", "severity": "critical", "issue": "Broken internal or external links found.",
     "recommendation": "Fix or remove broken links; set up periodic automated link checking."},
    {"category": "Link Health", "severity": "warning",
     "issue": "High failure rate across sampled links, possibly due to bot-blocking.",
     "recommendation": "Manually verify a sample of failed links -- a high failure rate often means the site is "
                        "blocking automated requests, not that the links are genuinely broken."},
    {"category": "Accessibility", "severity": "critical", "issue": "Images do not have alt attributes.",
     "recommendation": "Add descriptive alt attributes to informative images for screen reader users."},
    {"category": "Accessibility", "severity": "warning",
     "issue": "Insufficient color contrast between text and background.",
     "recommendation": "Increase the contrast ratio to at least 4.5:1 for normal text per WCAG AA."},
    {"category": "Accessibility", "severity": "warning", "issue": "Buttons or links do not have an accessible name.",
     "recommendation": "Add descriptive text, aria-label, or visually-hidden text for screen readers."},
    {"category": "Accessibility", "severity": "warning",
     "issue": "Heading elements are not in a sequentially-descending order.",
     "recommendation": "Reorder headings so each level follows logically without skipping levels."},
    {"category": "Best Practices", "severity": "warning",
     "issue": "Page includes a JavaScript library with a known security vulnerability.",
     "recommendation": "Upgrade the library to a patched version or replace it with a maintained alternative."},
    {"category": "Best Practices", "severity": "warning",
     "issue": "Browser console errors were logged while loading the page.",
     "recommendation": "Investigate and resolve the JavaScript errors logged to the console."},
    {"category": "Best Practices", "severity": "good",
     "issue": "No deprecated APIs or known-vulnerable libraries detected.",
     "recommendation": "No action needed -- continue monitoring dependencies for new vulnerabilities."},
]
for _entry in SEED_FINDINGS:
    _entry["source"] = "seed"
    _entry.setdefault("domain", None)
    _entry.setdefault("timestamp", None)


def _findings_from_report(report: dict) -> list[dict]:
    """Flatten one stored report into individual finding entries, tagged
    with where they came from."""
    entries = []
    for cat in report.get("categories", []):
        if not isinstance(cat, dict):
            continue
        for f in cat.get("findings", []):
            entries.append({
                "category": cat.get("name"),
                "severity": f.get("severity"),
                "issue": f.get("issue", ""),
                "recommendation": f.get("recommendation", ""),
                "source": "real_audit",
                "domain": report.get("_domain"),
                "timestamp": report.get("_timestamp"),
            })
    return entries


def load_real_findings() -> list[dict]:
    """Every individual finding from every stored audit, flattened."""
    entries = []
    for report in memory.get_all_full_audits():
        entries.extend(_findings_from_report(report))
    return entries


def build_corpus(include_seed: bool = True, include_real: bool = True) -> list[dict]:
    """Combine the seed knowledge base and real audit history findings into
    one searchable corpus. Every entry keeps its "source" tag."""
    corpus: list[dict] = []
    if include_seed:
        corpus.extend(dict(e) for e in SEED_FINDINGS)
    if include_real:
        corpus.extend(load_real_findings())
    return corpus


def _entry_text(entry: dict) -> str:
    return f"{entry.get('issue', '')} {entry.get('recommendation', '')}".strip()


@dataclass
class FindingsIndex:
    corpus: list[dict]
    backend: str  # "tfidf" or "embedding"
    vectorizer: object = None
    embedder: object = None
    matrix: object = None


def _build_tfidf_index(corpus: list[dict]) -> FindingsIndex:
    texts = [_entry_text(e) for e in corpus]
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    matrix = vectorizer.fit_transform(texts)
    return FindingsIndex(corpus=corpus, backend="tfidf", vectorizer=vectorizer, matrix=matrix)


def _try_build_embedding_index(corpus: list[dict], log_fn: Callable[[str], None]) -> FindingsIndex | None:
    """Attempt to build a real sentence-embedding index. Returns None
    (never raises) if sentence-transformers isn't installed or the model
    can't be loaded (e.g. no network to download it) -- caller falls back
    to TF-IDF in that case."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log_fn("  -> sentence-transformers not installed -- falling back to TF-IDF. "
               "Install it (`pip install sentence-transformers`) for real semantic embeddings.")
        return None

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [_entry_text(e) for e in corpus]
        embeddings = model.encode(texts, normalize_embeddings=True)
    except Exception as e:
        log_fn(f"  -> Could not load/run the embedding model ({e}) -- falling back to TF-IDF.")
        return None

    return FindingsIndex(corpus=corpus, backend="embedding", embedder=model, matrix=np.asarray(embeddings))


def build_index(
    backend: str = "tfidf",
    include_seed: bool = True,
    include_real: bool = True,
    log_fn: Callable[[str], None] | None = None,
) -> FindingsIndex:
    """backend: "tfidf" (default, always available) or "embedding" (tries
    real sentence embeddings, falls back to TF-IDF automatically -- see
    module docstring)."""
    log_fn = log_fn or (lambda msg: None)
    corpus = build_corpus(include_seed=include_seed, include_real=include_real)
    if not corpus:
        raise ValueError("No findings available to index (seed and real history both empty/disabled).")

    if backend == "embedding":
        index = _try_build_embedding_index(corpus, log_fn)
        if index is not None:
            return index
        # fall through to TF-IDF

    return _build_tfidf_index(corpus)


def search(index: FindingsIndex, query: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    """Return the top_k most similar findings to `query`, each with a
    similarity score and full source metadata."""
    if index.backend == "embedding":
        query_vec = index.embedder.encode([query], normalize_embeddings=True)
    else:
        query_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(query_vec, index.matrix)[0]

    ranked = sorted(range(len(index.corpus)), key=lambda i: sims[i], reverse=True)

    results = []
    for i in ranked:
        entry = index.corpus[i]
        if category and entry.get("category") != category:
            continue
        results.append({
            "similarity": round(float(sims[i]), 4),
            "category": entry.get("category"),
            "severity": entry.get("severity"),
            "issue": entry.get("issue"),
            "recommendation": entry.get("recommendation"),
            "source": entry.get("source"),
            "domain": entry.get("domain"),
            "timestamp": entry.get("timestamp"),
        })
        if len(results) >= top_k:
            break

    return results


def print_search_results(query: str, results: list[dict], backend: str) -> None:
    print("\n" + "=" * 64)
    print(f"FINDINGS SIMILARITY SEARCH  (backend: {backend})")
    print("=" * 64)
    print(f'Query: "{query}"\n')
    if not results:
        print("No matching findings.")
    for r in results:
        source_label = "seed reference" if r["source"] == "seed" else f"real audit: {r['domain']} ({r['timestamp']})"
        print(f"  [{r['similarity']:.3f}] ({r['category']}, {r['severity']}) -- {source_label}")
        print(f"    Issue:          {r['issue']}")
        print(f"    Recommendation: {r['recommendation']}")
        print()
    print("=" * 64 + "\n")