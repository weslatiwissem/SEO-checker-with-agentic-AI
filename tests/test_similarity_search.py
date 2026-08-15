from __future__ import annotations

from agent import similarity_search as ss


def _sample_report(domain="example.com", timestamp="2026-08-01T00:00:00", findings=None):
    if findings is None:
        findings = [{"severity": "critical", "issue": "Custom finding.", "recommendation": "Custom fix."}]
    return {
        "_domain": domain, "_timestamp": timestamp,
        "categories": [{"name": "Web Security", "findings": findings}],
    }


class TestSeedFindings:
    def test_every_entry_has_required_fields(self):
        for entry in ss.SEED_FINDINGS:
            assert entry.get("category")
            assert entry.get("severity") in {"good", "warning", "critical"}
            assert entry.get("issue")
            assert entry.get("recommendation")

    def test_every_entry_tagged_as_seed_source(self):
        assert all(e["source"] == "seed" for e in ss.SEED_FINDINGS)

    def test_covers_multiple_categories(self):
        categories = {e["category"] for e in ss.SEED_FINDINGS}
        assert len(categories) >= 5  # meaningfully spans the specialist categories

    def test_has_a_reasonable_number_of_entries(self):
        assert len(ss.SEED_FINDINGS) >= 15


class TestFindingsFromReport:
    def test_extracts_findings_with_metadata(self):
        report = _sample_report(domain="a.com", timestamp="t1")
        entries = ss._findings_from_report(report)
        assert len(entries) == 1
        assert entries[0]["category"] == "Web Security"
        assert entries[0]["source"] == "real_audit"
        assert entries[0]["domain"] == "a.com"
        assert entries[0]["timestamp"] == "t1"

    def test_handles_report_with_no_categories(self):
        assert ss._findings_from_report({"categories": []}) == []

    def test_handles_multiple_findings_across_categories(self):
        report = {
            "_domain": "a.com", "_timestamp": "t1",
            "categories": [
                {"name": "Web Security", "findings": [{"severity": "critical", "issue": "x", "recommendation": "y"}]},
                {"name": "Page Speed", "findings": [{"severity": "warning", "issue": "z", "recommendation": "w"}]},
            ],
        }
        entries = ss._findings_from_report(report)
        assert len(entries) == 2
        assert {e["category"] for e in entries} == {"Web Security", "Page Speed"}


class TestLoadRealFindings:
    def test_flattens_across_multiple_reports(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [
            _sample_report(domain="a.com"), _sample_report(domain="b.com"),
        ])
        entries = ss.load_real_findings()
        assert len(entries) == 2
        assert {e["domain"] for e in entries} == {"a.com", "b.com"}

    def test_empty_history_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        assert ss.load_real_findings() == []


class TestBuildCorpus:
    def test_includes_seed_by_default(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        corpus = ss.build_corpus()
        assert len(corpus) == len(ss.SEED_FINDINGS)

    def test_excludes_seed_when_requested(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report()])
        corpus = ss.build_corpus(include_seed=False)
        assert len(corpus) == 1
        assert corpus[0]["source"] == "real_audit"

    def test_excludes_real_when_requested(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report()])
        corpus = ss.build_corpus(include_real=False)
        assert len(corpus) == len(ss.SEED_FINDINGS)
        assert all(e["source"] == "seed" for e in corpus)

    def test_combines_both_by_default(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report()])
        corpus = ss.build_corpus()
        assert len(corpus) == len(ss.SEED_FINDINGS) + 1

    def test_mutating_returned_corpus_does_not_affect_seed_constant(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        corpus = ss.build_corpus()
        corpus[0]["issue"] = "mutated"
        assert ss.SEED_FINDINGS[0]["issue"] != "mutated"


class TestBuildIndexTfidf:
    def test_builds_tfidf_index_by_default(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        index = ss.build_index()
        assert index.backend == "tfidf"
        assert index.vectorizer is not None
        assert index.matrix.shape[0] == len(ss.SEED_FINDINGS)

    def test_raises_on_empty_corpus(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        try:
            ss.build_index(include_seed=False, include_real=False)
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestBuildIndexEmbeddingFallback:
    def test_falls_back_to_tfidf_when_sentence_transformers_missing(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        logs = []
        index = ss.build_index(backend="embedding", log_fn=logs.append)
        assert index.backend == "tfidf"
        assert any("not installed" in msg for msg in logs)

    def test_falls_back_to_tfidf_when_model_load_raises(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])

        class FakeSentenceTransformerModule:
            class SentenceTransformer:
                def __init__(self, *a, **kw):
                    raise RuntimeError("simulated: no network to download model")

        import sys
        monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformerModule)

        logs = []
        index = ss.build_index(backend="embedding", log_fn=logs.append)
        assert index.backend == "tfidf"
        assert any("Could not load" in msg for msg in logs)

    def test_uses_embedding_backend_when_available(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])

        class FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                import numpy as np
                # Deterministic fake embedding: length-based, just needs to be a valid vector per text.
                return np.array([[len(t), 0.0, 1.0] for t in texts], dtype=float)

        class FakeSentenceTransformerModule:
            class SentenceTransformer:
                def __init__(self, *a, **kw):
                    pass

                def encode(self, texts, normalize_embeddings=True):
                    return FakeModel().encode(texts, normalize_embeddings)

        import sys
        monkeypatch.setitem(sys.modules, "sentence_transformers", FakeSentenceTransformerModule)

        index = ss.build_index(backend="embedding")
        assert index.backend == "embedding"
        assert index.embedder is not None


class TestSearch:
    def test_returns_top_k_sorted_descending(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        index = ss.build_index()
        results = ss.search(index, "meta description missing", top_k=3)
        assert len(results) == 3
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_most_relevant_seed_entry_ranks_first(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        index = ss.build_index()
        results = ss.search(index, "the meta description tag is missing from the page", top_k=1)
        assert "meta description" in results[0]["issue"].lower()

    def test_category_filter_restricts_results(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        index = ss.build_index()
        results = ss.search(index, "issue", top_k=5, category="Accessibility")
        assert all(r["category"] == "Accessibility" for r in results)

    def test_real_finding_ranks_above_seed_when_more_relevant(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report(
            findings=[{"severity": "critical", "issue": "Extremely unusual bespoke phrase xyzzycorp cipher issue.",
                       "recommendation": "Fix the xyzzycorp cipher."}],
        )])
        index = ss.build_index()
        results = ss.search(index, "xyzzycorp cipher issue", top_k=1)
        assert results[0]["source"] == "real_audit"

    def test_result_includes_full_source_metadata(self, monkeypatch):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report(domain="a.com", timestamp="t1")])
        index = ss.build_index()
        results = ss.search(index, "custom finding", top_k=1)
        assert results[0]["domain"] == "a.com"
        assert results[0]["timestamp"] == "t1"


class TestPrintSearchResults:
    def test_does_not_raise_with_results(self, monkeypatch, capsys):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [])
        index = ss.build_index()
        results = ss.search(index, "missing meta description", top_k=2)
        ss.print_search_results("missing meta description", results, index.backend)
        captured = capsys.readouterr()
        assert "FINDINGS SIMILARITY SEARCH" in captured.out

    def test_does_not_raise_with_no_results(self, capsys):
        ss.print_search_results("query", [], "tfidf")
        captured = capsys.readouterr()
        assert "No matching findings" in captured.out

    def test_labels_seed_vs_real_audit_source_distinctly(self, monkeypatch, capsys):
        monkeypatch.setattr(ss.memory, "get_all_full_audits", lambda: [_sample_report(domain="a.com")])
        index = ss.build_index()
        results = ss.search(index, "custom finding", top_k=1)
        ss.print_search_results("custom finding", results, index.backend)
        captured = capsys.readouterr()
        assert "real audit: a.com" in captured.out