"""Tests for agent/critic.py::reflect_and_revise.

PROJECT_HANDOFF.md flags this as the single most regression-prone function in
the codebase: on the last allowed reflection round, if the critic still
rejects the draft, the loop must stop there rather than calling the
synthesizer one more time with nothing left to review it. That exact fix was
accidentally reverted once already. These tests exist specifically to catch
that regression if it ever happens again.
"""
from __future__ import annotations

from agent import critic


def _approved_review():
    return {"approved": True, "issues": [], "instructions_for_revision": ""}


def _rejected_review(issue="needs work"):
    return {"approved": False, "issues": [issue], "instructions_for_revision": "fix it"}


class TestReflectAndRevise:
    def test_approved_on_first_round_makes_exactly_one_synthesizer_and_one_critic_call(self, monkeypatch):
        synth_calls = []
        critique_calls = []
        monkeypatch.setattr(critic, "run_synthesizer", lambda *a, **kw: synth_calls.append(1) or {"draft": "v1"})
        monkeypatch.setattr(critic, "critique", lambda *a, **kw: critique_calls.append(1) or _approved_review())

        draft, log = critic.reflect_and_revise("https://example.com", {}, None)

        assert len(synth_calls) == 1
        assert len(critique_calls) == 1
        assert draft == {"draft": "v1"}
        assert log[-1]["review"]["approved"] is True

    def test_rejected_then_approved_makes_two_synthesizer_calls(self, monkeypatch):
        synth_calls = []
        critique_calls = []

        def fake_synth(*a, **kw):
            synth_calls.append(kw.get("key_index"))
            return {"draft": f"v{len(synth_calls)}"}

        def fake_critique(*a, **kw):
            critique_calls.append(1)
            return _rejected_review() if len(critique_calls) == 1 else _approved_review()

        monkeypatch.setattr(critic, "run_synthesizer", fake_synth)
        monkeypatch.setattr(critic, "critique", fake_critique)

        draft, log = critic.reflect_and_revise("https://example.com", {}, None)

        assert len(synth_calls) == 2
        assert len(critique_calls) == 2
        assert draft == {"draft": "v2"}
        assert log[-1]["review"]["approved"] is True

    def test_rejected_on_final_round_stops_without_extra_unreviewed_synthesis(self, monkeypatch):
        """THE regression test. MAX_REFLECTION_ROUNDS=2 by default: synthesizer
        runs once, critic rejects (round 1), synthesizer revises, critic
        rejects again (round 2, the final round) -> loop must stop here.
        Exactly 2 synthesizer calls and 2 critic calls -- never a 3rd,
        unreviewed synthesizer call."""
        synth_calls = []
        critique_calls = []
        monkeypatch.setattr(critic, "run_synthesizer", lambda *a, **kw: synth_calls.append(1) or {"draft": f"v{len(synth_calls)}"})
        monkeypatch.setattr(critic, "critique", lambda *a, **kw: critique_calls.append(1) or _rejected_review())
        monkeypatch.setattr(critic, "MAX_REFLECTION_ROUNDS", 2)

        draft, log = critic.reflect_and_revise("https://example.com", {}, None)

        assert len(synth_calls) == 2, "must not call the synthesizer a 3rd, unreviewed time"
        assert len(critique_calls) == 2
        assert draft == {"draft": "v2"}
        assert log[-1]["review"]["approved"] is False

    def test_returned_draft_matches_last_reflection_log_entry(self, monkeypatch):
        """The draft returned must be exactly the one the last reflection_log
        entry describes -- not a further, unreviewed revision of it."""
        monkeypatch.setattr(critic, "run_synthesizer", lambda *a, **kw: {"draft": "final"})
        monkeypatch.setattr(critic, "critique", lambda *a, **kw: _rejected_review("still bad"))
        monkeypatch.setattr(critic, "MAX_REFLECTION_ROUNDS", 1)

        draft, log = critic.reflect_and_revise("https://example.com", {}, None)

        assert len(log) == 1
        assert draft == {"draft": "final"}

    def test_key_index_advances_on_every_call_not_reset_per_round(self, monkeypatch):
        seen_key_indices = []

        def fake_synth(*a, **kw):
            seen_key_indices.append(("synth", kw.get("key_index")))
            return {"draft": "d"}

        def fake_critique(*a, **kw):
            idx = kw.get("key_index")
            seen_key_indices.append(("critic", idx))
            return _approved_review()

        monkeypatch.setattr(critic, "run_synthesizer", fake_synth)
        monkeypatch.setattr(critic, "critique", fake_critique)

        critic.reflect_and_revise("https://example.com", {}, None, starting_key_index=3)

        # synthesizer gets key 3, critic gets key 4 -- strictly increasing,
        # never resetting back to the starting index mid-run.
        assert seen_key_indices[0] == ("synth", 3)
        assert seen_key_indices[1] == ("critic", 4)

    def test_critic_feedback_is_fed_back_into_revised_synthesizer_call(self, monkeypatch):
        captured_reports = []

        def fake_synth(url, specialist_reports, previous_audit, **kw):
            captured_reports.append(dict(specialist_reports))
            return {"draft": f"v{len(captured_reports)}"}

        calls = {"n": 0}

        def fake_critique(*a, **kw):
            calls["n"] += 1
            return _rejected_review("bad math") if calls["n"] == 1 else _approved_review()

        monkeypatch.setattr(critic, "run_synthesizer", fake_synth)
        monkeypatch.setattr(critic, "critique", fake_critique)

        critic.reflect_and_revise("https://example.com", {"technical_seo": {"score": 90}}, None)

        # Second synthesizer call should have received the critic's feedback.
        assert "_critic_feedback" in captured_reports[1]
        assert captured_reports[1]["_critic_feedback"]["issues"] == ["bad math"]
        # First call should NOT have had feedback (nothing to feed back yet).
        assert "_critic_feedback" not in captured_reports[0]
