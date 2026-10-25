"""Unit tests for the pipeline. No API key, GPU or model download needed:
LLMs are replaced with simple fakes."""

from langchain_core.runnables import RunnableLambda

from meeting_assistant import (
    MeetingResult,
    analyze_transcript,
    build_minutes_chain,
    build_minutes_prompt,
    build_terminology_chain,
    remove_non_ascii,
)


def recording_llm(reply: str, seen: list):
    """A fake LLM that records the prompt it receives and returns a fixed reply."""
    def _call(prompt_value):
        seen.append(prompt_value.to_string())
        return reply
    return RunnableLambda(_call)


def test_remove_non_ascii_strips_accents_and_symbols():
    assert remove_non_ascii("café 401k ✓") == "caf 401k "


def test_minutes_prompt_includes_context():
    text = build_minutes_prompt().format(context="Budget approved for Q3")
    assert "Budget approved for Q3" in text
    assert "## Task List" in text


def test_minutes_chain_passes_transcript_as_plain_text():
    # Regression test: passing a dict to a RunnablePassthrough chain nests it,
    # so the prompt would contain "{'context': ...}" instead of the transcript.
    seen = []
    chain = build_minutes_chain(recording_llm("minutes", seen))
    assert chain.invoke("We agreed to launch in May.") == "minutes"
    assert "We agreed to launch in May." in seen[0]
    assert "{'context'" not in seen[0]


def test_terminology_chain_sends_system_rules_and_transcript():
    seen = []
    chain = build_terminology_chain(recording_llm("fixed", seen))
    assert chain.invoke("contribute to your 401k") == "fixed"
    assert "401(k) retirement savings plan" in seen[0]  # rules are in the system prompt
    assert "contribute to your 401k" in seen[0]


def test_analyze_transcript_runs_both_steps_in_order():
    calls = []
    terminology = RunnableLambda(lambda t: calls.append(("terms", t)) or "corrected")
    minutes = RunnableLambda(lambda t: calls.append(("minutes", t)) or "the minutes")

    result = analyze_transcript("raw café text", terminology, minutes)

    assert calls == [("terms", "raw caf text"), ("minutes", "corrected")]
    assert result == MeetingResult("raw café text", "corrected", "the minutes")


def test_report_markdown_contains_all_sections():
    md = MeetingResult("raw", "fixed", "## Meeting Minutes").to_markdown()
    assert md.startswith("## Meeting Minutes")
    assert "## Corrected transcript" in md and "## Raw transcript" in md
