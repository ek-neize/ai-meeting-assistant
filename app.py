"""Gradio web app for the AI Meeting Assistant."""

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from meeting_assistant import (
    analyze_transcript,
    build_minutes_chain,
    build_terminology_chain,
    make_llm,
    remove_non_ascii,
    transcribe,
)

load_dotenv()

OUTPUT_DIR = Path("outputs")


@lru_cache(maxsize=1)
def get_chains():
    """Build the LLM chains once, on first use (after the API key check)."""
    terminology = build_terminology_chain(make_llm(temperature=0.2, top_p=0.6))
    minutes = build_minutes_chain(make_llm(temperature=0.5, max_tokens=1024))
    return terminology, minutes


def save_report(markdown: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"meeting_minutes_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def process_meeting(audio_path, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file first.")
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")

    progress(0.1, desc="Transcribing audio with Whisper...")
    raw = transcribe(audio_path)
    if not remove_non_ascii(raw).strip():
        raise gr.Error("No speech was detected in this recording.")

    progress(0.6, desc="Correcting terminology and writing minutes...")
    terminology_chain, minutes_chain = get_chains()
    try:
        result = analyze_transcript(raw, terminology_chain, minutes_chain)
    except Exception as exc:  # surface API errors (bad key, no credit, wrong model) in the UI
        raise gr.Error(f"OpenAI request failed: {exc}") from exc

    report_path = save_report(result.to_markdown())
    return (
        result.minutes,
        result.corrected_transcript,
        result.raw_transcript,
        gr.DownloadButton(value=report_path, visible=True),
    )


with gr.Blocks(title="AI Meeting Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI Meeting Assistant\n"
        "Upload or record a meeting. The app transcribes it with **Whisper**, corrects "
        "financial terminology with an **OpenAI** model, and produces **meeting minutes "
        "and a task list** you can download."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Meeting audio (.wav, .mp3, .m4a...)",
            )
            run_button = gr.Button("Generate minutes", variant="primary")
            download_button = gr.DownloadButton("Download report (.md)", visible=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Minutes & tasks"):
                    minutes_output = gr.Markdown("_Results will appear here._")
                with gr.Tab("Corrected transcript"):
                    corrected_output = gr.Textbox(lines=18, show_label=False, show_copy_button=True)
                with gr.Tab("Raw transcript"):
                    raw_output = gr.Textbox(lines=18, show_label=False, show_copy_button=True)

    run_button.click(
        process_meeting,
        inputs=audio_input,
        outputs=[minutes_output, corrected_output, raw_output, download_button],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
