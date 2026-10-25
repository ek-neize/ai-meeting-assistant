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

# Longest recording accepted, in seconds. Keeps a public demo fast and cheap.
MAX_AUDIO_SECONDS = int(os.getenv("MAX_AUDIO_SECONDS", "600"))

# Visual theme: soft indigo-to-sky gradient page, white rounded cards, gradient button.
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #eef2ff 0%, #e0f2fe 50%, #f5f3ff 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)",
    block_background_fill="#ffffff",
    block_background_fill_dark="#1e293b",
    block_border_width="0px",
    block_radius="16px",
    block_shadow="0 4px 20px rgba(79, 70, 229, 0.10)",
    button_primary_background_fill="linear-gradient(90deg, #4f46e5, #0ea5e9)",
    button_primary_background_fill_hover="linear-gradient(90deg, #4338ca, #0284c7)",
    button_primary_text_color="#ffffff",
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
    color: #ffffff; padding: 28px 32px; border-radius: 18px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
}
.hero h1 { margin: 0 0 6px; font-size: 2rem; font-weight: 700; color: #ffffff; }
.hero p { margin: 0; font-size: 1.02rem; color: rgba(255, 255, 255, 0.92); }
.hero .chips span {
    display: inline-block; margin: 14px 8px 0 0; padding: 4px 12px;
    border-radius: 999px; background: rgba(255, 255, 255, 0.18); font-size: 0.85rem;
    color: #ffffff;
}
.hero-wrap { background: transparent !important; box-shadow: none !important; padding: 0 !important; }
.results-card {
    background: var(--block-background-fill); border-radius: 16px;
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.10); padding: 8px 18px 18px !important;
    min-height: 340px;
}
"""

HEADER_HTML = """
<div class="hero">
  <h1>AI Meeting Assistant</h1>
  <p>Upload or record a meeting. It is transcribed with Whisper, financial terms are corrected
  by an OpenAI model, and you get meeting minutes and a task list to download.</p>
  <div class="chips"><span>Whisper</span><span>LangChain</span><span>OpenAI</span><span>Gradio</span></div>
</div>
"""


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


with gr.Blocks(title="AI Meeting Assistant", theme=THEME, css=CSS) as demo:
    gr.HTML(HEADER_HTML, elem_classes="hero-wrap")

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                max_length=MAX_AUDIO_SECONDS,
                label=f"Meeting audio (.wav, .mp3, .m4a; up to {MAX_AUDIO_SECONDS // 60} min)",
            )
            run_button = gr.Button("Generate minutes", variant="primary")
            download_button = gr.DownloadButton("Download report (.md)", visible=False)

        with gr.Column(scale=2, elem_classes="results-card"):
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
    # On Hugging Face Spaces (SPACE_ID is set there), load Whisper at startup
    # so the first visitor doesn't wait for the model download.
    if os.getenv("SPACE_ID"):
        from meeting_assistant import DEFAULT_WHISPER_MODEL, get_transcriber

        get_transcriber(os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL))

    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
