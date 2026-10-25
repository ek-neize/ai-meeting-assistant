"""Core pipeline for the AI Meeting Assistant.

audio -> Whisper transcript -> ASCII clean-up -> terminology correction (LLM)
      -> meeting minutes and task list (LLM)

Heavy dependencies (torch, transformers, langchain-openai) are imported lazily
so this module can be imported and unit-tested without them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

DEFAULT_WHISPER_MODEL = "openai/whisper-medium"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

TERMINOLOGY_SYSTEM_PROMPT = """You are an assistant specializing in financial products. \
You process transcripts of earnings calls and meetings so that every reference to a \
financial product or common financial term is in the correct format.

Rules:
- For terms usually abbreviated as acronyms, spell out the full term followed by the \
acronym in parentheses. Examples: '401k' -> '401(k) retirement savings plan', \
'HSA' -> 'Health Savings Account (HSA)', 'ROA' -> 'Return on Assets (ROA)', \
'VaR' -> 'Value at Risk (VaR)', 'PB' -> 'Price to Book (PB) ratio'.
- Convert spoken numbers that name financial products into numeric form followed by the \
product name. Examples: 'five two nine' -> '529 (Education Savings Plan)', \
'four zero one k' -> '401(k) (Retirement Savings Plan)'.
- Some acronyms are ambiguous (e.g. 'LTV' can mean 'Loan to Value' or 'Lifetime Value'). \
Use the context to pick the right meaning.
- Leave spelled-out figures that are not financial products (like 'twenty three percent') unchanged.
- Fix obvious transcription errors and punctuation, but do not change the meaning.

Output the corrected transcript, then a section headed 'Changes:' listing each term you changed."""

MINUTES_TEMPLATE = """Generate meeting minutes and a list of tasks based on the provided context.
Format the answer in Markdown using exactly these sections:

## Meeting Minutes
### Key points discussed
### Decisions made

## Task List
A Markdown table with columns: Task | Owner | Deadline.
Use names and dates only if they are stated in the context; otherwise write "Unassigned" or "TBD".
Do not invent people or dates.

Context:
{context}
"""


def build_terminology_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", TERMINOLOGY_SYSTEM_PROMPT), ("human", "{transcript}")]
    )


def build_minutes_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(MINUTES_TEMPLATE)


# --------------------------------------------------------------------------- #
# Chains
# --------------------------------------------------------------------------- #

def build_terminology_chain(llm: Runnable) -> Runnable:
    """str transcript -> str corrected transcript."""
    return {"transcript": RunnablePassthrough()} | build_terminology_prompt() | llm | StrOutputParser()


def build_minutes_chain(llm: Runnable) -> Runnable:
    """str transcript -> str Markdown minutes and task list."""
    return {"context": RunnablePassthrough()} | build_minutes_prompt() | llm | StrOutputParser()


def make_llm(temperature: float, top_p: float = 1.0, max_tokens: int | None = None):
    """Create an OpenAI chat model. Reads OPENAI_API_KEY and OPENAI_MODEL from the environment."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=2)
def get_transcriber(model_name: str):
    """Load a Whisper pipeline once and reuse it. Uses a GPU when one is available."""
    import torch
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=0 if torch.cuda.is_available() else -1,
    )


def transcribe(audio_path: str, model_name: str | None = None) -> str:
    model_name = model_name or os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
    return get_transcriber(model_name)(audio_path, batch_size=8)["text"].strip()


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #

def remove_non_ascii(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) < 128)


@dataclass
class MeetingResult:
    raw_transcript: str
    corrected_transcript: str
    minutes: str

    def to_markdown(self) -> str:
        return (
            f"{self.minutes.strip()}\n\n---\n\n"
            f"## Corrected transcript\n\n{self.corrected_transcript.strip()}\n\n"
            f"## Raw transcript\n\n{self.raw_transcript.strip()}\n"
        )


def analyze_transcript(
    raw_transcript: str,
    terminology_chain: Runnable,
    minutes_chain: Runnable,
) -> MeetingResult:
    """Run the two LLM steps on a transcript. Chains are injected so they can be faked in tests."""
    cleaned = remove_non_ascii(raw_transcript)
    corrected = terminology_chain.invoke(cleaned)
    minutes = minutes_chain.invoke(corrected)
    return MeetingResult(raw_transcript, corrected, minutes)
