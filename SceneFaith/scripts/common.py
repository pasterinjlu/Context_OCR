"""Frozen response parsing, prompts, and L/C/O scoring."""
import re
from typing import Any

def normalize_prediction(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    value = lines[0] if lines else cleaned
    value = re.sub(r"^(?:answer|output|text|result)\s*:\s*", "", value, flags=re.I)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`“”‘’":
        value = value[1:-1].strip()
    return value

def normalize_output(value: Any) -> str:
    """Apply the exact-match normalization declared in the paper."""
    return " ".join(str(value or "").strip().casefold().split())

PROMPTS = {
    "neutral": (
        "Read the text enclosed by the single red rectangular outline in the image. "
        "Return only that text and nothing else."
    ),
    "exact_copy": (
        "Copy the text enclosed by the single red rectangular outline exactly as it "
        "appears, character by character. Preserve all spelling errors, capitalization, "
        "digits, punctuation, and spaces, even if the text looks incorrect. Do not infer "
        "or correct the intended word. Do not output chain-of-thought, reasoning, analysis, "
        "or explanation. Return only the copied text and nothing else."
    ),
    "context_blind": (
        "Copy the text enclosed by the single red rectangular outline exactly as it appears. "
        "Use only the visible characters inside the red rectangle. Ignore all surrounding "
        "objects, background information, and scene semantics. Preserve every spelling "
        "error, digit, punctuation mark, capitalization, and space. Do not output "
        "chain-of-thought, reasoning, analysis, or explanation. Return only the copied text "
        "and nothing else."
    ),
    "correction_first": (
        "Infer the intended word represented by the text inside the single red rectangular "
        "outline. Use the surrounding scene when helpful and silently correct apparent "
        "spelling errors. Do not output chain-of-thought, reasoning, analysis, or explanation. "
        "Return only the corrected word and nothing else."
    ),
}

QWEN_NEUTRAL = (
    "Read the text enclosed by the single red rectangular outline in the image. "
    "Do not output chain-of-thought, reasoning, analysis, or explanation. "
    "Return only the text itself and nothing else."
)

def prompt_for(model: str, variant: str) -> str:
    if variant == "neutral" and "qwen" in model.casefold():
        return QWEN_NEUTRAL
    return PROMPTS[variant]

def classify(raw, row):
    prediction = normalize_prediction(raw)
    value = normalize_output(prediction)
    literal, canonical = (normalize_output(row[k]) for k in ("observed_text", "canonical_text"))
    if literal == canonical:
        raise ValueError("Literal and canonical labels must differ")
    return prediction, "L" if value == literal else "C" if value == canonical else "O"
