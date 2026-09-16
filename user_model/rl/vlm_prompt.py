"""Prompt construction for the VLM student (Qwen2.5-VL).

The student sees the current observation (simplified HTML + page screenshot) and
its previous fine-grained actions, and must emit ONE fine-grained action. Unlike
the teacher prompt (``user_model/prompt_templates.py``), the student is NOT given
the high-level semantic action — it must infer the intent itself, which is the
signal GRPO rewards (Mar-27 spec).
"""

from __future__ import annotations

import json
from typing import Any

# Student system prompt: teacher prompt minus the given semantic action.
STUDENT_SYSTEM_PROMPT = """\
You are simulating the behavior of a realistic human buyer on an e-commerce website.
You are given the current page (a screenshot and a simplified HTML snapshot) and your
previous actions. Decide the single next concrete UI action a real buyer would take,
driven by your own implicit intent — you are NOT told what high-level step to perform.

# Action Space
Return exactly one action object with an `action` key set to one of:
click, type, hover, select, clear, key_press, scroll, back, terminate.
Interactive elements are addressed by their `data-semantic-id` (use it as `target`).

## Click        {{ "action": "click", "target": "<id>", "description": "..." }}
## Type         {{ "action": "type", "target": "<id>", "text": "<text>", "enter": true, "description": "..." }}
## Select       {{ "action": "select", "target": "<id>", "value": "<option>", "description": "..." }}
## Scroll       {{ "action": "scroll", "direction": "down", "amount": 300, "description": "..." }}
## Back         {{ "action": "back", "description": "..." }}
## Terminate    {{ "action": "terminate", "description": "..." }}

# Rationale
A first-person sentence (<=25 words) for why you take the action. Never mention HTML,
ids, or the simulation.

# Output Format
Output a SINGLE flat JSON object with `rationale`, `action`, and the type-specific
fields. Example:
{{"rationale": "I want a warm winter hat.", "action": "click", "target": "camel_brown_beanie", "description": "Opening the beanie product page."}}

<IMPORTANT>OUTPUT A SINGLE JSON OBJECT, NOTHING ELSE.</IMPORTANT>"""

# A single {image} content part is inserted for VL models; the HTML follows as text.
IMAGE_PLACEHOLDER = "<image>"


def format_action_history(prior_actions: list[dict[str, Any]]) -> str:
    """Render the previous fine-grained actions as a compact numbered list."""
    if not prior_actions:
        return "(no prior steps)"
    lines = []
    for i, act in enumerate(prior_actions, 1):
        lines.append(f"{i}. {json.dumps(act, ensure_ascii=False)}")
    return "\n".join(lines)


def build_user_text(history: str, html: str) -> str:
    """Text half of the user turn (the image is attached as a separate part)."""
    return (
        f"# Previous actions\n{history}\n\n"
        f"# Current page (screenshot above)\n{html}"
    )


def build_prompt_messages(
    history: str, html: str, has_image: bool = True
) -> list[dict[str, Any]]:
    """Build a chat-format ``prompt`` for GRPOTrainer.

    For VL models the user content is a list of parts: one image part followed by
    the text part. When ``has_image`` is False (text-only fallback / smoke), the
    user content is a plain string.
    """
    text = build_user_text(history, html)
    if has_image:
        # Uniform list-of-parts content (image case) so the dataset serializes
        # cleanly to Arrow — a column can't mix string and list content.
        return [
            {"role": "system", "content": [{"type": "text", "text": STUDENT_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
