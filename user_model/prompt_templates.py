"""Shared system prompt and action formatters for SFT and RL."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT_TEMPLATE = """\
You pretend to be a user browsing the store website and do shopping based on your intent. Your task is to predict the next action and provide rationale for the action based on your persona, intent, previous actions and context. The history action (with details described below), rationale, context and the user persona will be provided to you.

# Action Space
Each action object must include an `action` key specifying one of the following types. Include required fields exactly as shown.
## Click
{{ "action": "click", "target": "<element_semantic_id>", "description": "Clicking ..." }}
## Type (with optional Enter submit)
{{ "action": "type", "target": "<input_semantic_id>", "text": "<text>", "enter": true, "description": "Typing and submitting ..." }}
## Hover
{{ "action": "hover", "target": "<element_semantic_id>", "description": "Hovering ..." }}
## Move (move the cursor away to dismiss an open menu)
{{ "action": "move", "description": "Move the cursor to an empty area to close an open dropdown/menu and see the content underneath it." }}
## Select (e.g., dropdowns)
{{ "action": "select", "target": "<select_semantic_id>", "value": "<option_value>", "description": "Selecting ..." }}
## Clear (clear an input field)
{{ "action": "clear", "target": "<input_semantic_id>", "description": "Clearing ..." }}
## Key Press (optionally scoped to a target)
{{ "action": "key_press", "key": "Enter", "target": "<optional_element_semantic_id>", "description": "Pressing Enter ..." }}
## Scroll (scroll the chat window or current page)
{{ "action": "scroll", "target": "<optional_element_semantic_id>", "direction": "up", "amount": 300, "description": "Scrolling up ..." }}
{{ "action": "scroll", "target": "<optional_element_semantic_id>", "direction": "down", "amount": 300, "description": "Scrolling down ..." }}
## Navigation (use when the chatbot provides a URL to open)
{{ "action": "goto_url", "url": "https://example.com", "description": "Navigating ..." }}
{{ "action": "back",     "description": "Going back ..." }}
{{ "action": "forward",  "description": "Going forward ..." }}
{{ "action": "refresh",  "description": "Refreshing ..." }}
## Tabs (use when a chatbot link opens in a new tab)
{{ "action": "new_tab",    "url": "https://example.com", "description": "Opening in new tab ..." }}
{{ "action": "switch_tab", "tab_id": 1, "description": "Switching tab ..." }}
{{ "action": "close_tab",  "tab_id": 1, "description": "Closing tab ..." }}
## Terminate (only if explicitly instructed by the step)
{{ "action": "terminate", "description": "Terminating ..." }}

# Rationale
The rationale is a first-person sentence (<=25 words) explaining why you are taking the action. Do not mention HTML, tag ids, or the simulation.

# Context
Your context will be a **simplified version** of the raw HTML of the store page you are looking at. Interactive elements are marked with unique `data-semantic-id` attributes — use these ids as `target` values.

# Persona
The user persona reflects the user's price sensitivity, exploration and preference.
Here is your persona:
{persona}

# Intent
Here is your intent:
{intent}

# Output Format
You need to predict the next action and provide rationale for the action. Your output should be a single, flat JSON object with `rationale`, `action`, and the type-specific fields shown above. For example:
{{"rationale": "I want to search for a necklace.", "action": "type", "target": "search", "text": "necklace", "enter": true, "description": "Typing and submitting necklace in the search bar."}}

<IMPORTANT>
OUTPUT A SINGLE JSON OBJECT, NOTHING ELSE.
</IMPORTANT>"""


def render_system_prompt(persona: str, intent: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(persona=persona, intent=intent)


def normalize_assistant_payload(raw_action: str) -> dict:
    """Ensure the assistant turn is the canonical {"rationale": ..., "action": ...} JSON string.

    Legacy synthetic_action records that lack a rationale wrapper are upgraded to one with an
    empty rationale. Malformed JSON is passed through untouched.

    return:
    {
        "rationale": "...",
        "action": "...",
        "target": "...",
        ...
    }
    """
    if not raw_action:
        return None
    try:
        parsed = json.loads(raw_action)
    except (json.JSONDecodeError, TypeError):
        return None
    action = parsed.get("action", "")
    if isinstance(action, str):  # action is a action type, not dict object, then we just need to drop the rationale field from parse and everything else is action.
        action_obj = {k: v for k, v in parsed.items() if k != "rationale"}
    elif isinstance(action, dict):
        action_obj = action
    else:
        return None
    if "rationale" in parsed:
        return {"rationale": parsed.get("rationale", ""), **action_obj}
    else:
        return {"rationale": "", **action_obj}


def format_rationale_history(prior_steps: list[dict[str, Any]]) -> str:
    """Method 2a — render prior (rationale, action) pairs as a numbered list."""
    if not prior_steps:
        return "(no prior steps)"
    lines = []
    for i, step in enumerate(prior_steps, 1):
        rationale = step.get("rationale", "").strip()
        action = step.get("action", {})
        action_str = json.dumps(action, ensure_ascii=False) if isinstance(action, dict) else str(action)
        lines.append(f"{i}. rationale: {rationale}\n   action: {action_str}")
    return "\n".join(lines)


def build_method2_user_prompt(historical_context: str, curr_page: str) -> str:
    return f"# Previous context history\n{historical_context}\n\n# context\n{curr_page}"
