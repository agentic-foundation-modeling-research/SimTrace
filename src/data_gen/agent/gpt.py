import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import tiktoken

# anthropic import for claude computer use
import anthropic
import litellm
# litellm._turn_on_debug()
from anthropic.types.beta import (
    BetaTextBlockParam,
    BetaToolUseBlockParam,
)

from dotenv import load_dotenv

from . import context

# Module-level defaults — overridden at startup by init_from_config(cfg)
_default_provider: str = "openai"
_default_model: str = "gpt-5-mini"
_default_embed_model: str = "text-embedding-3-small"
_default_enable_thinking: bool | int | str | None = None
_default_max_tokens: int = 64000
_default_request_timeout: float = 120

_EMBED_MAX_TOKENS = 8191
_EMBED_ENCODING = tiktoken.get_encoding("cl100k_base")
_embed_logger = logging.getLogger(__name__)

# GPT-5.6 family through /v1/responses.
# Keep older and non-OpenAI models on Chat Completions unless the upstream
# explicitly rejects that endpoint and tells us to use Responses instead.
_RESPONSES_API_MODEL_PREFIXES = ("gpt-5.6",)

load_dotenv()
anthropic_client = anthropic.Anthropic()
anthropic_model: str = "claude-sonnet-4-20250514"


def init_from_config(cfg) -> None:
    global \
        _default_provider, \
        _default_model, \
        _default_embed_model, \
        anthropic_model, \
        _default_enable_thinking, \
        _default_max_tokens, \
        _default_request_timeout
    _default_provider = cfg.llm.provider
    _default_model = cfg.llm.model
    _default_embed_model = cfg.llm.embed_model
    anthropic_model = cfg.llm.computer_use_model
    _default_enable_thinking = cfg.llm.enable_thinking
    _default_max_tokens = cfg.llm.max_tokens
    _default_request_timeout = getattr(cfg.llm, "request_timeout", 120)


def async_retry(times=3):
    def func_wrapper(f):
        async def wrapper(*args, **kwargs):
            wait = 1
            max_wait = 5
            last_exc = None
            for _ in range(times):
                # noinspection PyBroadException
                try:
                    return await f(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    print("got exc", exc)
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, max_wait)
                    pass
            if last_exc:
                raise last_exc

        return wrapper

    return func_wrapper


def retry(times=3):
    def func_wrapper(f):
        def wrapper(*args, **kwargs):
            wait = 1
            max_wait = 5
            last_exc = None
            for _ in range(times):
                # noinspection PyBroadException
                try:
                    return f(*args, **kwargs)
                except Exception as exc:
                    print("got exc", exc)
                    last_exc = exc
                    time.sleep(wait)
                    wait = min(wait * 2, max_wait)
                    pass
            if last_exc:
                raise last_exc

        return wrapper

    return func_wrapper


def _extract_json_string(text: str) -> str:
    import regex

    json_pattern = r"\{(?:[^{}]*|(?R))*\}"
    matches = regex.findall(json_pattern, text, regex.DOTALL)
    if matches:
        return matches[0]
    else:
        raise Exception("No JSON object found in the response")


def _to_response_format(schema: Any) -> Any:
    """Normalize a pydantic BaseModel class or json-schema dict into the litellm
    response_format payload that triggers OpenAI structured outputs."""
    if isinstance(schema, dict):
        return schema
    if isinstance(schema, type):
        try:
            from pydantic import BaseModel
        except ImportError as e:
            raise RuntimeError(
                "pydantic is required to pass a BaseModel as response_schema"
            ) from e
        if issubclass(schema, BaseModel):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
    raise TypeError(
        f"response_schema must be a pydantic BaseModel class or json-schema dict, got {type(schema).__name__}"
    )


def _model_requires_responses_api(provider: str, model: str) -> bool:
    """Return whether a model is known to be available only via Responses."""
    if provider.lower() != "openai":
        return False
    bare_model = model.lower().rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return any(
        bare_model == prefix or bare_model.startswith(f"{prefix}-")
        for prefix in _RESPONSES_API_MODEL_PREFIXES
    )


def _error_requires_responses_api(exc: Exception, provider: str) -> bool:
    """Detect an upstream rejection that explicitly requires /v1/responses."""
    if provider.lower() != "openai":
        return False
    message = str(exc).lower()
    return (
        "/v1/chat/completions" in message
        and "/v1/responses" in message
        and ("not supported" in message or "please use" in message)
    )


def _to_responses_call_kwargs(call_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate Chat Completions-only fields to their Responses equivalents."""
    responses_kwargs = dict(call_kwargs)

    reasoning_effort = responses_kwargs.pop("reasoning_effort", None)
    if reasoning_effort is not None and "reasoning" not in responses_kwargs:
        responses_kwargs["reasoning"] = {"effort": reasoning_effort}

    response_format = responses_kwargs.pop("response_format", None)
    if response_format is not None and "text" not in responses_kwargs:
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_schema"
        ):
            json_schema = response_format.get("json_schema", {})
            responses_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response"),
                    "schema": json_schema.get("schema", {}),
                    "strict": json_schema.get("strict", True),
                }
            }
        else:
            responses_kwargs["text"] = {"format": response_format}

    return responses_kwargs


def _value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from either LiteLLM's model objects or plain dictionaries."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _responses_output_text(response: Any) -> str:
    """Extract assistant text from a LiteLLM Responses API result."""
    direct_text = _value(response, "output_text")
    if isinstance(direct_text, str):
        return direct_text

    text_parts = []
    for output_item in _value(response, "output", []) or []:
        if _value(output_item, "type") != "message":
            continue
        for content_item in _value(output_item, "content", []) or []:
            if _value(content_item, "type") == "output_text":
                text = _value(content_item, "text", "")
                if text:
                    text_parts.append(text)
    return "".join(text_parts)


def _responses_usage(response: Any) -> tuple[int, int]:
    usage = _value(response, "usage")
    return (
        _value(usage, "input_tokens", 0) or 0,
        _value(usage, "output_tokens", 0) or 0,
    )


async def _async_responses(
    *,
    model: str,
    messages: Any,
    max_output_tokens: int,
    timeout: float,
    base_url: str | None,
    call_kwargs: Dict[str, Any],
) -> Any:
    responses_kwargs = _to_responses_call_kwargs(call_kwargs)
    responses_kwargs.setdefault("store", False)
    responses_kwargs.setdefault("cache", {"no-cache": True, "no-store": True})
    responses_kwargs.update(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        drop_params=True,
        stream=False,
        timeout=timeout,
    )
    if base_url:
        responses_kwargs["base_url"] = base_url
    return await litellm.aresponses(**responses_kwargs)


def _sync_responses(
    *,
    model: str,
    messages: Any,
    call_kwargs: Dict[str, Any],
) -> Any:
    responses_kwargs = _to_responses_call_kwargs(call_kwargs)
    responses_kwargs.setdefault("store", False)
    responses_kwargs.setdefault("cache", {"no-cache": True, "no-store": True})
    responses_kwargs.update(
        model=model,
        input=messages,
        drop_params=True,
        stream=False,
    )
    return litellm.responses(**responses_kwargs)


@async_retry()
async def async_chat(
    messages,
    model_name: str | None = None,
    json_mode: bool = False,
    log: bool = True,
    max_tokens: int | None = None,
    enable_thinking: bool | int | str | None = None,
    provider: str | None = None,
    response_schema: Any | None = None,
    llm_config: Any | None = None,
    request_timeout: float | None = None,
    **kwargs,
) -> str:
    """
    Async chat completion that returns the string LLM output.

    Args:
        model_name: model identifier (e.g. "gpt-5-mini"); falls back to llm_model in base.yaml
        provider: provider prefix (e.g. "openai", "gemini", "anthropic"); falls back to llm_provider in base.yaml
        json_mode: whether the result should be json-deserializable
        log: whether to append raw messages/response to the active LogApiCall
        max_tokens: the maximum number of tokens
        enable_thinking: enable extended thinking (anthropic only); pass an int to set a custom budget_tokens
        response_schema: a pydantic BaseModel subclass (or json-schema dict) used to constrain
            the output via structured outputs. When set on OpenAI, this takes precedence over
            json_mode's generic json_object response_format. Removes the need for the caller
            to re-parse free-form text into JSON.

    Returns:
        A single string object outputted by the LLM.
    """
    mgr = context.api_call_manager.get()
    if mgr and log:
        mgr.request.append(messages)

    _provider = provider or _default_provider
    _model = model_name or _default_model
    litellm_model = f"{_provider}/{_model}"
    call_kwargs: Dict[str, Any] = dict(**kwargs)
    if enable_thinking:
        if _provider == "anthropic":
            call_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": enable_thinking
                if isinstance(enable_thinking, int)
                else 32000,
            }
        else:  # openai, gemini
            call_kwargs["reasoning_effort"] = (
                enable_thinking if isinstance(enable_thinking, str) else "medium"
            )
    if response_schema is not None:
        call_kwargs["response_format"] = _to_response_format(response_schema)
    elif json_mode and _provider == "openai":
        call_kwargs["response_format"] = {"type": "json_object"}
    # Resolve max_tokens: explicit arg > base.yaml llm.max_tokens > 64000 fallback.
    # OpenAI's gpt-4o / gpt-5-mini cap completion at 16384, so clamp for openai
    # provider while leaving gemini/anthropic free to use the configured value.
    resolved_max_tokens = max_tokens if max_tokens is not None else _default_max_tokens
    effective_max_tokens = (
        min(resolved_max_tokens, 16000) if _provider == "openai" else resolved_max_tokens
    )
    # Resolve timeout: explicit arg > llm_config.request_timeout > module default.
    # Passed natively to litellm (aborts the request cleanly and raises litellm.Timeout,
    # which @async_retry retries) rather than wrapping in asyncio.wait_for — the latter
    # cancels the coroutine mid-flight and triggers litellm logging-worker noise.
    effective_timeout = (
        request_timeout
        if request_timeout is not None
        else getattr(llm_config, "request_timeout", None) or _default_request_timeout
    )

    acompletion_kwargs: Dict[str, Any] = dict(
        model=litellm_model,
        messages=messages,
        max_tokens=effective_max_tokens,
        drop_params=True,
        stream=False,
        timeout=effective_timeout,
        # Never read from or write to litellm's client-side response cache, so every call hits the real API.
        cache={"no-cache": True, "no-store": True},
        **call_kwargs,
    )
    base_url = getattr(llm_config, "base_url", None)
    if base_url:
        acompletion_kwargs["base_url"] = base_url

    uses_responses_api = _model_requires_responses_api(_provider, _model)
    if uses_responses_api:
        response = await _async_responses(
            model=litellm_model,
            messages=messages,
            max_output_tokens=effective_max_tokens,
            timeout=effective_timeout,
            base_url=base_url,
            call_kwargs=call_kwargs,
        )
        content = _responses_output_text(response)
    else:
        try:
            response = await litellm.acompletion(**acompletion_kwargs)
            content = response.choices[0].message.get("content", "")
        except Exception as exc:
            # This also handles future Responses-only OpenAI models without
            # forcing all existing models away from Chat Completions.
            if not _error_requires_responses_api(exc, _provider):
                raise
            uses_responses_api = True
            response = await _async_responses(
                model=litellm_model,
                messages=messages,
                max_output_tokens=effective_max_tokens,
                timeout=effective_timeout,
                base_url=base_url,
                call_kwargs=call_kwargs,
            )
            content = _responses_output_text(response)

        if not uses_responses_api:
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                print("finish_reason:", finish_reason)
                print("content:", content)
                print("response:", response)

    if uses_responses_api:
        status = _value(response, "status")
        if status and status != "completed":
            print("response_status:", status)
            print("content:", content)
            print("response:", response)

    if mgr and log:
        mgr.response.append(content)
        if uses_responses_api:
            prompt_tokens, completion_tokens = _responses_usage(response)
        else:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
        mgr.prompt_tokens += prompt_tokens
        mgr.completion_tokens += completion_tokens
        # mgr.price_cost += litellm.completion_cost(completion_response=response)
        mgr.model_name.append(litellm_model)

    if json_mode:
        try:
            json_str = _extract_json_string(content)
            _ = json.loads(json_str)
            return json_str
        except Exception as e:
            print(e)
            print(content)
            raise Exception("Invalid JSON in response") from e
    return content


@retry()
def chat(
    messages,
    model_name: str | None = None,
    enable_thinking: bool | int | str | None = None,
    json_mode: bool = False,
    provider: str | None = None,
    **kwargs,
) -> str:
    """
    Returns LLM text completion given list of formatted messages.

    Args:
        model_name: model identifier; falls back to llm_model in base.yaml
        provider: provider prefix; falls back to llm_provider in base.yaml
        enable_thinking: enable extended thinking; pass an int for a custom budget_tokens
        json_mode: whether to enable JSON mode

    Returns:
        String output of the LLM model
    """
    _provider = provider or _default_provider
    _model = model_name or _default_model
    litellm_model = f"{_provider}/{_model}"
    call_kwargs: Dict[str, Any] = dict(**kwargs)
    if enable_thinking:
        if _provider == "anthropic":
            call_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": enable_thinking
                if isinstance(enable_thinking, int)
                else 1024,
            }
        else:  # openai, gemini
            call_kwargs["reasoning_effort"] = (
                enable_thinking if isinstance(enable_thinking, str) else "medium"
            )
    if json_mode and _provider == "openai":
        call_kwargs["response_format"] = {"type": "json_object"}

    try:
        if _model_requires_responses_api(_provider, _model):
            response = _sync_responses(
                model=litellm_model,
                messages=messages,
                call_kwargs=call_kwargs,
            )
            return _responses_output_text(response)

        try:
            response = litellm.completion(
                model=litellm_model,
                messages=messages,
                drop_params=True,
                # Never read from or write to litellm's client-side response cache.
                cache={"no-cache": True, "no-store": True},
                **call_kwargs,
            )
            return response.choices[0].message["content"]
        except Exception as exc:
            if not _error_requires_responses_api(exc, _provider):
                raise
            response = _sync_responses(
                model=litellm_model,
                messages=messages,
                call_kwargs=call_kwargs,
            )
            return _responses_output_text(response)
    except Exception as e:
        print(messages)
        print(e)
        raise e


def _truncate_for_embedding(text: str) -> str:
    tokens = _EMBED_ENCODING.encode(text)
    if len(tokens) <= _EMBED_MAX_TOKENS:
        return text
    _embed_logger.warning(
        "Truncating embedding input from %d to %d tokens",
        len(tokens),
        _EMBED_MAX_TOKENS,
    )
    return _EMBED_ENCODING.decode(tokens[:_EMBED_MAX_TOKENS])


async def embed_text(
    texts: list[str],
    model_name: str | None = None,
    provider: str | None = None,
) -> list[list[float]]:
    """
    Embed a list of texts using the provider configured in base.yaml.

    Args:
        model_name: embedding model identifier; falls back to llm_embed_model in base.yaml
        provider: provider prefix; falls back to llm_provider in base.yaml

    Returns:
        List of list[float] representing each of the embedded texts
    """
    try:
        _provider = provider or _default_provider
        _model = model_name or _default_embed_model
        litellm_model = f"{_provider}/{_model}"
        truncated = [_truncate_for_embedding(t) for t in texts]
        response = await litellm.aembedding(model=litellm_model, input=truncated)
        return [e["embedding"] for e in response.data]
    except Exception as e:
        print(texts)
        print(e)
        raise e


def chat_anthropic_computer_use(
    messages,
    system: BetaTextBlockParam,
    model=anthropic_model,
    screen_width: int = 1024,
    screen_height: int = 768,
) -> (list[BetaToolUseBlockParam], list[Dict, Any]):
    """
    Given a system block and JSON messages, return the tool use block generated by the computer use tool
    """
    response = anthropic_client.beta.messages.create(
        model=model,
        max_tokens=1024,
        tools=[
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": screen_width,
                "display_height_px": screen_height,
                "display_number": 1,
            },
            {
                "name": "web_browser",
                "description": "High-level browser controls",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "switch_tab",
                                "forward",
                                "back",
                                "new_tab",
                                "goto_url",
                                "close_tab",
                                "terminate",
                            ],
                        },
                        "tab_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Zero-based index for switch_tab and close_tab",
                        },
                        "url": {
                            "type": "string",
                            "description": "URL input, only required for goto_url and new_tab",
                        },
                    },
                    "required": ["action"],
                },
            },
        ],
        system=[system],
        messages=messages,
        betas=["computer-use-2025-01-24"],
    )

    return response


def load_prompt(prompt_name, prompt_dir="ux_prompts"):
    root = Path(__file__).parent.absolute()
    dirs = [prompt_dir] if isinstance(prompt_dir, str) else list(prompt_dir)
    for d in dirs:
        path = root / d / f"{prompt_name}.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt '{prompt_name}' not found in any of {dirs}")
