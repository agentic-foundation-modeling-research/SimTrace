#!/usr/bin/env python3
"""Rewrite collection titles using a vendor mapping and an LLM review.

The rewrite happens in two passes:

1. Every case-insensitive *substring* matching an original vendor name is
   replaced with that vendor's synthetic name. Longer vendor names take
   precedence over shorter, overlapping names.
2. Titles with no vendor match are reviewed by an LLM. Generic collection
   names are kept; source-identifying names (named campaigns, proprietary
   series, people, places, and similar labels) are generalized.

The final titles are required to be globally unique (case-insensitive). Only
the ``title`` field is changed; handles and every other collection field are
left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_BASE_URL = None
DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MAX_ATTEMPTS = 4


def normalized_title(value: str) -> str:
    """Normalize a title for uniqueness comparisons."""
    return unicodedata.normalize("NFKC", value).strip().casefold()


def extract_json(text: str) -> Any:
    """Parse JSON, tolerating a surrounding Markdown code fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON without leaving a partially written destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_collections(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("collections.json must contain a JSON array")
    for index, collection in enumerate(data):
        if not isinstance(collection, dict):
            raise ValueError(f"Collection at index {index} must be an object")
        if (
            not isinstance(collection.get("title"), str)
            or not collection["title"].strip()
        ):
            raise ValueError(
                f"Collection at index {index} must have a non-empty string title"
            )
    return data


def validate_vendor_mapping(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        raise ValueError("vendor_mapping.json must contain a JSON object")

    mapping: dict[str, str] = {}
    for original, replacement in data.items():
        if not isinstance(original, str) or not original.strip():
            raise ValueError("Every vendor mapping key must be a non-empty string")
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(
                f"Replacement for vendor {original!r} must be a non-empty string"
            )
        original_name = original.strip()
        replacement_name = replacement.strip()
        mapping[original_name] = replacement_name
    return mapping


class VendorSubstringRewriter:
    """Perform all vendor substitutions in one pass to avoid cascading."""

    def __init__(self, mapping: dict[str, str]) -> None:
        ordered = sorted(
            mapping, key=lambda value: (-len(value), value.casefold(), value)
        )
        self._replacement_by_exact = dict(mapping)
        self._replacement_by_key: dict[str, str] = {}
        pattern_names: list[str] = []
        for original in ordered:
            key = normalized_title(original)
            # Capitalization variants can legitimately have different mappings,
            # e.g. "BRIO" and "Brio". Exact spelling wins during replacement;
            # the first sorted variant is the deterministic mixed-case fallback.
            if key not in self._replacement_by_key:
                self._replacement_by_key[key] = mapping[original]
                pattern_names.append(original)
        self._pattern = (
            re.compile(
                "|".join(re.escape(original) for original in pattern_names),
                re.IGNORECASE,
            )
            if pattern_names
            else None
        )

    def rewrite(self, title: str) -> tuple[str, tuple[str, ...]]:
        """Return the rewritten title and synthetic vendor names inserted."""
        if self._pattern is None:
            return title, ()

        replacements: list[str] = []

        def replace(match: re.Match[str]) -> str:
            matched_text = match.group(0)
            replacement = self._replacement_by_exact.get(matched_text)
            if replacement is None:
                replacement = self._replacement_by_key[normalized_title(matched_text)]
            replacements.append(replacement)
            return replacement

        rewritten = self._pattern.sub(replace, title)
        rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
        # Preserve first-occurrence order for prompt constraints.
        unique_replacements = tuple(dict.fromkeys(replacements))
        return rewritten, unique_replacements


@dataclass(frozen=True)
class ReviewItem:
    index: int
    title: str
    required_terms: tuple[str, ...] = ()
    reason: str = "no_vendor_match"


class CollectionTitleReviewer:
    """Ask an LLM to generalize only source-identifying collection titles."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str = DEFAULT_MODEL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.client = client
        self.model = model
        self.max_attempts = max_attempts

    def review(
        self,
        items: list[ReviewItem],
        reserved_titles: list[str],
    ) -> dict[int, str]:
        if not items:
            return {}

        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            prompt = self._build_prompt(items, reserved_titles, feedback)
            response_text = self._request(prompt)
            data = extract_json(response_text)
            try:
                return self._validate_response(data, items, reserved_titles)
            except ValueError as exc:
                feedback = str(exc)
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        "The LLM did not return a valid, globally unique title set "
                        f"after {self.max_attempts} attempts. Last error: {exc}"
                    ) from exc
                print(f"LLM response attempt {attempt} was invalid: {exc}; retrying...")
        raise AssertionError("unreachable")

    def _request(self, prompt: str) -> str:
        # Streaming avoids SDK timeouts on large collection catalogs while still
        # yielding one assembled message for strict JSON validation.
        with self.client.messages.stream(
            model=self.model,
            max_tokens=32000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def _build_prompt(
        items: list[ReviewItem], reserved_titles: list[str], feedback: str
    ) -> str:
        payload = [
            {
                "index": item.index,
                "title": item.title,
                "required_terms": list(item.required_terms),
                "reason": item.reason,
            }
            for item in items
        ]
        prompt = f"""You are anonymizing an ecommerce collection catalog.

For every input item, decide whether its title is uniquely identifying to the
source store. Identifying titles include names of real people, creators,
curators, locations, proprietary campaigns or series, store-exclusive labels,
and real brands that escaped the deterministic vendor mapping. Rewrite those
titles as natural, general ecommerce collection names that preserve the
original product theme and useful specificity.

Keep a title unchanged when it is already a generic category, product type,
age or price range, season, holiday, event, heritage month, or ordinary shopping
category. Do not make already-generic names vaguer. Do not invent brands,
people, places, or proprietary series.

Rules:
- Return exactly one result for every input index and no other indices.
- Every returned title must be non-empty and globally unique,
  case-insensitively, including against RESERVED TITLES.
- If an item's required_terms list is non-empty, preserve each of those exact
  synthetic vendor strings in its returned title while making it unique.
- Preserve capitalization, punctuation, and grammar naturally.
- Return ONLY a JSON array using this exact shape:
  [{{"index": 0, "title": "Final title", "was_identifying": false}}]

RESERVED TITLES (already rewritten; do not reuse):
{json.dumps(reserved_titles, ensure_ascii=False, indent=2)}

INPUT ITEMS:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
        if feedback:
            prompt += f"\nThe previous response failed validation:\n{feedback}\nFix it.\n"
        return prompt

    @staticmethod
    def _validate_response(
        data: Any,
        items: list[ReviewItem],
        reserved_titles: list[str],
    ) -> dict[int, str]:
        if not isinstance(data, list):
            raise ValueError("response must be a JSON array")

        expected = {item.index: item for item in items}
        result: dict[int, str] = {}
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("every response row must be an object")
            index = row.get("index")
            title = row.get("title")
            if not isinstance(index, int) or index not in expected:
                raise ValueError(f"unexpected or invalid index: {index!r}")
            if index in result:
                raise ValueError(f"duplicate response index: {index}")
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"index {index} has an empty or invalid title")
            title = title.strip()
            missing_terms = [
                term
                for term in expected[index].required_terms
                if term.casefold() not in title.casefold()
            ]
            if missing_terms:
                raise ValueError(
                    f"index {index} omitted required synthetic vendor terms: "
                    f"{missing_terms}"
                )
            result[index] = title

        missing_indices = sorted(set(expected) - set(result))
        if missing_indices:
            raise ValueError(f"response omitted indices: {missing_indices}")

        seen = {normalized_title(title): title for title in reserved_titles}
        for index, title in result.items():
            key = normalized_title(title)
            if key in seen:
                raise ValueError(
                    f"title {title!r} at index {index} duplicates {seen[key]!r}"
                )
            seen[key] = title
        return result


def rewrite_collection_titles(
    collections: list[dict[str, Any]],
    vendor_mapping: dict[str, str],
    reviewer: CollectionTitleReviewer | None,
) -> list[dict[str, Any]]:
    """Return copied collections whose titles satisfy both rewrite passes."""
    substring_rewriter = VendorSubstringRewriter(vendor_mapping)
    output = [dict(collection) for collection in collections]

    candidates: list[ReviewItem] = []
    reserved_titles: list[str] = []
    reserved_keys: set[str] = set()

    for index, collection in enumerate(output):
        rewritten, replacement_terms = substring_rewriter.rewrite(collection["title"])
        collection["title"] = rewritten
        key = normalized_title(rewritten)

        if not replacement_terms:
            candidates.append(ReviewItem(index=index, title=rewritten))
        elif key in reserved_keys:
            # Vendor replacement is still honored, but an LLM must distinguish
            # this duplicate while retaining the inserted synthetic name(s).
            candidates.append(
                ReviewItem(
                    index=index,
                    title=rewritten,
                    required_terms=replacement_terms,
                    reason="duplicate_after_vendor_replacement",
                )
            )
        else:
            reserved_keys.add(key)
            reserved_titles.append(rewritten)

    if candidates:
        if reviewer is None:
            raise ValueError(
                "An LLM reviewer is required for titles without vendor matches"
            )
        reviewed = reviewer.review(candidates, reserved_titles)
        for index, title in reviewed.items():
            output[index]["title"] = title

    final_seen: dict[str, int] = {}
    for index, collection in enumerate(output):
        key = normalized_title(collection["title"])
        if key in final_seen:
            raise RuntimeError(
                f"Final title {collection['title']!r} at index {index} duplicates "
                f"index {final_seen[key]}"
            )
        final_seen[key] = index
    return output


def resolve_path(input_dir: Path, value: Path | None, default_name: str) -> Path:
    if value is None:
        return input_dir / default_name
    return value if value.is_absolute() else (Path.cwd() / value).resolve()


def main(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    collections_path = resolve_path(input_dir, args.collections_file, "collections.json")
    mapping_path = resolve_path(input_dir, args.vendor_mapping, "vendor_mapping.json")
    output_path = resolve_path(input_dir, args.output_file, "collections.json")

    collections = validate_collections(load_json(collections_path))
    vendor_mapping = validate_vendor_mapping(load_json(mapping_path))

    deterministic = VendorSubstringRewriter(vendor_mapping)
    needs_llm = any(not deterministic.rewrite(item["title"])[1] for item in collections)
    reviewer = None
    deterministic_titles = {
        normalized_title(deterministic.rewrite(item["title"])[0])
        for item in collections
    }
    if needs_llm or len(deterministic_titles) != len(collections):
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {args.api_key_env} is required for LLM review"
            )
        client = anthropic.Anthropic(api_key=api_key, base_url=args.base_url)
        reviewer = CollectionTitleReviewer(
            client=client, model=args.model, max_attempts=args.max_attempts
        )

    rewritten = rewrite_collection_titles(collections, vendor_mapping, reviewer)
    changes = [
        (before["title"], after["title"])
        for before, after in zip(collections, rewritten, strict=True)
        if before["title"] != after["title"]
    ]

    print(
        f"Reviewed {len(collections)} collections; changed {len(changes)} title(s); "
        f"all {len(rewritten)} final titles are unique."
    )
    for before, after in changes:
        print(f"  {before!r} -> {after!r}")

    if args.dry_run:
        print("Dry run: no file written.")
    else:
        atomic_write_json(output_path, rewritten)
        print(f"Wrote {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite collection titles with vendor_mapping.json, generalize "
            "source-identifying titles with an LLM, and enforce unique names."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing collections.json and vendor_mapping.json.",
    )
    parser.add_argument(
        "--collections-file",
        type=Path,
        default=None,
        help=(
            "Collections JSON path, relative to the working directory "
            "(default: INPUT_DIR/collections.json)."
        ),
    )
    parser.add_argument(
        "--vendor-mapping",
        type=Path,
        default=None,
        help=(
            "Vendor mapping path, relative to the working directory "
            "(default: INPUT_DIR/vendor_mapping.json)."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Output path, relative to the working directory "
            "(default: overwrite INPUT_DIR/collections.json atomically)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed changes without writing the output file.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Text model name.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Optional Anthropic-compatible API base URL (default: Anthropic API).",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"API-key environment variable (default: {DEFAULT_API_KEY_ENV}).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum LLM validation attempts (default: {DEFAULT_MAX_ATTEMPTS}).",
    )
    return parser


if __name__ == "__main__":
    try:
        arguments = build_parser().parse_args()
        if arguments.max_attempts < 1:
            raise ValueError("--max-attempts must be at least 1")
        raise SystemExit(main(arguments))
    except (ValueError, RuntimeError, anthropic.APIError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
