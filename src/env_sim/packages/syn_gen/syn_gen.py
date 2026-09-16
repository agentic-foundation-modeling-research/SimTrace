#!/usr/bin/env python3
"""Anonymize a synthetic shop's catalog and homepage images.

Rewrites a shop in five ordered stages:

  1. TitleRewriter       - deterministically replaces vendor mentions in the
                           title with an LLM-invented fake brand name.
  2. DescriptionRewriter - LLM rewrites description_html to stay close to the
                           original while being consistent with the new title.
  3. ImageRewriter       - regenerates each product image with no text / brand
                           names, either as an image-to-image variation
                           (mask-sensitive) or by captioning the original then
                           generating text-to-image (copyright-preserve).
  4. Collection rewriter - replaces vendor names and generalizes identifying
                           collection titles.
  5. Homepage rewriter   - regenerates hero and banner images with the
                           copyright-preserve image pipeline.

Usage:
    uv run python packages/syn_gen/syn_gen.py \
        </path/to/source/data/products.json> \
        --collections-file </path/to/source/data/collections.json> \
        --homepage-file </path/to/source/data/homepage.json> \
        --workers 20 \
        --output-dir </path/to/source/data> \
        --image-mode copyright-preserve [choice: copyright-preserve, mask-sensitive]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any, Required, TypedDict, cast

import anthropic
import httpx
import rewrite_collections
from openai import OpenAI
from PIL import Image, ImageOps

# Unbuffered print for real-time progress.
print = partial(print, flush=True)  # noqa: A001


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEXT_MODEL = "claude-sonnet-4-6"
IMAGE_MODEL = "gpt-image-1"

# Anthropic rejects image inputs larger than 5 MiB. Keep caption inputs well
# below that boundary so base64/multipart overhead and format differences do
# not cause borderline failures.
CAPTION_IMAGE_MAX_BYTES = 4 * 1024 * 1024
CAPTION_IMAGE_MAX_EDGE = 1568

# Retries for covering every vendor when one LLM call drops/truncates entries.
MAX_MAPPING_ATTEMPTS = 5

@cache
def get_text_client() -> anthropic.Anthropic:
    """Return a native Anthropic client configured from its standard environment."""
    return anthropic.Anthropic()


@cache
def get_image_client() -> OpenAI:
    """Return a native OpenAI client configured from its standard environment."""
    return OpenAI()


def _extract_tokens(usage) -> tuple[int | None, int | None]:
    """Normalize Anthropic (input/output_tokens) and OpenAI (prompt/completion_tokens) usage."""
    if usage is None:
        return None, None
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    return inp, out


class CostLedger:
    """Thread-safe ledger of every LLM call's token + time cost."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def record(self, call_type, model, usage, elapsed, **context) -> None:
        inp, out = _extract_tokens(usage)
        with self.lock:
            self.calls.append({
                "type": call_type,
                "model": model,
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": (inp or 0) + (out or 0),
                "elapsed_seconds": round(elapsed, 3),
                **context,
            })

    def report(self, wall_clock_seconds: float | None = None) -> dict:
        with self.lock:
            calls = list(self.calls)
        totals = {
            "num_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            # Sum of per-call durations. Under concurrency calls overlap, so this
            # is cumulative LLM time and exceeds real elapsed time.
            "cumulative_call_seconds": round(sum(c["elapsed_seconds"] for c in calls), 3),
        }
        if wall_clock_seconds is not None:
            totals["wall_clock_seconds"] = round(wall_clock_seconds, 3)
        return {"calls": calls, "totals": totals}


LEDGER = CostLedger()

# ---------------------------------------------------------------------------
# Shared helpers (mirrors packages/shop_gen/synthesize_product_data.py)
# ---------------------------------------------------------------------------


def extract_json(text: str) -> Any:
    """Extract JSON from an LLM response, tolerating markdown fences."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def strip_html_fence(text: str) -> str:
    """Remove a surrounding ```html ... ``` (or bare ```) fence if present."""
    text = text.strip()
    match = re.search(r"```(?:html)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def format_duration(seconds: float) -> str:
    """Format a duration as a compact, human-readable ETA."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON to ``path`` atomically so a crash mid-write can't corrupt it."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def prepare_caption_image(image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    """Downsize oversized image inputs for Anthropic vision captioning."""
    if len(image_bytes) <= CAPTION_IMAGE_MAX_BYTES:
        return image_bytes, mime_type

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        image.thumbnail(
            (CAPTION_IMAGE_MAX_EDGE, CAPTION_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS
        )
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Normally the resize alone is far below 4 MiB. Lower JPEG quality if
        # an unusually detailed image remains too large.
        for quality in (85, 75, 65, 55, 45):
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            compressed = output.getvalue()
            if len(compressed) <= CAPTION_IMAGE_MAX_BYTES:
                return compressed, "image/jpeg"

    raise RuntimeError("Could not compress caption image below the 5 MB API limit")


def mask_image(
    prompt: str,
    output_path: Path,
    client,
    input_image: bytes | None = None,
    input_mime: str = "image/jpeg",
    context: dict | None = None,
) -> None:
    """Generate or edit an image through OpenAI's Images API."""
    t0 = time.perf_counter()
    if input_image is None:
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=prompt,
            output_format="png",
        )
    else:
        response = client.images.edit(
            model=IMAGE_MODEL,
            image=("input.png", input_image, input_mime),
            prompt=prompt,
            output_format="png",
        )
    b64_out = response.data[0].b64_json if response.data else None
    if not b64_out:
        raise RuntimeError("No base64 image data in Images API response")
    usage = getattr(response, "usage", None)

    LEDGER.record(
        "image_generation", IMAGE_MODEL, usage,
        time.perf_counter() - t0, **(context or {})
    )
    output_path.write_bytes(base64.b64decode(b64_out))


def generate_image(
    product_title: str,
    product_description: str,
    output_path: Path,
    client,
    input_image: bytes | None = None,
    input_mime: str = "image/jpeg",
    context: dict | None = None,
) -> tuple[str, str]:
    """Regenerate a product image through a text caption (copyright-preserve).

    Unlike ``mask_image``'s image-to-image variation, the original pixels are
    never fed to the generator: we first caption the original in words, then
    generate a fresh image from that caption alone, so the output is not
    directly derived from the copyrighted source."""
    # 1. caption the original image (text ok, but no logos / brand names).
    if input_image is not None:
        input_image, input_mime = prepare_caption_image(input_image, input_mime)
        b64 = base64.b64encode(input_image).decode("ascii")
        t0 = time.perf_counter()
        caption_msg = get_text_client().messages.create(
            model=TEXT_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": input_mime,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this product photo for an image generator: "
                                "subject, composition, colors, materials, lighting, and "
                                "background. You may describe text printed on the product "
                                "itself if it is part of the design, but do NOT mention any "
                                "logos, watermarks, brand names, or vendor names."
                            ),
                        },
                    ],
                }
            ],
        )
        LEDGER.record(
            "image_caption", TEXT_MODEL, getattr(caption_msg, "usage", None),
            time.perf_counter() - t0, **(context or {})
        )
        caption = caption_msg.content[0].text.strip()
    else:
        caption = f"{product_title}. {product_description}"

    # 2. generate a fresh image from the caption alone (text-to-image).
    prompt = (
        f"Create a clean, photorealistic product photo of '{product_title}' based on "
        f"this description:\n{caption}\n\n"
        "Limit the use of text, do NOT render any logos, watermarks, "
        "brand names, or vendor names, and do NOT render any text that contradicts the "
        "product title or description."
    )
    mask_image(prompt, output_path, client, context=context)
    return caption, prompt


# ---------------------------------------------------------------------------
# 1. Title
# ---------------------------------------------------------------------------


class TitleRewriter:
    """Replaces vendor mentions in product titles with fake brand names."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client
        self.mapping: dict[str, str] = {}

    @staticmethod
    def _vendor_key(value: str) -> str:
        """Normalize punctuation commonly changed by language models."""
        return value.strip().replace("\u2018", "'").replace("\u2019", "'").casefold()

    def build_vendor_mapping(
        self,
        vendors: list[str],
        initial_mapping: dict[str, str] | None = None,
        checkpoint_path: Path | None = None,
    ) -> dict[str, str]:
        """Invent one fake brand per real vendor, verifying full coverage.

        A single LLM call over a long vendor list can drop or truncate entries,
        which would silently leak the real name. So we re-request the vendors
        still missing a valid fake name until every vendor is covered (or we
        exhaust ``MAX_MAPPING_ATTEMPTS`` and fail loudly). Successful entries
        are checkpointed after every attempt so a later run can resume."""
        unique = sorted({v.strip() for v in vendors if v and v.strip()})
        if not unique:
            self.mapping = {}
            return self.mapping

        mapping: dict[str, str] = {}
        initial_mapping = initial_mapping or {}
        normalized_initial = {
            self._vendor_key(str(vendor)): fake
            for vendor, fake in initial_mapping.items()
        }
        used: set[str] = set()
        for vendor in unique:
            fake = str(
                initial_mapping.get(
                    vendor, normalized_initial.get(self._vendor_key(vendor), "")
                )
            ).strip()
            fake_key = fake.casefold()
            if fake and fake_key != vendor.casefold() and fake_key not in used:
                mapping[vendor] = fake
                used.add(fake_key)

        if checkpoint_path is not None:
            _atomic_write_json(checkpoint_path, mapping)

        pending = [v for v in unique if v not in mapping]
        for attempt in range(1, MAX_MAPPING_ATTEMPTS + 1):
            if not pending:
                break
            data = self._request_mapping(pending, sorted(mapping.values()))
            used = {f.lower() for f in mapping.values()}
            normalized_data = {
                self._vendor_key(str(vendor)): fake for vendor, fake in data.items()
            }
            for vendor in pending:
                fake = str(
                    data.get(vendor, normalized_data.get(self._vendor_key(vendor), ""))
                ).strip()
                # A valid fake is non-empty, not the real name back, and not a
                # duplicate of one already assigned to another vendor.
                if fake and fake.lower() != vendor.lower() and fake.lower() not in used:
                    mapping[vendor] = fake
                    used.add(fake.lower())
            self.mapping = mapping
            if checkpoint_path is not None:
                _atomic_write_json(checkpoint_path, mapping)
            pending = [v for v in unique if v not in mapping]
            if not pending:
                break
            print(
                f"  vendor mapping attempt {attempt}: {len(pending)}/{len(unique)} "
                f"still missing, retrying..."
            )

        if pending:
            raise RuntimeError(
                f"Could not generate fake names for {len(pending)} vendors after "
                f"{MAX_MAPPING_ATTEMPTS} attempts: {pending}"
            )

        self.mapping = mapping
        return self.mapping

    def _request_mapping(
        self, vendors: list[str], used: list[str]
    ) -> dict[str, str]:
        """One LLM call: invent a fake brand for each vendor. {} if unparseable."""
        prompt = (
            "You are anonymizing a product catalog. For each real brand/vendor name "
            "below, invent a fictional replacement brand name. Rules:\n"
            "- The fake name must NOT resemble any real-world brand.\n"
            "- Keep it short (1-2 words), plausible as a consumer brand.\n"
            "- Use a single distinctive word where possible (it will be substituted "
            "into product titles).\n"
            "- Each fake name must be UNIQUE across all vendors.\n"
            "- Include EVERY vendor listed; the JSON must have one key per vendor.\n"
        )
        if used:
            prompt += (
                "- Do NOT reuse any of these already-assigned fake names:\n"
                + json.dumps(used, ensure_ascii=False, indent=2)
                + "\n"
            )
        prompt += (
            "\nReturn ONLY a JSON object mapping each original name to its fake name.\n\n"
            "Vendors:\n" + json.dumps(vendors, ensure_ascii=False, indent=2)
        )
        prompt += "\n\n Return:\n"
        t0 = time.perf_counter()
        # Anthropic requires streaming for requests whose maximum output could
        # take longer than ten minutes. Consume the stream fully, then recover
        # the assembled Message so JSON parsing and usage accounting stay the
        # same as for a non-streaming request.
        with self.client.messages.stream(
            model=TEXT_MODEL,
            max_tokens=100000,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
        LEDGER.record(
            "vendor_mapping", TEXT_MODEL, getattr(message, "usage", None),
            time.perf_counter() - t0, num_vendors=len(vendors)
        )
        text = message.content[0].text
        data = extract_json(text)
        return data if isinstance(data, dict) else {}

    def rewrite(self, title: str, vendor: str) -> str:
        """Replace whole-word vendor mentions in ``title`` (case-insensitive)."""
        if not title or not vendor:
            return title
        fake = self.mapping.get(vendor.strip())
        if not fake:
            return title

        # Assert a non-alphanumeric char (or string edge) on each side so the
        # vendor matches as a standalone word/phrase, not a substring inside a
        # larger word. Lookaround (vs \b) keeps punctuation-led/-trailed vendor
        # names like "& Co" matching correctly.
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(vendor.strip()) + r"(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        new_title = pattern.sub(fake, title)
        return re.sub(r"\s{2,}", " ", new_title).strip()


# ---------------------------------------------------------------------------
# 2. Description
# ---------------------------------------------------------------------------


class DescriptionRewriter:
    """LLM-rewrites description_html, consistent with the new title."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.client = client

    def rewrite(
        self,
        description_html: str,
        original_title: str,
        new_title: str,
        fake_vendor: str,
    ) -> str:
        if not description_html or not description_html.strip():
            return description_html

        prompt = (
            "Rewrite the product description HTML below. Requirements:\n"
            "- Stay as close as possible to the original: same structure, length, "
            "tone, and HTML tags.\n"
            f"- Make it consistent with the new product title: \"{new_title}\" "
            f"(it was \"{original_title}\").\n"
            "- Remove any real brand, vendor, designer, or company names. Where a "
            f"brand is referenced, use \"{fake_vendor}\" instead.\n"
            "- Keep all factual product details (materials, measurements, contents).\n"
            "- Return ONLY the rewritten HTML, no commentary, no markdown fences.\n\n"
            f"Original description HTML:\n {description_html}\n\n"
            "New description HTML:\n"
        )
        t0 = time.perf_counter()
        message = self.client.messages.create(
            model=TEXT_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        LEDGER.record(
            "description", TEXT_MODEL, getattr(message, "usage", None),
            time.perf_counter() - t0, product_title=new_title
        )
        text = message.content[0].text
        return strip_html_fence(text)


# ---------------------------------------------------------------------------
# 3. Image
# ---------------------------------------------------------------------------


class ImageRewriter:
    """Regenerates product images with no text / brand, by image-to-image
    variation (mask-sensitive) or caption-then-text-to-image (copyright-preserve)."""

    def __init__(self, mode: str, client: OpenAI, images_dir: Path) -> None:
        self.mode = mode
        self.client = client
        self.images_dir = images_dir
        self.metadata: list[dict] = []
        self.meta_lock = threading.Lock()
        (self.images_dir / self.mode).mkdir(parents=True, exist_ok=True)

    def rewrite(
        self,
        image: dict,
        product_title: str,
        handle: str,
        fake_vendor: str,
        product_description: str,
    ) -> None:
        src = image.get("src", "")
        if not src:
            return
        try:
            resp = httpx.get(src, follow_redirects=True, timeout=60)
            resp.raise_for_status()
            original = resp.content
        except Exception as e:
            print(f"  ! image download failed ({src}): {e}")
            return

        mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not mime.startswith("image/"):
            mime = "image/jpeg"

        filename = f"{handle}-{image.get('id', image.get('position', 0))}.png"
        output_path = self.images_dir / self.mode / filename
        context = {
            "handle": handle,
            "product_title": product_title,
            "filename": filename,
            "mode": self.mode,
        }
        caption = None
        try:
            if self.mode == "mask-sensitive":
                prompt = (
                    f"Recreate a clean, photorealistic product photo similar to the provided "
                    f"image of '{product_title}' from '{fake_vendor}'. Match the subject, "
                    f"composition, colors, and "
                    f"background. Text on the product is allowed, but do NOT render any logos, "
                    f"watermarks, brand names, or vendor names, and do NOT render any text that "
                    f"contradicts the product title or description."
                )
                mask_image(
                    prompt, output_path, self.client, input_image=original,
                    input_mime=mime, context=context
                )
            elif self.mode == "copyright-preserve":
                caption, prompt = generate_image(
                    product_title, product_description, output_path, self.client,
                    input_image=original, input_mime=mime, context=context
                )
        except Exception as e:
            print(f"  ! image generation failed for '{product_title}' from '{fake_vendor}': {e}")
            return

        image["src"] = f"/images/{self.mode}/{filename}"
        with self.meta_lock:
            self.metadata.append({
                "src": image["src"],
                "filename": filename,
                "handle": handle,
                "product_title": product_title,
                "mode": self.mode,
                "caption": caption,
                "prompt": prompt,
            })


# ---------------------------------------------------------------------------
# 4. Collections
# ---------------------------------------------------------------------------


def rewrite_collections_file(
    collections_path: Path,
    output_dir: Path,
    vendor_mapping: dict[str, str],
) -> None:
    """Rewrite collection titles and save them in the output data directory."""
    collections = rewrite_collections.validate_collections(
        rewrite_collections.load_json(collections_path)
    )
    reviewer = rewrite_collections.CollectionTitleReviewer(
        client=get_text_client(),
        model=TEXT_MODEL,
        max_attempts=rewrite_collections.DEFAULT_MAX_ATTEMPTS,
    )
    rewritten = rewrite_collections.rewrite_collection_titles(
        collections,
        vendor_mapping,
        reviewer,
    )
    changes = [
        (before["title"], after["title"])
        for before, after in zip(collections, rewritten, strict=True)
        if before["title"] != after["title"]
    ]
    output_path = output_dir / "collections.json"
    rewrite_collections.atomic_write_json(output_path, rewritten)
    print(
        f"Reviewed {len(collections)} collections; changed {len(changes)} title(s); "
        f"wrote {output_path}"
    )


# ---------------------------------------------------------------------------
# 5. Homepage images
# ---------------------------------------------------------------------------


class HomepageImage(TypedDict, total=False):
    """Image record accepted by the ShopBackend homepage schema."""

    alt: str | None
    height: int
    id: int
    position: int
    src: Required[str]
    width: int


class Homepage(TypedDict):
    """Validated homepage fields used for image regeneration."""

    hero: HomepageImage | None
    banners: list[HomepageImage]


@dataclass(frozen=True)
class HomepageImageJob:
    """One homepage image awaiting regeneration."""

    image: HomepageImage
    role: str
    index: int


def validate_homepage(data: object) -> Homepage:
    """Validate the homepage fields needed for image regeneration."""
    if not isinstance(data, dict):
        raise ValueError("homepage.json must contain a JSON object")

    document = cast(dict[str, object], data)
    hero_data = document.get("hero")
    hero = _validate_homepage_image(hero_data, "hero") if hero_data is not None else None

    banners_data = document.get("banners")
    if not isinstance(banners_data, list):
        raise ValueError("homepage.json field 'banners' must be an array")
    banners = [
        _validate_homepage_image(banner, f"banners[{index}]")
        for index, banner in enumerate(cast(list[object], banners_data))
    ]
    return Homepage(hero=hero, banners=banners)


def _validate_homepage_image(data: object, location: str) -> HomepageImage:
    """Validate one homepage image record."""
    if not isinstance(data, dict):
        raise ValueError(f"homepage.json field '{location}' must be an object")
    image = cast(dict[str, object], data)
    src = image.get("src")
    if not isinstance(src, str) or not src.strip():
        raise ValueError(f"homepage.json field '{location}.src' must be a non-empty string")
    return cast(HomepageImage, image)


def _load_homepage_metadata(path: Path) -> list[dict[str, object]]:
    """Load a metadata checkpoint, ignoring malformed content."""
    if not path.is_file():
        return []
    try:
        existing: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"Ignoring invalid metadata checkpoint: {path}", file=sys.stderr)
        return []
    if not isinstance(existing, list):
        return []
    return [
        cast(dict[str, object], item)
        for item in cast(list[object], existing)
        if isinstance(item, dict)
    ]


class HomepageImageRewriter:
    """Regenerate homepage images with the copyright-preserve pipeline."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.images_dir = output_dir / "images" / "copyright-preserve"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: list[dict[str, object]] = []
        self.metadata_lock = threading.Lock()

    def is_done(self, image: HomepageImage) -> bool:
        """Return whether an image already points to an existing generated file."""
        src = image["src"]
        return (
            src.startswith("/images/copyright-preserve/")
            and (self.output_dir / src.lstrip("/")).is_file()
        )

    def rewrite(self, job: HomepageImageJob) -> bool:
        """Regenerate one homepage image and update its source in place."""
        source_url = job.image["src"]
        try:
            response = httpx.get(source_url, follow_redirects=True, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"  ! image download failed ({source_url}): {exc}")
            return False

        mime_type = response.headers.get("content-type", "image/jpeg")
        mime_type = mime_type.split(";", maxsplit=1)[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        image_id = job.image.get("id", job.image.get("position", job.index + 1))
        filename = f"homepage-{job.role}-{image_id}.png"
        output_path = self.images_dir / filename
        title = (
            "ecommerce homepage hero image"
            if job.role == "hero"
            else "ecommerce homepage promotional banner"
        )
        context: dict[str, object] = {
            "filename": filename,
            "mode": "copyright-preserve",
            "role": job.role,
            "source_url": source_url,
        }

        try:
            caption, prompt = generate_image(
                title,
                "",
                output_path,
                get_image_client(),
                input_image=response.content,
                input_mime=mime_type,
                context=context,
            )
        except Exception as exc:
            print(f"  ! image generation failed for {job.role} ({source_url}): {exc}")
            return False

        job.image["src"] = f"/images/copyright-preserve/{filename}"
        with self.metadata_lock:
            self.metadata.append(
                {
                    "src": job.image["src"],
                    "original_src": source_url,
                    "filename": filename,
                    "role": job.role,
                    "mode": "copyright-preserve",
                    "caption": caption,
                    "prompt": prompt,
                }
            )
        return True


def regenerate_homepage_images(
    homepage_path: Path,
    output_dir: Path,
    workers: int,
) -> bool:
    """Regenerate homepage hero and banner images and write checkpoints."""
    raw_data: object = json.loads(homepage_path.read_text(encoding="utf-8"))
    homepage = validate_homepage(raw_data)
    output_path = output_dir / "homepage.json"
    metadata_path = output_dir / "homepage_image_metadata.json"
    rewriter = HomepageImageRewriter(output_dir)

    jobs: list[HomepageImageJob] = []
    hero = homepage["hero"]
    if hero is not None:
        jobs.append(HomepageImageJob(image=hero, role="hero", index=0))
    jobs.extend(
        HomepageImageJob(image=banner, role=f"banner-{index + 1}", index=index)
        for index, banner in enumerate(homepage["banners"])
    )
    pending = [job for job in jobs if not rewriter.is_done(job.image)]
    skipped = len(jobs) - len(pending)
    print(
        f"Loaded {len(jobs)} homepage images; {skipped} already generated, "
        f"{len(pending)} to process"
    )
    rewriter.metadata = _load_homepage_metadata(metadata_path)

    def flush() -> None:
        _atomic_write_json(output_path, homepage)
        with rewriter.metadata_lock:
            _atomic_write_json(metadata_path, rewriter.metadata)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_job: dict[Future[bool], HomepageImageJob] = {
            executor.submit(rewriter.rewrite, job): job for job in pending
        }
        for completed, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                succeeded = future.result()
            except Exception as exc:
                print(f"  ! unexpected failure for {job.role}: {exc}")
                succeeded = False
            if not succeeded:
                failures += 1
            print(
                f"  [{completed}/{len(pending)}] {job.role}: "
                f"{'done' if succeeded else 'failed'}"
            )
            flush()

    flush()
    print(f"Wrote {output_path} and {metadata_path.name}")
    return failures == 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process_product(
    product: dict,
    title_rw: TitleRewriter,
    desc_rw: DescriptionRewriter,
    image_rw: ImageRewriter,
) -> str:
    """Rewrite title -> description -> images for a single product, in place."""
    original_title = product.get("title", "")
    vendor = product.get("vendor", "")
    fake_vendor = title_rw.mapping.get(vendor.strip(), vendor) if vendor else vendor
    product["vendor"] = fake_vendor

    new_title = title_rw.rewrite(original_title, vendor)
    product["title"] = new_title

    new_description = desc_rw.rewrite(
        product.get("description_html", ""), original_title, new_title, fake_vendor
    )
    product["description_html"] = new_description

    handle = product.get("handle", f"product-{product.get('id', 0)}")
    for image in product.get("images", []):
        image_rw.rewrite(image, new_title, handle, fake_vendor, new_description)

    return new_title


def _prepare_title_rewriter(
    products: list[dict[str, Any]],
    mapping_path: Path,
) -> tuple[TitleRewriter, dict[str, str]]:
    """Load or complete the vendor mapping used by all text rewrite stages."""
    title_rewriter = TitleRewriter(get_text_client())
    vendors = [product.get("vendor", "") for product in products]
    existing_mapping: dict[str, str] = {}
    if mapping_path.exists():
        existing_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print(
            f"Loaded vendor mapping checkpoint ({len(existing_mapping)} vendors) "
            f"from {mapping_path}"
        )

    required_vendors = {vendor.strip() for vendor in vendors if vendor and vendor.strip()}
    if required_vendors.issubset(existing_mapping):
        title_rewriter.mapping = existing_mapping
        print(f"Reusing complete vendor mapping ({len(existing_mapping)} vendors)")
        return title_rewriter, existing_mapping

    mapping = title_rewriter.build_vendor_mapping(
        vendors,
        initial_mapping=existing_mapping,
        checkpoint_path=mapping_path,
    )
    print(f"Built vendor mapping for {len(mapping)} vendors:")
    for original, fake in mapping.items():
        print(f"  {original!r} -> {fake!r}")
    return title_rewriter, mapping


def rewrite_products_file(
    args: argparse.Namespace,
    products_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Run the product title, description, and image rewrite stages."""
    products = json.loads(products_path.read_text(encoding="utf-8"))
    if args.products_index is not None:
        products = [products[i] for i in args.products_index]
    print(f"Loaded {len(products)} products from {products_path}")

    title_rw, mapping = _prepare_title_rewriter(
        products,
        output_dir / "vendor_mapping.json",
    )

    desc_rw = DescriptionRewriter(get_text_client())
    image_rw = ImageRewriter(args.image_mode, get_image_client(), output_dir / "images")

    # Resume: load the prior checkpoint and skip products already fully generated.
    products_out_path = output_dir / "products.json"
    metadata_out_path = output_dir / "image_metadata.json"
    prev = {}
    if products_out_path.exists():
        prev = {p.get("id"): p for p in json.loads(products_out_path.read_text(encoding="utf-8"))}

    def is_done(prod) -> bool:
        prior = prev.get(prod.get("id"))
        if prior is None:
            return False
        for img in prior.get("images", []):
            src = img.get("src", "")
            if not src.startswith("/images/") or not (output_dir / src.lstrip("/")).exists():
                return False
        return True

    todo = []
    done_handles = set()
    for i, p in enumerate(products):
        if is_done(p):
            products[i] = prev[p["id"]]  # reuse rewritten title/desc/images
            done_handles.add(products[i].get("handle"))
        else:
            todo.append(p)
    skipped = len(products) - len(todo)

    # Seed metadata with prior entries for skipped products so the artifact stays complete.
    if skipped and metadata_out_path.exists():
        prev_meta = json.loads(metadata_out_path.read_text(encoding="utf-8"))
        image_rw.metadata = [m for m in prev_meta if m.get("handle") in done_handles]

    def flush() -> None:
        _atomic_write_json(products_out_path, products)
        with image_rw.meta_lock:
            _atomic_write_json(metadata_out_path, list(image_rw.metadata))

    print(f"Resuming: {skipped} already done, {len(todo)} to process")
    progress_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_product = {
            executor.submit(process_product, p, title_rw, desc_rw, image_rw): p
            for p in todo
        }
        for done, future in enumerate(as_completed(future_to_product), start=1):
            elapsed = time.perf_counter() - progress_start
            eta = elapsed / done * (len(todo) - done)
            progress = f"[{done}/{len(todo)}] ETA {format_duration(eta)}"
            try:
                new_title = future.result()
                print(f"  {progress} {new_title}")
            except Exception as e:
                p = future_to_product[future]
                print(f"  {progress} FAILED ({p.get('title')!r}): {e}")
            flush()

    flush()
    print("Product title, description, and image rewrites complete.")
    return mapping


def main(args: argparse.Namespace) -> int:
    """Run the complete five-stage synthetic-shop rewrite pipeline."""
    products_path = Path(args.products_file)
    collections_path = (
        Path(args.collections_file)
        if args.collections_file is not None
        else products_path.with_name("collections.json")
    )
    homepage_path = (
        Path(args.homepage_file)
        if args.homepage_file is not None
        else products_path.with_name("homepage.json")
    )
    missing_paths = [
        path for path in (products_path, collections_path, homepage_path) if not path.is_file()
    ]
    if missing_paths:
        for path in missing_paths:
            print(f"File not found: {path}", file=sys.stderr)
        return 1

    wall_start = time.perf_counter()
    output_dir = args.output_dir or products_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = rewrite_products_file(args, products_path, output_dir)

    rewrite_collections_file(collections_path, output_dir, mapping)
    print("Collection rewrite complete.")

    homepage_succeeded = regenerate_homepage_images(
        homepage_path,
        output_dir,
        args.workers,
    )
    cost_path = output_dir / "cost_report.json"
    _atomic_write_json(cost_path, LEDGER.report(time.perf_counter() - wall_start))
    if not homepage_succeeded:
        print("Homepage image rewrite completed with failures.", file=sys.stderr)
        return 1

    print(
        "Done! Wrote products.json, collections.json, homepage.json, image metadata, "
        f"and {cost_path.name} under {output_dir}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the five-stage pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Anonymize a shop's product titles, descriptions, product images, "
            "collection titles, and homepage images in sequence."
        )
    )
    parser.add_argument(
        "products_file",
        type=Path,
        help="Path to the products.json file to rewrite.",
    )
    parser.add_argument(
        "--collections-file",
        type=Path,
        default=None,
        help="Collections input (default: collections.json beside products_file).",
    )
    parser.add_argument(
        "--homepage-file",
        type=Path,
        default=None,
        help="Homepage input (default: homepage.json beside products_file).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent product or homepage-image jobs (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save rewritten data and generated images.",
    )
    parser.add_argument(
        "--products-index",
        nargs="+",
        type=int,
        default=None,
        help="Index/indices of products to process: single (5) or list (1,2,3). "
        "If omitted, all are processed.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["mask-sensitive", "copyright-preserve"],
        default="mask-sensitive",
        help=(
            "Product-image mode: 'mask-sensitive' (default) or "
            "'copyright-preserve'. Homepage images always use copyright-preserve."
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))
