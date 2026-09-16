#!/usr/bin/env python3
"""Run only the homepage-image stage from the unified ``syn_gen`` pipeline.

Usage:
    uv run python packages/syn_gen/regenerate_homepage_images.py \
        </path/to/source/data/homepage.json> \
        --output-dir </path/to/source/data> \
        --workers 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import syn_gen as product_syn_gen


@dataclass(frozen=True)
class CliArgs:
    """Command-line arguments."""

    homepage_file: Path
    output_dir: Path | None
    workers: int


def atomic_write_json(path: Path, data: object) -> None:
    """Atomically write JSON without leaving a partial destination file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def regenerate_homepage(args: CliArgs) -> int:
    """Regenerate all remote images in a homepage document."""
    if not args.homepage_file.is_file():
        print(f"File not found: {args.homepage_file}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or args.homepage_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    try:
        succeeded = product_syn_gen.regenerate_homepage_images(
            args.homepage_file,
            output_dir,
            args.workers,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"Invalid homepage file: {exc}", file=sys.stderr)
        return 1

    cost_path = output_dir / "homepage_image_cost_report.json"
    atomic_write_json(
        cost_path,
        product_syn_gen.LEDGER.report(time.perf_counter() - wall_start),
    )
    print(f"Wrote {cost_path}", flush=True)
    return 0 if succeeded else 1


def parse_args() -> CliArgs:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate homepage hero and banner images with syn_gen.py's "
            "copyright-preserve image pipeline."
        )
    )
    parser.add_argument("homepage_file", type=Path, help="Path to homepage.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output data directory (default: the input file's directory).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of images processed in parallel (default: 4).",
    )
    namespace = parser.parse_args()
    return CliArgs(
        homepage_file=namespace.homepage_file,
        output_dir=namespace.output_dir,
        workers=namespace.workers,
    )


if __name__ == "__main__":
    raise SystemExit(regenerate_homepage(parse_args()))
