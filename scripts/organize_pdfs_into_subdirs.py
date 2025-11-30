#!/usr/bin/env python3
"""
Organize flat PDFs into per-file subdirectories under a target root.

Example:
    python scripts/organize_pdfs_into_subdirs.py \
        --src data/examples \
        --dst organized_pdfs \
        --move
Result:
    organized_pdfs/foo/foo.pdf
    organized_pdfs/bar/bar.pdf
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in sorted(root.glob("*.pdf")):
        if p.is_file():
            yield p


def organize_pdfs(src: Path, dst: Path, move: bool) -> None:
    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source directory not found: {src}")

    dst.mkdir(parents=True, exist_ok=True)

    pdfs = list(iter_pdfs(src))
    if not pdfs:
        print(f"No PDFs found under {src}")
        return

    op = shutil.move if move else shutil.copy2

    for pdf_path in pdfs:
        subdir = dst / pdf_path.stem
        subdir.mkdir(parents=True, exist_ok=True)
        target_path = subdir / pdf_path.name

        if target_path.exists():
            print(f"[SKIP] Exists: {target_path}")
            continue

        op(str(pdf_path), str(target_path))
        print(f"[{ 'MOVE' if move else 'COPY' }] {pdf_path} -> {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Place each PDF into its own subdirectory under a target root."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source directory containing PDFs (non-recursive).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Destination root directory where subdirectories will be created.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copy (default is copy).",
    )

    args = parser.parse_args()
    organize_pdfs(Path(args.src), Path(args.dst), move=args.move)


if __name__ == "__main__":
    main()
