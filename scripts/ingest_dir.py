#!/usr/bin/env python3
"""Bulk-ingest a directory of documents directly through the RagEngine.

Usage::

    python scripts/ingest_dir.py ./docs --tenant demo

Walks the directory, extracts text from supported files, and ingests each as a
document whose ``doc_id`` is its relative path (so re-running updates in place).
Runs in-process — no server required — which makes it handy for seeding an
index or for batch/offline reindexing jobs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.rag import RagEngine  # noqa: E402

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".rst", ".csv", ".json"}


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk ingest a directory.")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tenant", default="demo")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        return 2

    engine = RagEngine(get_settings())
    root = args.directory.resolve()

    total_docs = total_chunks = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix.lower() != ".pdf":
            continue
        text = read_text(path)
        if not text.strip():
            continue
        rel = str(path.relative_to(root))
        res = engine.ingest(
            tenant=args.tenant, text=text, source=rel, doc_id=rel,
            metadata={"path": rel, "ext": path.suffix.lower().lstrip(".")},
        )
        status = "skipped (unchanged)" if res.skipped else f"{res.chunks} chunks"
        print(f"  {rel}: {status}")
        total_docs += 1
        total_chunks += res.chunks

    print(f"\nIngested {total_docs} documents, {total_chunks} new chunks "
          f"into tenant '{args.tenant}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
