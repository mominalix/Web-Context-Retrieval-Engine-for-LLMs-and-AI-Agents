"""Command-line interface for the semantic web retrieval pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace

from .config import Settings
from .exceptions import SearchEngineError
from .pipeline import SearchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-search",
        description="Fetch and rank token-budgeted web context for an AI agent.",
    )
    parser.add_argument("query", help="The web query to retrieve context for")
    parser.add_argument("--num-results", type=int, help="Number of web results to inspect")
    parser.add_argument("--top-k", type=int, help="Maximum number of context chunks")
    parser.add_argument("--max-context-tokens", type=int, help="Hard context token budget")
    parser.add_argument("--model", dest="model_name", help="Sentence Transformers model name")
    parser.add_argument(
        "--vector-backend", choices=("auto", "numpy", "faiss"), help="Vector index backend"
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON")
    parser.add_argument("--verbose", action="store_true", help="Show skipped-page warnings")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    overrides = {
        "search_results": args.num_results,
        "top_k": args.top_k,
        "max_context_tokens": args.max_context_tokens,
        "model_name": args.model_name,
        "vector_backend": args.vector_backend,
    }
    settings = replace(
        settings,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    try:
        with SearchPipeline(settings) as pipeline:
            response = pipeline.run(args.query)
    except (SearchEngineError, ValueError) as exc:
        parser.exit(1, f"semantic-search: error: {exc}\n")

    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(
        f"Retrieved {len(response.hits)} chunks ({response.context_tokens} tokens) "
        f"from {response.indexed_documents} documents."
    )
    for hit in response.hits:
        print(f"\n[{hit.rank}] score={hit.score:.3f} source={hit.source or 'unknown'}")
        print(hit.text)
    if response.failures and args.verbose:
        print(f"\nSkipped or fell back on {len(response.failures)} result(s).")
    return 0
