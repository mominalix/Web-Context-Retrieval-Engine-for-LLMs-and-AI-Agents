"""Compatibility entry point. Prefer ``python -m search_engine``."""

from search_engine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
