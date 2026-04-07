#!/usr/bin/env python3

import logging

from src.pipeline import run_pipeline


def configure_logging() -> None:
    """Configure the CLI logging format once at process startup."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def main() -> None:
    """Run the pinned training-and-inference pipeline."""

    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
