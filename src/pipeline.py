"""Top-level orchestration for the reproducible DER baseline run."""

import hashlib
import logging
import random
from pathlib import Path

import numpy as np

from src.contracts import RunConfig
from src.modeling import ResearchBaseline

DEFAULT_RUN_CONFIG = RunConfig()
LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed the local RNGs used by the training pipeline."""

    random.seed(seed)
    np.random.seed(seed)


def file_sha256(path: Path) -> str:
    """Hash a file in chunks so large submissions do not spike memory."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(config: RunConfig = DEFAULT_RUN_CONFIG) -> ResearchBaseline:
    """Train the baseline and emit the submission file for the pinned config."""

    seed_everything(config.seed)
    LOGGER.info(
        "[run] starting pipeline with train=%s test=%s submission=%s",
        config.train_path,
        config.test_path,
        config.submission_path,
    )
    baseline = ResearchBaseline(config.baseline_config()).fit(config.train_path)
    baseline.predict_test(config.test_path, config.submission_path)
    LOGGER.info(
        "[solution] submission_sha256=%s path=%s",
        file_sha256(config.submission_path),
        config.submission_path,
    )
    return baseline
