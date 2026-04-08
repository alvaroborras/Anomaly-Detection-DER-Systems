#!/usr/bin/env python3

"""Generate a runnable Kaggle notebook that recreates this repo with `%%writefile` cells."""

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT_PATH = Path("notebook.ipynb")
KAGGLE_INPUT_PATH = "/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems"
KAGGLE_WORKING_PATH = "/kaggle/working"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def discover_source_files(repo_root: Path) -> list[Path]:
    """Return the local source files that must exist for the current import graph."""

    src_dir = repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing source package directory: {src_dir}")
    main_path = src_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"Missing entrypoint module: {main_path}")

    src_files = sorted(path.relative_to(repo_root) for path in src_dir.rglob("*.py"))
    if not src_files:
        raise FileNotFoundError(f"No Python source files found under: {src_dir}")
    return src_files


def setup_cell_source(source_files: list[Path]) -> str:
    """Create parent directories and ensure the local repo root is importable."""

    directories = sorted({path.parent.as_posix() for path in source_files if path.parent != Path(".")})
    directory_lines = "\n".join(f'    "{directory}",' for directory in directories) or "    "
    return f"""from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
SOURCE_DIRS = [
{directory_lines}
]

for relative_dir in SOURCE_DIRS:
    (PROJECT_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"project_root={{PROJECT_ROOT}}")
print(f"competition_data_exists={{Path('{KAGGLE_INPUT_PATH}').exists()}}")
print(f"kaggle_working_exists={{Path('{KAGGLE_WORKING_PATH}').exists()}}")
"""


def writefile_cell_source(relative_path: Path, contents: str) -> str:
    return f"%%writefile {relative_path.as_posix()}\n{contents}"


def run_cell_source() -> str:
    return """import sys

!{sys.executable} -m src.main
"""


def preview_cell_source() -> str:
    return f"""import hashlib
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


submission_path = Path("{KAGGLE_WORKING_PATH}") / "submission.csv"
print(f"submission_exists={{submission_path.exists()}}")
if submission_path.exists():
    print(f"submission_sha256={{file_sha256(submission_path)}}")
    pd.read_csv(submission_path).head()
"""


def notebook_cells(repo_root: Path) -> list[dict]:
    source_files = discover_source_files(repo_root)
    manifest = "\n".join(f"- `{path.as_posix()}`" for path in source_files)
    cells = [
        markdown_cell(
            "# DER Kaggle submission notebook\n\n"
            "This notebook was generated from the repo source tree.\n\n"
            "Run the cells from top to bottom to recreate the local package layout with `%%writefile`, "
            "then execute the unchanged `src.main` module entrypoint.\n\n"
            "## Included files\n"
            f"{manifest}"
        ),
        code_cell(setup_cell_source(source_files)),
    ]
    for relative_path in source_files:
        contents = (repo_root / relative_path).read_text()
        cells.append(code_cell(writefile_cell_source(relative_path, contents)))
    cells.append(
        markdown_cell(
            "## Run the pipeline\n\n"
            "This cell executes the same repo entrypoint (`python -m src.main`) in a fresh Python subprocess so notebook reruns pick up the latest written files."
        )
    )
    cells.append(code_cell(run_cell_source()))
    cells.append(
        markdown_cell(
            "## Inspect the generated submission\n\n"
            "This optional cell prints the submission SHA256 and previews the first rows of `submission.csv`."
        )
    )
    cells.append(code_cell(preview_cell_source()))
    return cells


def build_notebook(repo_root: Path) -> dict:
    return {
        "cells": notebook_cells(repo_root),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(repo_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(repo_root)
    output_path.write_text(json.dumps(notebook, indent=2) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Notebook path to write (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_path = args.output
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    written_path = write_notebook(repo_root, output_path)
    print(f"Wrote notebook to {written_path}")


if __name__ == "__main__":
    main()
