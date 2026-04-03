# Kaggle CPU environment

This repo can be run locally against the exact Kaggle CPU notebook image that
produced the captured environment snapshot.

Environment fingerprint:

- Kaggle image: `gcr.io/kaggle-images/python:v168`
- Image digest: `sha256:02c72a7c98e5e0895056901d9c715d181cd30eae392491235dfea93e6d0de3ed`
- Kaggle `BUILD_DATE`: `20260319-213519`
- Kaggle `GIT_COMMIT`: `c292018b280631cbfe6f4f16fc6a84f2786b5f86`
- Python: `3.12.12`
- OS: `Ubuntu 22.04.5 LTS`

The package snapshot is recorded in `pip_list.txt` and the OS package snapshot
is recorded in `dpkg_query.txt`.

## Build the wrapper image

```bash
docker build --platform linux/amd64 -t der-kaggle-cpu .
```

The wrapper image does not install any extra Python packages. It pins the exact
Kaggle CPU base image, creates `/opt/der-uv-env` with
`--system-site-packages --without-pip`, and sets `UV_PROJECT_ENVIRONMENT` so
`uv run` reuses the Kaggle-installed packages. `--without-pip` is required
because this Kaggle image does not ship `ensurepip`.

On Apple Silicon, keep `--platform linux/amd64` for both `docker build` and
`docker run`; the Kaggle CPU image is published for `linux/amd64`.

## Run the fixed Docker workflow

`run_docker.sh` is now a single-purpose wrapper for reproducing the pinned
`main.py` run inside the Docker image. It mounts:

- the repo root at `/workspace`
- `data/` at `/kaggle/input/competitions/cyber-physical-anomaly-detection-for-der-systems`
- `kaggle-working/` at `/kaggle/working`

Run it with no arguments:

```bash
./run_docker.sh
```

That command is the canonical reproducible entrypoint.

Train/test paths, seed, parallelism, model settings, Docker image name,
platform, and output location are all pinned in code so future users get the
same published recipe by default.

The submission is always written to `kaggle-working/submission.csv` on the
host.

`run_docker.sh` does not accept any extra arguments, and `main.py` itself does
not accept any arguments.

## Verify the container

```bash
python --version
cat /etc/os-release | grep PRETTY_NAME
python - <<'PY'
import catboost, numpy, pandas, pyarrow, sklearn, tqdm, xgboost

print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
print("scikit-learn", sklearn.__version__)
print("xgboost", xgboost.__version__)
print("catboost", catboost.__version__)
print("tqdm", tqdm.__version__)
PY
```

## Notes

- `run_docker.sh` is the canonical way to reproduce the pinned Docker run.
- The script always runs `uv run python main.py` inside the container.
- The Docker image configures `uv` to use `/opt/der-uv-env`, which inherits the
-  Kaggle image's preinstalled Python packages.
- Docker will not reproduce Kaggle's host kernel exactly, so `uname -r` may
-  differ locally.
