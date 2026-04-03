FROM gcr.io/kaggle-images/python@sha256:02c72a7c98e5e0895056901d9c715d181cd30eae392491235dfea93e6d0de3ed

LABEL org.opencontainers.image.title="DER Kaggle CPU runtime"
LABEL org.opencontainers.image.description="Pinned Kaggle CPU notebook environment matching BUILD_DATE=20260319-213519 and GIT_COMMIT=c292018b280631cbfe6f4f16fc6a84f2786b5f86."
LABEL org.opencontainers.image.source="https://github.com/Kaggle/docker-python"

WORKDIR /workspace

RUN python3 -m venv --system-site-packages --without-pip /opt/der-uv-env

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/der-uv-env

CMD ["/bin/bash"]
