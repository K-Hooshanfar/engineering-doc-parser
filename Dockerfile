# syntax=docker/dockerfile:1

# GPU image (CUDA). Requires NVIDIA Container Toolkit on the host.
# Build:  docker build -t engineering-doc-parser .
# Crop:   docker run --gpus all -v "%CD%\data:/data" engineering-doc-parser run.py --config /data/run.yaml
# Extract: docker run --gpus all -v "%CD%\data:/data" engineering-doc-parser extract.py --config /data/extract.yaml

ARG BASE_IMAGE=ultralytics/ultralytics:8.3.70-python-3.10
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/data/hf_cache

WORKDIR /app

# Ultralytics image already ships torch+CUDA, ultralytics, opencv, pillow, tqdm.
COPY requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-runtime.txt

COPY src/ /app/src/
COPY train.py run.py extract.py /app/
COPY configs/ /app/configs/

ENTRYPOINT ["python"]
CMD ["run.py", "--help"]
