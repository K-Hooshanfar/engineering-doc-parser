# syntax=docker/dockerfile:1

# Use a slim Python base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies commonly required by OpenCV and scientific stacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt /app/requirements.txt
COPY requirements-dev.txt /app/requirements-dev.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install -r /app/requirements-dev.txt

# Copy the whole project (so inference can import local modules)
COPY . /app

# Set working directory to Inference folder by default
WORKDIR /app/Inference

# Default entrypoint runs the inference script; pass args with `docker run ... -- <args>`
ENTRYPOINT ["python", "inference_yolo.py"]
# Allow users to append command-line args, e.g., model path, image dir, etc.
CMD ["--help"]


