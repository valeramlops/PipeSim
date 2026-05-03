# We use the official, easy-to-use Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python form creating cache files (.pyc) and buffering output (so that logs are displayed immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1 \
    # Path for local bin
    PATH="/home/appuser/.local/bin:$PATH"

# Create an unprivileged user for security
RUN useradd -m -s /bin/bash appuser

# install system dependencies for OpenCV and YOLO
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    libpq-dev \
    python3-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install --no-cache-dir uv

RUN uv pip install --system --no-cache torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple

# Copy list of libraries and installing them
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match

# Copy all other code for project container
COPY . .

# Create the necessary folders and assign permissions to our secure user
RUN mkdir -p /app/uploads/videos /app/uploads/images /app/logs /tmp/Ultralytics \
    && chown -R appuser:appuser /app /tmp/Ultralytics

# Open 8000 port
EXPOSE 8000

# Server start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]