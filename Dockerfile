# We use the official, easy-to-use Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python form creating cache files (.pyc) and buffering output (so that logs are displayed immediately)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy list of libraries and installing them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Copy all other code for project container
COPY . .

# Open 8000 port
EXPOSE 8000

# Server start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]