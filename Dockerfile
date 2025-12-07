# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set metadata labels
LABEL maintainer="Till Ermold <till.ermold@code.berlin>"
LABEL version="1.0"
LABEL description="Docker image for Swiss German ASR project"

# Create a non-root user
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies as root
COPY requirements_local.txt .
ENV PIP_DEFAULT_TIMEOUT=1000
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --resume-retries 20 -r requirements_local.txt

# Copy the application code
COPY . .

# Set ownership
RUN chown -R appuser:appuser /app

# Pre-create cache directories with correct ownership
RUN mkdir -p /home/appuser/.cache/huggingface /home/appuser/.cache/whisper && \
    chown -R appuser:appuser /home/appuser/.cache

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Command to run the FastAPI backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
