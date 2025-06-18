# Use Python 3.12 slim image as base
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    gcc \
    supervisor \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
RUN mkdir -p /var/log/supervisor && chown appuser:appuser /var/log/supervisor


USER appuser

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser redis_stream/requirements.txt .

# Install Python dependencies with security updates
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
RUN mkdir services

# Copy the rest of the application
COPY --chown=appuser:appuser .env .env
COPY --chown=appuser:appuser redis_stream ./redis_stream
COPY --chown=appuser:appuser services/init_redis_stream.py services/__init__.py
COPY --chown=appuser:appuser services/redis_service.py services/
COPY --chown=appuser:appuser config ./config




# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Copy file cấu hình Supervisor
COPY redis_stream/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Chạy supervisor để quản lý nhiều script
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]