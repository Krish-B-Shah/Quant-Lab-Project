# Multi-stage build for GatorAI quant platform
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash gatorai

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/processed data/raw && \
    chown -R gatorai:gatorai /app

# Switch to non-root user
USER gatorai

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, 'src'); from data.manager import DataManager; print('OK')" || exit 1

# Default command
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Development stage
FROM base as development

# Install additional dev dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort \
    mypy \
    jupyter \
    mkdocs \
    mkdocs-material \
    mkdocstrings[python]

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/processed data/raw && \
    chown -R gatorai:gatorai /app

# Switch to non-root user
USER gatorai

# Expose ports
EXPOSE 8501 8888

# Default command for development
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
