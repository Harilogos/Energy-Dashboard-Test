# Multi-stage Dockerfile for Energy Generation Dashboard

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements files
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    if [ -f requirements-test.txt ]; then pip install -r requirements-test.txt; fi

# Stage 3: Development image (includes test dependencies)
FROM dependencies as development

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command for development
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Testing image
FROM development as testing

# Switch back to root for test setup
USER root

# Install additional testing tools
RUN pip install pytest-xdist pytest-html pytest-json-report

# Create test directories
RUN mkdir -p /app/test_reports /app/cache/test /app/logs/test

# Change ownership back to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Command for running tests
CMD ["python", "tests/test_runner.py", "--all", "--verbose"]

# Stage 5: Production image (minimal, no test dependencies)
FROM base as production

# Copy only production requirements
COPY requirements.txt ./

# Install only production dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip cache purge

# Copy application code (excluding test files)
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser frontend/ ./frontend/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /app/Data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Production command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# Stage 6: CI/CD image (for GitHub Actions)
FROM testing as ci

# Switch to root for CI setup
USER root

# Install additional CI tools
RUN apt-get update && apt-get install -y \
    jq \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for CI
RUN pip install \
    flake8 \
    black \
    isort \
    mypy \
    bandit \
    safety \
    coverage \
    pytest-benchmark

# Create CI directories
RUN mkdir -p /app/ci_reports /app/security_reports

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# CI command (can be overridden)
CMD ["python", "tests/test_validation.py"]
