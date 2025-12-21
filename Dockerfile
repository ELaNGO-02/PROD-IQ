# Use official Python runtime as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt mcp_requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r mcp_requirements.txt

# Copy entire project
COPY . .

# Create user for security (HuggingFace best practice)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH=$HOME/.local/bin:$PATH

# HuggingFace Spaces specific
ENV HF_HOME=/tmp/huggingface

# Expose port (default for HF Spaces is 7860, but we can change)
EXPOSE 7860

# Generate artifacts on startup (if not present)
RUN if [ ! -d "ml_core/artifacts" ]; then \
    python scripts/generate_artifacts.py; \
    fi

# Start MCP server
CMD ["python", "mcp_server/server.py"]
