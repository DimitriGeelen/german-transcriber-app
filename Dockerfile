# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent interactive prompts during installation
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# - ffmpeg is required by openai-whisper
# - git and rustc might be needed for pip install dependencies (like tokenizers)
# - Clean up apt lists afterwards to keep image size down
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        build-essential \
        # Install Rust via rustup (needed for tokenizers)
        && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
        # Add rust to path for subsequent commands
        export PATH="/root/.cargo/bin:${PATH}" && \
    # Clean up
    apt-get purge -y --auto-remove build-essential curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add Rust to PATH permanently for subsequent stages/commands
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
# Use a virtual environment within the container for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (default 8000, but respects PORT env var)
# Note: This informs Docker, but doesn't automatically use the PORT env var value.
# Use the default, user can map differently at runtime.
EXPOSE 8000

# Define the command to run the application
# Use environment variables for host/port configuration at runtime
# Default to 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 