# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set work directory
WORKDIR /home/user/app

# Install system dependencies (must run as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with UID 1000 to match Hugging Face's default user
RUN useradd -m -u 1000 user

# Install Python dependencies using the CPU-only index for PyTorch to keep build size small and fast
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application files and set ownership to our user
COPY --chown=user . .

# Ensure static/uploads and static/audio directories exist with correct owner permissions
RUN mkdir -p static/uploads static/audio && \
    chown -R user:user static/uploads static/audio && \
    chmod -R 755 static/uploads static/audio

# Switch to the non-root user
USER user

# Expose the port the app runs on
EXPOSE 7860

# Command to run the app using uvicorn on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
