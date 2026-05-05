# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories and set permissions for Hugging Face (User 1000)
RUN mkdir -p static/uploads static/audio && \
    chmod -R 777 static/uploads static/audio

# Expose the port the app runs on
EXPOSE 7860

# Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
