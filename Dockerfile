# Use the official Python 3.11 image
FROM python:3.11-slim

# Set up a working directory
WORKDIR /app

# Install any system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    evo \
    numpy \
    scipy \
    scikit-image \
    lpips \
    pillow \
    tqdm \
    plyfile

# Copy the project files into the container (if applicable)
COPY viz.py /app/
