# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY requirements.txt .
COPY service.py .
COPY common.py .
COPY config.py .
COPY classifier.py .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "75"]