FROM python:3.11-slim

# Install ffmpeg runtime only
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies using pre-built wheels (avoid compiling av from source)
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/

# Expose port
EXPOSE 3000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]
