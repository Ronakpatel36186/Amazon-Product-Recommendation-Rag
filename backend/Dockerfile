FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for FAISS and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy entire project into container
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]