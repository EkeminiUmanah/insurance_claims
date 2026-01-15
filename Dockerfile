# Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding large data via .dockerignore)
COPY . .

# Expose API port
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
