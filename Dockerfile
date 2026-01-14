# Use Python 3.12
FROM python:3.12-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your existing code (main.py, models.py, fer_backend.py)
COPY . .

# Command to run FastAPI using Uvicorn
# Google Cloud Run expects the app to listen on port 8080
# We added '--workers 1' to ensure the Memory Buffer is not split across processes
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]