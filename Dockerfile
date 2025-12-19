# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Make start script executable (if exists)
RUN if [ -f start.sh ]; then chmod +x start.sh; fi

# Expose port (Render will set PORT env variable)
EXPOSE $PORT

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden by Render
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
