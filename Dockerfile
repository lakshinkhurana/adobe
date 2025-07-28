# Use a stable and full-featured base
FROM python:3.10-bullseye

# Set environment variables for performance and cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy code and install requirements
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create input/output dirs inside container if not mounted
RUN mkdir -p /app/input /app/output

# Use all CPU cores when needed
CMD ["python", "1a.py", "--input", "./input", "--output", "./output"]
