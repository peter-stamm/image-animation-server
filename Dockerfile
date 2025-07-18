FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libgl1-mesa-glx git && \
    apt-get clean

# Set workdir
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
