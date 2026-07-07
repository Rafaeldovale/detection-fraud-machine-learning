# 1. Use official stable Ubuntu as base (Highly compatible with non-AVX CPUs)
FROM ubuntu:24.04

# 2. Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install Python 3 and Pip using Ubuntu packages (Pre-compiled for maximum compatibility)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy your pinned requirements file into the container
COPY requirements.txt /app/

# 6. Install the exact same versions from your .venv_wsl using pip
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# 7. Copy the application folders into the container
COPY ./api /app/api
COPY ./models /app/models

# 8. Expose the API port
EXPOSE 8000

# 9. Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
