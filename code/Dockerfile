FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and dataset into the image
COPY ./ ./
COPY ../datasets/ datasets/

# Default entrypoint runs the main script
ENTRYPOINT ["python", "igreedy.py"]
