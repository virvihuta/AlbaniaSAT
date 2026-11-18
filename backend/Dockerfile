# Docker image for the AlbaniaSAT backend (Regis)
FROM python:3.12-slim

WORKDIR /app

# Install the libraries first, so Docker can reuse this step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY . .

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]