FROM python:3.10-slim

WORKDIR /app

# Copy and install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app and ngrok wrapper
COPY app .

# Run the FastAPI app via ngrok
CMD ["python", "ngrok_server.py"]
