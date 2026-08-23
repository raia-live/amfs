FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STATUS_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persist history across restarts when a volume is mounted at /data.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080

CMD ["python", "app.py"]
