FROM python:3.9-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git ffmpeg libsm6 libxext6

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

WORKDIR /app

# Create a virtual environment
RUN python -m venv /app/venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD . /app/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt && \
    exec gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
