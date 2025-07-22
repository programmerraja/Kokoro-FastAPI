# Kokoro-FastAPI
A minimal Kokoro-FastAPI server


docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest

docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.4


docker build -t kokoro-cpu -f kokoro-cpu.dockerfile .

docker run -v "$(pwd)/api/src/:/api/src/" -p 9801:9801 kokoro-cpu