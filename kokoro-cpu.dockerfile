FROM python:3.10-slim

ENV HF_HOME=/root/.cache/huggingface/hub

# RUN apt-get update 
# && apt-get install -y 
    # build-essential \
    # cmake \
    # && rm -rf /var/lib/apt/lists/*

RUN pip install numpy websockets asyncio loguru kokoro  inflect pydantic torch

# COPY requirements.txt .

# RUN pip install -r requirements.txt

COPY . .


EXPOSE 9802

RUN chmod +x download.sh
RUN ./download.sh

CMD ["python", "server.py"]