FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    dnsutils \
    patch

COPY src/ /app/src/
COPY pyproject.toml /app/
COPY environment.yml /app/

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "ocr_microservice_cpu_env", "/bin/bash", "-c"]

RUN pip install -e .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["conda", "run", "-n", "ocr_microservice_cpu_env", "python", "/app/src/ocr_microservice/message_bus_consumer/message_bus_consumer.py"]
# ENTRYPOINT ["tail", "-f", "/dev/null"] # Use for debugging