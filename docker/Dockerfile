FROM continuumio/miniconda3:4.10.3

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

COPY src/ocr_microservice ./ocr_microservice

SHELL ["conda", "run", "-n", "ocr_microservice_cpu_env", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "-n", "ocr_microservice_cpu_env", "python", "./ocr_microservice/message_bus_consumer/message_bus_consumer.py"]
