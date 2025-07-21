FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY ./app /app/app
COPY ./models /app/models


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
