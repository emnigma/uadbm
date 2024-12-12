ARG BUILDPLATFORM="arm64"

FROM --platform=${BUILDPLATFORM} ubuntu:22.04

RUN apt update && \
    apt install python3.11 python3.11-venv python3-pip -y && \
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /app

COPY docker_requirements.txt requirements.txt

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install --user --upgrade pip && \
    pip install --user -U pip setuptools wheel && \
    pip install --user -r requirements.txt

COPY dataloaders dataloaders
COPY mains mains
COPY models models
COPY trainers trainers
COPY utils utils
COPY recon_one.py recon_one.py
COPY content content
COPY config.json config.json

ENTRYPOINT ["python3", "recon_one.py"]
