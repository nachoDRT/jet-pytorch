FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
        python3.10 python3-pip libjpeg-dev zlib1g-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
WORKDIR /app

RUN pip install --upgrade pip

RUN pip install --no-cache-dir "numpy<2"

RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.1.2+cu118 \
        torchvision==0.16.2+cu118

COPY dockerfiles/rtx4090/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip check

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

ENV WANDB_API_KEY=<your_token>
ENV HUGGINGFACE_HUB_TOKEN=<your_token>

COPY . .

ENV PYTHONPATH=/app

RUN chmod +x docker_src/docker_start.sh
CMD ["bash", "docker_src/docker_start.sh"]
