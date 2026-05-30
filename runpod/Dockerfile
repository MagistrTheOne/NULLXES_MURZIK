# NULLXES MURZIK training image for RunPod
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NCCL_SOCKET_IFNAME=ens1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git rsync rclone sentencepiece \
    && rm -rf /var/lib/apt/lists/*

COPY runpod/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# FlashAttention (build on RunPod GPU pod if wheel fails)
RUN pip install flash-attn --no-build-isolation || true

# Clone LlamaFactory — replace with NULLXES fork when murzik_moe is registered
RUN git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git /opt/LlamaFactory \
    && pip install -e /opt/LlamaFactory

COPY . /workspace/NULLXES_MURZIK
ENV PYTHONPATH="/workspace/NULLXES_MURZIK:${PYTHONPATH}"

CMD ["bash"]
