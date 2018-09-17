FROM nvidia/cuda:9.2-cudnn7-runtime

# Install Python & dependencies
RUN \
  apt-get update && \
  apt-get install -y bzip2 curl build-essential fluidsynth curl lame git && \
  curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
  bash /tmp/miniconda.sh -bfp /usr/local && \
  rm -rf /tmp/miniconda.sh

RUN conda install python=3.6 -y

COPY requirements.txt /tmp/

# Install app dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN pip install git+https://github.com/NVIDIA/apex
RUN conda install -c conda-forge onnx
RUN pip install git+https://github.com/onnx/onnx-tensorflow

CMD ["bash"]