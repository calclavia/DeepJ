FROM ubuntu:latest

# Install Python (Most of them are for development?)
RUN \
  apt-get update && \
  apt-get install -y git python3 python3-dev python3-pip python3-dev python3-tk libasound2-dev python3-augeas swig

COPY requirements.txt /src/

# Install app dependencies
RUN pip3 install --no-cache-dir -r /src/requirements.txt

# Install Python-midi
RUN git clone https://github.com/vishnubob/python-midi && cd python-midi && git checkout feature/python3 && python3 setup.py install

# Bundle app source
COPY . /src/

CMD ["bash"]