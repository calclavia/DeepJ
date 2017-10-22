FROM ubuntu:latest

# Install Python & dependencies
RUN \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip git fluidsynth curl

# Download Soundfonts
RUN mkdir src && \
    curl -o /src/acoustic_grand_piano.sf2 http://zenvoid.org/audio/acoustic_grand_piano_ydp_20080910.sf2

COPY requirements.txt /src/

# Install app dependencies
RUN pip3 install --no-cache-dir -r /src/requirements.txt

# Bundle app source
COPY . /src/

CMD ["bash"]