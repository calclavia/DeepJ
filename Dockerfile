FROM python:3.6

COPY requirements.txt /src/

# Install app dependencies
RUN pip3 install --no-cache-dir -r /src/requirements.txt

# Bundle app source
COPY . /src/

CMD ["bash"]