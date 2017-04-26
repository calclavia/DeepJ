# Install all Python related things
# Run this file from the parent dir of the repository

sudo apt-get update

# Python files
yes | sudo apt-get install python3 python3-pip python3-dev

# For Python midi
yes | sudo apt-get install libasound2-dev python3-augeas swig

# Install Python MIDI
git clone https://github.com/vishnubob/python-midi/
cd python-midi
git checkout feature/python3
sudo python3 setup.py install
cd ../

# Install project requirements
cd music-generator
sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl 
sudo pip3 install -r requirements.txt
cd ../
