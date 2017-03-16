# Python files
yes | sudo apt-get install python3 python3-pip python3-dev

# For Python midi
yes | sudo apt-get install libasound2-dev python3-augeas swig

# Install Python MIDI
git clone https://github.com/vishnubob/python-midi/
cd python-midi
sudo python3 setup.py install
cd ../
