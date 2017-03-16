# Tensorflow setup on Linux instance
# http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html
# https://www.tensorflow.org/install/install_linux

# Download CUDA
wget -O cuda.deb https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1404-8-0-local-ga2_8.0.61-1_amd64-deb

# Ensure same name is used
yes | sudo dpkg -i cuda.deb
yes | sudo apt-get update
yes | sudo apt-get install cuda

# Install libcupti-dev library
yes | sudo apt-get install libcupti-dev
