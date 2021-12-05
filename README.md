# Utilities for training and inferring deep voice models.

Infer a text file into multiple phrases that can be redone, edited, or removed.

Train Nvidia Tacotron2, Waveglow, and Hifi-Gan models.

"Easy" setup and usage for those new to deepvoice.

# INSTALLATION

It is highly recommend to install anaconda and create a virtual environment to work from. 

git clone this repository

cd deepvoice-model-utilities

git clone https://github.com/NVIDIA/tacotron2.git

cd tacotron2

git submodule init

git submodule update

cd ..

git clone https://github.com/NVIDIA/waveglow.git

cd waveglow

git submodule init

git submodule update

cd ..

git clone https://github.com/jik876/hifi-gan.git

Install latest pip version of pytorch (CUDA binaries are inlcuded!):
https://pytorch.org/get-started/locally/

pip install -r requirements.txt --user

Copy the tacotron2 and waveglow replace files into the installed directories

# USAGE

Video tutorial coming soon!
