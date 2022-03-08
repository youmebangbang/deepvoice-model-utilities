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

# FORMATTING OF TRAINING AND VALIDATION .CSV FILES

Entries are deliminated with the | character. No quotes around text. Split at least 50 entries from your training.csv into your validation.csv

For Tacotron 2:

[wave file name with extension] | [text of audio]

example:  120.wav|Here is the text of my audio. It is some good text.

For Waveglow:

[wave file name with extension]

example:  120.wav

For Hifi-GAN:

[wave file name without extension] | [text of audio]

example:  120|Here is the text of my audio. It is some good text.


