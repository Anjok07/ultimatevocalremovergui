# Ultimate Vocal Remover GUI v5.6 (Ubuntu, no-call-home version)

This is a fork of https://github.com/Anjok07/ultimatevocalremovergui

I created it for 2 reasons:

1. Provide an up-to-date installation instructions for Ubuntu 24.

2. Remove all code that talks to the project home (e.g. update checking, VIP codes etc.) except for model downloading.

## Installation on Ubuntu 24

1. Python 3.9 seems to be required. I could not get it to work on newer versions of Python. So:

```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```

2. Certain packages are required:

```
sudo apt install python3.9 cmake python3.9-tk software-properties-common libcairo2-dev python3.9-dev python3.9-distutils ffmpeg python3.9-venv libgirepository-2.0-dev gir1.2-gst-plugins-base-1.0
```

3. Python depencencies installation requires an additional env variable to make one deprecated package happy (otherwise it insists on using its replacement which breaks depencencies):

```
python3.9 -m venv venv
source venv/bin/activate
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -r requirements.txt
```
