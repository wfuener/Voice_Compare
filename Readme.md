# Voice Compare
A Python script for comparing two voice samples using ECAPA-TDNN speaker embeddings.

High level overview of the script:
1. Downloads the hardcoded videos off youtube and places them in the `../data` directory.
2. Extracts the audio from the video
3. 


## System Requirements
This has only been tested on Ubuntu so you may need to install different OS level packages for your OS.

```bash
sudo apt install ffmpeg 
```

## Python Requirements
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Run
python voice_compare.py
