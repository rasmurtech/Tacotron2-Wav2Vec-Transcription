# Tacotron2-Wav2Vec-Transcription

A Python script that uses the Wav2Vec2 model to transcribe .wav files and generates a .txt file for training a Tacotron2 text-to-speech model.

## Overview

This script transcribes audio files in the WAV format using the Wav2Vec2 model from the Hugging Face Transformers library. It processes each WAV file, generates its corresponding transcription, and saves the transcription along with the file path in a .txt file. This .txt file can then be used for training a Tacotron2 text-to-speech model.

## Prerequisites

- Python 3.6 or higher
- PyTorch
- torchaudio
- transformers

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/Tacotron2-Wav2Vec-Transcription.git

markdown
Copy code

2. Install the required packages:

pip install torch torchaudio transformers

markdown
Copy code

## Usage

1. Modify the `transcribe_wav2vec.py` script to set the appropriate input and output paths.

2. Run the script:

python transcribe_wav2vec.py

csharp
Copy code

## Output

The output .txt file contains one line per transcribed WAV file, with the file path followed by a pipe character `|` and the transcription. The format is as follows:

/content/TTS-TT2/wavs/1.wav|transcription of the first file
/content/TTS-TT2/wavs/2.wav|transcription of the second file
...
/content/TTS-TT2/wavs/n.wav|transcription of the nth file

Copy code
