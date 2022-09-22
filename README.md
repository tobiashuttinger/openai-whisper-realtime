# OpenAI Whisper Realtime

This is a quick experiment to achieve almost realtime transcription using Whisper.

## How to use

Install the requirements:
```
pip install -r requirements.txt
```

Run the script: 
```
python openai-whisper-realtime.py
```

Dependencies:
- Python > 3.7
- whisper
- sounddevice
- numpy
- asyncio

A very fast CPU or GPU is recommended.

## How it works

The systems default audio input is captured with python, split into small chunks and is then fed to OpenAI's original transcription function. It tries (currently rather poorly) to detect word breaks and doesn't split the audio buffer in those cases.
With how the model is designed, it doesn't make the most sense to do this, but i found it would be worth trying. It works acceptably well.


## ToDo:
- Improve transcription performance
- Improve detection of word breaks or pauses, split the buffer dynamically
- Refactoring
- Clean stdout
