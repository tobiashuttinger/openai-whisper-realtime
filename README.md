# OpenAI Whisper Realtime

This is a quick experiment to achieve almost realtime transcription using Whisper.

## How to use

Run the script `openai-whisper-realtime.py` with Python 3.7 or greater. 

Dependencies:
whisper
sounddevice
numpy
asyncio

A very fast CPU or GPU is recommended.

## How it works

The systems default audio input is captured with python, split into small chunks and is then fed to OpenAI's original transcription function. It tries (currently rather poorly) to detect word breaks and doesn't split the audio buffer in those cases. 


## ToDo:
- Improve transcription performance
- Improve detection of word breaks or pauses, split the buffer dynamically
- Refactoring
- Clean stdout
