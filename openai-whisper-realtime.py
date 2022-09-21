import sounddevice as sd
import numpy as np

import whisper

import asyncio
import queue
import sys

global_ndarray = None

model = whisper.load_model("base")

async def inputstream_generator():
	"""Generator that yields blocks of input data as NumPy arrays."""
	q_in = asyncio.Queue()
	loop = asyncio.get_event_loop()

	def callback(indata, frame_count, time_info, status):
		loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

	stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=24576, callback=callback)
	with stream:
		while True:
			indata, status = await q_in.get()
			yield indata, status
			
		
async def process_audio_buffer():
	global global_ndarray
	async for indata, status in inputstream_generator():
		#print(abs(indata[-1, -1]))
		
		if (global_ndarray is not None):
			global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
		else:
			global_ndarray = indata
			
		if (abs(indata[-1, -1]) > 100):
			continue
		else:
			local_ndarray = global_ndarray.copy()
			global_ndarray = None
			indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
			result = model.transcribe(indata_transformed, language='English')
			print(result["text"])
		#if (avgResult = np.average(indata.reshape(-1, n), axis=1))


async def main():
	print('\nActivating wire ...\n')
	audio_task = asyncio.create_task(process_audio_buffer())
	for i in range(100, 0, -1):
		await asyncio.sleep(1)
	audio_task.cancel()
	try:
		await audio_task
	except asyncio.CancelledError:
		print('\nWire was cancelled')


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')

