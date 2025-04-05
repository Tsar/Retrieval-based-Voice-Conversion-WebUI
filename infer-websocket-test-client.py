#!/usr/bin/env python3

import os
import re
import random
import asyncio
import websockets
import numpy as np
import soundfile

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

TARGET_VOICE = 'voicevox_speaker_43'
TRANSPOSE_BY = 6
URL = f'ws://localhost:7411/v1/voice_conversion?target_voice={TARGET_VOICE}&transpose_by={TRANSPOSE_BY}'

TEST_DATA_DIR = 'websocket-test-client-data'

def saveToWav(filename, audioData, sampleRate=24000, subtype='PCM_16'):
    soundfile.write(filename, np.frombuffer(audioData, dtype=np.int16), sampleRate, subtype=subtype)
    print(f'Saved "{filename}" with {len(audioData)} bytes of audio data')

def read_audio_files_parts():
    audio_map = {}
    for filename in os.listdir(TEST_DATA_DIR):
        match = re.match(r'(msg_server_audio_.+?)_part_(\d+).bin', filename)
        if match:
            with open(f'{TEST_DATA_DIR}/{filename}', 'rb') as fPart:
                partContent = fPart.read()
            audio_key = match.group(1)
            part_num = int(match.group(2))
            if audio_key in audio_map:
                audio_map[audio_key][part_num] = partContent
            else:
                audio_map[audio_key] = {part_num: partContent}
    audio_map2 = {}
    for audio_key in audio_map:
        audio_by_part_num = audio_map[audio_key]
        audio_parts_array = []
        for part_num in sorted(audio_by_part_num.keys()):
            audio_parts_array.append(audio_by_part_num[part_num])
        audio_map2[audio_key] = audio_parts_array
    print(f'Found {len(audio_map2)} audio messages:')
    for audio_key in audio_map2:
        audio_parts_array = audio_map2[audio_key]
        print(f' * "{audio_key}" - {len(audio_parts_array)} parts')
        saveToWav(filename=f'{TEST_DATA_DIR}/WHOLE_{audio_key}.wav', audioData=b''.join(audio_parts_array))
    return audio_map2

async def main():
    audio_files_parts = read_audio_files_parts()
    chosen_audio_key = random.choice(list(audio_files_parts.keys()))

    async with websockets.connect(URL, additional_headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as websocket:
        stopEvent = asyncio.Event()

        async def receiver():
            buffer = b''
            while not stopEvent.is_set():
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=1)
                    print(f'Received data of size {len(data)}')
                    buffer += data
                except asyncio.TimeoutError:
                    continue  # periodically checking if we need to stop
            saveToWav(filename='OUTPUT.wav', audioData=buffer)
            print('Receiver stopped gracefully')

        asyncio.create_task(receiver())
        for i, audio_part in enumerate(audio_files_parts[chosen_audio_key], start=1):
            print(f'Sending "{chosen_audio_key}", part {i}')
            await websocket.send(audio_part)
            audio_part_duration = len(audio_part) / 24000 / 2
            await asyncio.sleep(audio_part_duration / 3)  # simulate stream which is 3 times faster than realtime
        await websocket.send('end_message')
        await asyncio.sleep(5)
        stopEvent.set()
        await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(main())
