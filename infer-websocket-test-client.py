#!/usr/bin/env python3

import os
import re
import random
import time
import asyncio
import websockets
import numpy as np
import soundfile

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

INPUT_VOICE = 'sage'
TARGET_VOICE = 'voicevox_speaker_43'
URL = f'ws://localhost:7411/v1/voice_conversion?input_voice={INPUT_VOICE}&target_voice={TARGET_VOICE}'

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

    first_part_sent_ts = None
    first_part_received_ts = None
    end_message_received_ts = None
    received_total = 0

    async with websockets.connect(URL, additional_headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as websocket:
        async def receiver():
            nonlocal first_part_received_ts, end_message_received_ts, received_total
            buffer = b''
            while True:
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=1)
                    if isinstance(data, bytes):
                        if first_part_received_ts is None:
                            first_part_received_ts = time.time()
                        print(f'Received data of size {len(data)}')
                        buffer += data
                    elif isinstance(data, str):
                        if data == 'end_message':
                            end_message_received_ts = time.time()
                            print('Received end_message')
                            break
                except asyncio.TimeoutError:
                    continue  # periodically checking if we need to stop
            saveToWav(filename=f'{TEST_DATA_DIR}/OUTPUT.wav', audioData=buffer)
            received_total = len(buffer)
            print('Receiver stopped')

        receive_task = asyncio.create_task(receiver())
        sent_total = 0
        for i, audio_part in enumerate(audio_files_parts[chosen_audio_key], start=1):
            print(f'Sending "{chosen_audio_key}", part {i}, size {len(audio_part)}')
            await websocket.send(audio_part)
            sent_total += len(audio_part)
            if first_part_sent_ts is None:
                first_part_sent_ts = time.time()
            #audio_part_duration = len(audio_part) / 24000 / 2
            #await asyncio.sleep(audio_part_duration / 4)  # simulate stream which is 4 times faster than realtime
        await websocket.send('end_message')
        end_message_sent_ts = time.time()
        await receive_task
        print(f'Delay from first part sent till first part received: {(first_part_received_ts - first_part_sent_ts) * 1000:.2f} ms')
        print(f'Sending all parts took: {(end_message_sent_ts - first_part_sent_ts) * 1000:.2f} ms')
        receiving_elapsed = end_message_received_ts - first_part_received_ts
        print(f'Receiving all parts took: {receiving_elapsed * 1000:.2f} ms')
        print(f'Original audio duration: {sent_total / 24000 / 2:.3f} s')
        received_audio_duration = received_total / 24000 / 2
        print(f'Received audio duration: {received_audio_duration:.3f} s')
        print(f'Receiving was {received_audio_duration / receiving_elapsed:.2f} times faster than realtime')

if __name__ == '__main__':
    asyncio.run(main())
