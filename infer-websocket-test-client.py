#!/usr/bin/env python3

import os
import sys
import re
import random
import itertools
import time
from datetime import datetime
import asyncio
import websockets
import numpy as np
import soundfile

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

SCHEMA = os.environ.get('SCHEMA', 'ws')
HOST = os.environ.get('HOST', 'localhost')
PORT = int(os.environ.get('PORT', 7411))

INPUT_VOICE = 'sage'
DEFAULT_TARGET_VOICE = 'voicevox_speaker_43'
URL_PREFIX = f'{SCHEMA}://{HOST}:{PORT}/v1/voice_conversion?input_voice={INPUT_VOICE}&target_voice='

TEST_DATA_DIR = 'websocket-test-client-data'

ts = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def save_to_wav(log_prefix, filename, audio_data, sample_rate=24000, subtype='PCM_16'):
    soundfile.write(filename, np.frombuffer(audio_data, dtype=np.int16), sample_rate, subtype=subtype)
    print(f'{log_prefix}Saved "{filename}" with {len(audio_data)} bytes of audio data')

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
        #saveToWav(filename=f'{TEST_DATA_DIR}/WHOLE_{audio_key}.wav', audioData=b''.join(audio_parts_array))
    print()
    return audio_map2

async def run_test_client(client_num, target_voice, audio_key, audio_parts, output_filename):
    url = URL_PREFIX + target_voice
    log_prefix = f'[client {client_num:02d}] '
    first_part_sent_ts = None
    first_part_received_ts = None
    end_message_received_ts = None
    received_total = 0

    async with websockets.connect(url, additional_headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as websocket:
        async def receiver():
            nonlocal first_part_received_ts, end_message_received_ts, received_total
            buffer = b''
            data_num = 1
            while True:
                try:
                    data = await asyncio.wait_for(websocket.recv(), timeout=1)
                    if isinstance(data, bytes):
                        recv_ts = time.time()
                        if first_part_received_ts is None:
                            first_part_received_ts = recv_ts
                        delta_with_realtime = recv_ts - first_part_received_ts - len(buffer) / 24000 / 2
                        print(
                            f'[{ts()}]{log_prefix}Received data {data_num} of size {len(data)},'
                            f'delta with realtime = {delta_with_realtime * 1000:+.1f} ms'
                        )
                        buffer += data
                        data_num += 1
                    elif isinstance(data, str):
                        if data == 'end_message':
                            end_message_received_ts = time.time()
                            print(f'[{ts()}]{log_prefix}Received end_message')
                            break
                except asyncio.TimeoutError:
                    continue  # periodically checking if we need to stop
            save_to_wav(log_prefix=log_prefix, filename=output_filename, audio_data=buffer)
            received_total = len(buffer)
            print(f'{log_prefix}Receiver stopped')

        receive_task = asyncio.create_task(receiver())
        sent_total = 0
        for i, audio_part in enumerate(audio_parts, start=1):
            print(f'[{ts()}]{log_prefix}Sending "{audio_key}", part {i}, size {len(audio_part)}')
            await websocket.send(audio_part)
            sent_total += len(audio_part)
            if first_part_sent_ts is None:
                first_part_sent_ts = time.time()
            #audio_part_duration = len(audio_part) / 24000 / 2
            #await asyncio.sleep(audio_part_duration / 4)  # simulate stream which is 4 times faster than realtime
        await websocket.send('end_message')
        end_message_sent_ts = time.time()
        await receive_task

        receiving_elapsed = end_message_received_ts - first_part_received_ts
        received_audio_duration = received_total / 24000 / 2
        result = [
            f'============= {log_prefix}REPORT =============',
            f'Delay from first part sent till first part received: {(first_part_received_ts - first_part_sent_ts) * 1000:.2f} ms',
            f'Sending all parts took: {(end_message_sent_ts - first_part_sent_ts) * 1000:.1f} ms',
            f'Receiving all parts took: {receiving_elapsed * 1000:.1f} ms',
            f'Original audio duration: {sent_total / 24000 / 2:.3f} s',
            f'Received audio duration: {received_audio_duration:.3f} s',
            f'Receiving was {received_audio_duration / receiving_elapsed:.2f} times faster than realtime',
            f'==============================================',
        ]
        print(log_prefix + f'\n{log_prefix}'.join(result))

async def try_run_test_client(client_num, target_voice, audio_key, audio_parts, output_filename):
    try:
        await run_test_client(client_num, target_voice, audio_key, audio_parts, output_filename)
    except Exception as ex:
        print(f'CLIENT {client_num:02d} DIED WITH EXCEPTION: {ex}')

async def main():
    usage_message = f'Usage: {sys.argv[0]} <number_of_parallel_clients> [<target_voice>]'
    if len(sys.argv) < 2:
        print(usage_message)
        return 1
    try:
        clients_count = int(sys.argv[1])
    except ValueError:
        print(usage_message)
        return 1
    target_voice = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_TARGET_VOICE

    audio_files_parts = read_audio_files_parts()

    shuffled_keys = list(audio_files_parts.keys())
    random.shuffle(shuffled_keys)
    audio_keys = list(itertools.islice(itertools.cycle(shuffled_keys), clients_count))

    tasks = [try_run_test_client(
        client_num=i,
        target_voice=target_voice,
        audio_key=audio_key,
        audio_parts=audio_files_parts[audio_key],
        output_filename=f'{TEST_DATA_DIR}/output_{i:02d}.wav'
    ) for i, audio_key in enumerate(audio_keys, start=1)]
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == '__main__':
    asyncio.run(main())
