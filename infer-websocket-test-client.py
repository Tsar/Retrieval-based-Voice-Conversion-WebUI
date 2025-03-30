#!/usr/bin/env python3

import os
import asyncio
import websockets

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

TARGET_VOICE = 'voicevox_speaker_43'
TRANSPOSE_BY = 6
URL = f'ws://localhost:7411/v1/voice_conversion?target_voice={TARGET_VOICE}&transpose_by={TRANSPOSE_BY}'

async def main():
    async with websockets.connect(URL, additional_headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as websocket:
        await websocket.send('TODO')
        data = await websocket.recv()
        print(data)

if __name__ == '__main__':
    asyncio.run(main())
