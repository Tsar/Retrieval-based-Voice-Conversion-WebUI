#!/usr/bin/env python3

import os
import json
import time
import ssl
import uuid
import itertools
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import asyncio
import websockets
import numpy as np

logger = logging.getLogger('infer-batch-websocket-server')

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

PORT = int(os.environ.get('PORT', 7411))

SSL_CERT = os.environ.get('SSL_CERT_FILENAME')
SSL_KEY  = os.environ.get('SSL_KEY_FILENAME')
ALLOW_UNENCRYPTED_SERVING = int(os.environ.get('ALLOW_UNENCRYPTED_SERVING', 0))

INPUT_VOICES_PITCH = {
    'coral': 4,
    'shimmer': 0,
    'sage': 5,
}

TARGET_VOICES_PITCH = {
    'voicevox_speaker_43': 8,
}

BLOCK_DURATION_MS = 150
BLOCK_SIZE = 24000 * 2 * BLOCK_DURATION_MS // 1000  # PCM16, 24 KHz, mono audio

class Task:
    def __init__(
        self,
        priority: int,
        sequence: int,
        target_voice: str,
        transpose_by: int,
        block: bytes,
        is_last_block: bool,
        websocket,
    ):
        self.priority = priority
        self.sequence = sequence
        self.target_voice = target_voice
        self.transpose_by = transpose_by
        self.block = block
        self.is_last_block = is_last_block
        self.websocket = websocket

    def __lt__(self, other) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence

global_sequence = itertools.count()

preprocess_queue = asyncio.PriorityQueue()

executor = ThreadPoolExecutor(max_workers=cpu_count())  # TODO: Replace with ProcessPoolExecutor?

async def preprocess_queue_dispatch_loop():
    loop = asyncio.get_running_loop()
    while True:
        task: Task = await preprocess_queue.get()
        try:
            loop.run_in_executor(executor, preprocess, task)
        finally:
            preprocess_queue.task_done()

def preprocess(task: Task):
    assert len(task.block) == BLOCK_SIZE
    block_f32 = np.frombuffer(task.block, dtype=np.int16).astype(np.float32) / 32768.0
    # TODO


def error_message(message, log_prefix='', details_to_log=None):
    if details_to_log:
        logger.error(f'{log_prefix}{message} "{details_to_log}"')
    else:
        logger.error(f'{log_prefix}{message}')
    return json.dumps({'error': message})

async def handler(websocket):
    session_id = str(uuid.uuid4())
    log_prefix = f'Session {session_id}: '
    logger.info(f'{log_prefix}Incoming connection')

    parsed_url = urlparse(websocket.request.path)
    if parsed_url.path != '/v1/voice_conversion':
        await websocket.send(error_message(f'Unsupported path {parsed_url.path}', log_prefix=log_prefix))
        return

    auth_header = websocket.request.headers.get('Authorization')
    if auth_header is None:
        await websocket.send(error_message('No Authorization header', log_prefix=log_prefix))
        return
    if not auth_header.startswith(BEARER_PREFIX):
        await websocket.send(error_message('Bad Authorization header', log_prefix=log_prefix, details_to_log=auth_header))
        return
    auth_token = auth_header[len(BEARER_PREFIX):]
    if auth_token != AUTH_TOKEN:
        await websocket.send(error_message('Bad authentication token', log_prefix=log_prefix, details_to_log=auth_token))
        return

    params = parse_qs(parsed_url.query)
    if 'target_voice' not in params:
        await websocket.send(error_message('No target_voice in query params', log_prefix=log_prefix, details_to_log=params))
        return
    target_voice = params['target_voice'][0]
    if target_voice not in TARGET_VOICES_PITCH:
        await websocket.send(error_message(
            f'Unsupported target_voice, only the following are supported: {list(TARGET_VOICES_PITCH.keys())}',
            log_prefix=log_prefix,
            details_to_log=params,
        ))
        return

    if 'transpose_by' in params:
        transpose_by_str = params['transpose_by'][0]
        try:
            transpose_by = int(transpose_by_str)
        except ValueError:
            await websocket.send(error_message('Bad transpose_by value', log_prefix=log_prefix, details_to_log=transpose_by_str))
            return
    elif 'input_voice' in params:
        input_voice = params['input_voice'][0]
        if input_voice not in INPUT_VOICES_PITCH:
            await websocket.send(error_message(
                f'Unsupported input_voice, only the following are supported: {list(INPUT_VOICES_PITCH.keys())}',
                log_prefix=log_prefix,
                details_to_log=params,
            ))
            return
        transpose_by = TARGET_VOICES_PITCH[target_voice] - INPUT_VOICES_PITCH[input_voice]
    else:
        await websocket.send(error_message('No transpose_by or input_voice in query params', log_prefix=log_prefix, details_to_log=params))
        return

    logger.info(f'{log_prefix}Starting voice conversion to {target_voice} transposed by {transpose_by}')

    buffer = b''
    block_num = 0
    msg_start_ts_ms: Optional[int] = None

    try:
        while True:
            data = await websocket.recv()
            if msg_start_ts_ms is None:
                msg_start_ts_ms = int(time.time() * 1000)

            if isinstance(data, bytes):
                buffer += data
                while len(buffer) >= BLOCK_SIZE:
                    preprocess_queue.put_nowait(
                        Task(
                            priority=msg_start_ts_ms + BLOCK_DURATION_MS * block_num,
                            sequence=next(global_sequence),
                            target_voice=target_voice,
                            transpose_by=transpose_by,
                            block=buffer[:BLOCK_SIZE],
                            is_last_block=False,
                            websocket=websocket,
                        )
                    )
                    buffer = buffer[BLOCK_SIZE:]
                    block_num += 1
            elif isinstance(data, str):
                if data == 'end_message':
                    assert len(buffer) < BLOCK_SIZE
                    preprocess_queue.put_nowait(
                        Task(
                            priority=msg_start_ts_ms + BLOCK_DURATION_MS * block_num,
                            sequence=next(global_sequence),
                            target_voice=target_voice,
                            transpose_by=transpose_by,
                            block=buffer.ljust(BLOCK_SIZE, b'\x00'),
                            is_last_block=True,
                            websocket=websocket,
                        )
                    )
                    buffer = b''
                    msg_start_ts_ms = None
                    block_num = 0
                else:
                    logger.error(f'{log_prefix}Unrecognized text received: "{data}"')
            else:
                await websocket.send(error_message('Received unrecognized data type', details_to_log=data))
                continue
    except websockets.exceptions.ConnectionClosedOK as ex:
        logger.info(f'{log_prefix}Disconnected OK: {ex}')
    except websockets.exceptions.ConnectionClosedError as ex:
        logger.error(f'{log_prefix}Disconnected with error: {ex}')

async def main():
    ssl_context = None
    if SSL_CERT and SSL_KEY and os.path.isfile(SSL_CERT) and os.path.isfile(SSL_KEY):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT, keyfile=SSL_KEY)
        logger.info('SSL certificates found, using encryption')
    else:
        if ALLOW_UNENCRYPTED_SERVING == 1:
            logger.warning('SSL certificates NOT FOUND, launching without encryption')
        else:
            logger.warning('SSL certificates NOT FOUND, unencrypted serving prohibited')
            return

    asyncio.create_task(preprocess_queue_dispatch_loop())

    try:
        async with websockets.serve(handler, host='', port=PORT, ssl=ssl_context) as server:
            logger.info(f'WebSocket server started on port {PORT}')
            await server.serve_forever()
    except Exception as e:
        logger.error('Unhandled exception', exc_info=e)
    finally:
        logger.info('WebSocket server stopped')

if __name__ == '__main__':
    asyncio.run(main())
