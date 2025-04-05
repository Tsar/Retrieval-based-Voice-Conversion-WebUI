#!/usr/bin/env python3

import os
import json
import ssl
import uuid
import logging
from urllib.parse import urlparse, parse_qs
from typing import Optional
import asyncio
import websockets
import numpy as np

import realtime_rvc_processor

logger = logging.getLogger('infer-websocket-stream')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

SSL_CERT = os.environ['SSL_CERT_FILENAME']
SSL_KEY  = os.environ['SSL_KEY_FILENAME']

rvc_processor: Optional[realtime_rvc_processor.RVCProcessor] = None

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
    if 'transpose_by' not in params:
        await websocket.send(error_message('No transpose_by in query params', log_prefix=log_prefix, details_to_log=params))
        return

    target_voice = params['target_voice'][0]
    transpose_by_str = params['transpose_by'][0]
    try:
        transpose_by = int(transpose_by_str)
    except ValueError:
        await websocket.send(error_message('Bad transpose_by value', log_prefix=log_prefix, details_to_log=transpose_by_str))
        return

    logger.info(f'{log_prefix}Starting voice conversion to {target_voice} transposed by {transpose_by}')
    rvc_processor.rvc.change_key(transpose_by)
    try:
        while True:
            data = await websocket.recv()
            if isinstance(data, str):
                print(data)
                pass
            elif isinstance(data, bytes):
                #np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                #processed = rvc_processor.process_audio_block(data)
                #await websocket.send(processed)
                print('Received bytes')
            else:
                await websocket.send(error_message('Received unrecognized data type', details_to_log=data))
                continue
    except websockets.exceptions.ConnectionClosedOK as ex:
        logger.info(f'{log_prefix}Disconnected OK: {ex}')
    except websockets.exceptions.ConnectionClosedError as ex:
        logger.error(f'{log_prefix}Disconnected with error: {ex}')

async def main():
    global rvc_processor
    rvc_processor = realtime_rvc_processor.RVCProcessor(
        pth_path='assets/weights/voicevox_speaker_43.pth',
        index_path='logs/voicevox_speaker_43/added_IVF567_Flat_nprobe_1_voicevox_speaker_43_v2.index',
        samplerate=24000,
        pitch=12,
    )
    rvc_processor.start_vc()

    ssl_context = None
    if os.path.isfile(SSL_CERT) and os.path.isfile(SSL_KEY):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(SSL_CERT, keyfile=SSL_KEY)
        logger.info('SSL certificates found, using encryption')
    else:
        logger.warning('SSL certificates NOT FOUND, launching without encryption')

    try:
        async with websockets.serve(handler, host='', port=7411, ssl=ssl_context) as server:
            logger.info('WebSocket server started')
            await server.serve_forever()
    except Exception as e:
        logger.error('Unhandled exception', exc_info=e)
    finally:
        logger.info('WebSocket server stopped')

if __name__ == '__main__':
    asyncio.run(main())
