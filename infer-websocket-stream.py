#!/usr/bin/env python3

import os
import json
import ssl
import uuid
import logging
from urllib.parse import urlparse, parse_qs
import asyncio
import websockets

logger = logging.getLogger('infer-websocket-stream')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

AUTH_TOKEN = os.environ['AUTH_TOKEN']

SSL_CERT = os.environ['SSL_CERT_FILENAME']
SSL_KEY  = os.environ['SSL_KEY_FILENAME']

def error_message(message, details_to_log=None):
    if details_to_log:
        logger.error(f'{message} "{details_to_log}"')
    else:
        logger.error(f'{message}')
    return json.dumps({'error': message})

async def handler(websocket, path):
    params = parse_qs(urlparse(path).query)
    if 'auth_token' not in params:
        await websocket.send(error_message('No authentication token'))
        return
    auth_token = params['auth_token'][0]
    if auth_token != AUTH_TOKEN:
        await websocket.send(error_message('Bad authentication token', details_to_log=auth_token))
        return

    session_id = str(uuid.uuid4())
    log_prefix = f'session_id={session_id}: '
    logger.info(f'{log_prefix}Incoming connection')
    try:
        while True:
            data = await websocket.recv()
            if isinstance(data, str):
                pass
            elif isinstance(data, bytes):
                pass  # TODO
            else:
                await websocket.send(error_message('Received unrecognized data type', details_to_log=data))
                continue
    except websockets.exceptions.ConnectionClosedOK as ex:
        logger.info(f'{log_prefix}Disconnected OK: {ex}')
    except websockets.exceptions.ConnectionClosedError as ex:
        logger.error(f'{log_prefix}Disconnected with error: {ex}')

ssl_context = None
if os.path.isfile(SSL_CERT) and os.path.isfile(SSL_KEY):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(SSL_CERT, keyfile=SSL_KEY)
    logger.info('SSL certificates found, using encryption')
else:
    logger.warning('SSL certificates NOT FOUND, launching without encryption')

async def main():
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
