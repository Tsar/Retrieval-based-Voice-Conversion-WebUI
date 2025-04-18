#!/usr/bin/env python3

import time
import uuid
import ssl

import asyncio
import aiohttp

import torch

MODEL = 'xiangling-eng-30-epochs-with-pitch'
URL = f'https://rvc-inference-predictor.2dchan-inference.knative.k8s.finomen.net/v2/models/{MODEL}/infer'

P_LEN = 224
SKIP_HEAD = 200
RETURN_LENGTH = 20
RETURN_LENGTH2 = 20

C_feats = torch.load('../data/feats.pt')
C_p_len = torch.full((10,), P_LEN, dtype=torch.long)
C_cache_pitch = torch.load('../data/cache_pitch.pt')
C_cache_pitchf = torch.load('../data/cache_pitchf.pt')

async def net_g_inference(session):
    t0 = time.perf_counter()
    B = 1
    feats = C_feats[:B]
    p_len = C_p_len[:B]
    cache_pitch = C_cache_pitch[:B]
    cache_pitchf = C_cache_pitchf[:B]
    rnd = torch.rand(1, 192, feats.size(1))

    request = {
        'id': str(uuid.uuid4()),
        'inputs': [
            {
                'name': 'phone',
                'shape': list(feats.shape),
                'datatype': 'FP32',
                'data': feats.flatten().tolist(),
            },
            {
                'name': 'phone_lengths',
                'shape': list(p_len.shape),
                'datatype': 'INT64',
                'data': p_len.tolist(),
            },
            {
                'name': 'pitch',
                'shape': list(cache_pitch.shape),
                'datatype': 'INT64',
                'data': cache_pitch.flatten().tolist(),
            },
            {
                'name': 'pitchf',
                'shape': list(cache_pitchf.shape),
                'datatype': 'FP32',
                'data': cache_pitchf.flatten().tolist(),
            },
            {
                'name': 'ds',
                'shape': [1],
                'datatype': 'INT64',
                'data': [0],
            },
            {
                'name': 'rnd',
                'shape': list(rnd.shape),
                'datatype': 'FP32',
                'data': rnd.flatten().tolist(),
            },
        ]
    }

    t1 = time.perf_counter()
    async with session.post(URL, json=request) as resp:
        if resp.status != 200:
            print(f'Request failed with code {resp.status}: {await resp.text()}')
            return
        result = await resp.json()

    t2 = time.perf_counter()
    assert result['outputs'][0]['name'] == 'audio'
    assert result['outputs'][0]['shape'] == [B, 1, 89600]
    print(f'net_g inference [B={feats.size(0)}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

async def main():
    # Temporary hack while SSL certificate is broken on the server
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    no_verify_ssl_connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=no_verify_ssl_connector) as session:
        while True:
            await net_g_inference(session)

if __name__ == '__main__':
    asyncio.run(main())
