#!/usr/bin/env python3

import os
import time
import random
import uuid

import asyncio
import aiohttp

import torch

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

MODEL = 'xiangling-eng'
URL = f'https://rvc-inference-predictor.2dchan-inference.knative.finomen.net/v2/models/{MODEL}/infer'

RAND_BATCH_SIZE = bool(int(os.environ.get('RAND_BATCH_SIZE', 1)))

P_LEN = 224

C_feats = torch.load('../data/feats.pt')
C_p_len = torch.full((10,), P_LEN, dtype=torch.long)
C_cache_pitch = torch.load('../data/cache_pitch.pt')
C_cache_pitchf = torch.load('../data/cache_pitchf.pt')
C_sid = torch.zeros(10, dtype=torch.long)

async def net_g_inference(session):
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, 10)
    else:
        B = 10
    feats = C_feats[:B]
    p_len = C_p_len[:B]
    cache_pitch = C_cache_pitch[:B]
    cache_pitchf = C_cache_pitchf[:B]
    sid = C_sid[:B]

    request = {
        'id': str(uuid.uuid4()),
        'inputs': [
            {
                'name': 'feats',
                'shape': list(feats.shape),
                'datatype': 'FP32',
                'data': feats.flatten().tolist(),
            },
            {
                'name': 'p_len',
                'shape': [B, 1],
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
                'name': 'sid',
                'shape': [B, 1],
                'datatype': 'INT64',
                'data': sid.tolist(),
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

    assert result['model_name'] == MODEL
    assert len(result['outputs']) == 1
    output = result['outputs'][0]
    assert output['name'] == 'audio'
    assert output['datatype'] == 'FP32'
    res_last_dim_size = 8800 if MODEL == 'xiangling-eng' else 8000
    assert output['shape'] == [B, 1, res_last_dim_size]
    assert len(output['data']) == B * res_last_dim_size

    print(f'net_g inference [B={B}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms')

async def main():
    async with aiohttp.ClientSession(headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN}) as session:
        while True:
            await net_g_inference(session)

if __name__ == '__main__':
    asyncio.run(main())
