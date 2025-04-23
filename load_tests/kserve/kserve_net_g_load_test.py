#!/usr/bin/env python3

import os
import time
import random
import uuid
import json

import asyncio
import aiohttp

import torch

from aiohttp_trace import request_tracer

BEARER_PREFIX = 'Bearer '

AUTH_TOKEN = os.environ['AUTH_TOKEN']

MODEL = 'xiangling-eng'
URL = f'https://rvc-inference-predictor.2dchan-inference.knative.finomen.net/v2/models/{MODEL}/infer'

RAND_BATCH_SIZE = bool(int(os.environ.get('RAND_BATCH_SIZE', 0)))

P_LEN = 224

C_feats = torch.load('../data/feats.pt').float().cpu()
C_p_len = torch.full((10,), P_LEN, dtype=torch.long).cpu()
C_cache_pitch = torch.load('../data/cache_pitch.pt').cpu()
C_cache_pitchf = torch.load('../data/cache_pitchf.pt').float().cpu()
C_sid = torch.zeros(10, dtype=torch.long).cpu()

async def net_g_inference(session):
    t0 = time.perf_counter()
    if RAND_BATCH_SIZE:
        B = random.randint(1, 10)
    else:
        B = 10
    feats: torch.Tensor = C_feats[:B]
    p_len = C_p_len[:B]
    cache_pitch = C_cache_pitch[:B]
    cache_pitchf = C_cache_pitchf[:B]
    sid = C_sid[:B]

    inference_header = {
        'id': str(uuid.uuid4()),
        'inputs': [
            {
                'name': 'feats',
                'shape': list(feats.shape),
                'datatype': 'FP32',
                'parameters': {'binary_data_size': feats.numel() * feats.element_size()},
            },
            {
                'name': 'p_len',
                'shape': [B, 1],
                'datatype': 'INT64',
                'parameters': {'binary_data_size': p_len.numel() * p_len.element_size()},
            },
            {
                'name': 'pitch',
                'shape': list(cache_pitch.shape),
                'datatype': 'INT64',
                'parameters': {'binary_data_size': cache_pitch.numel() * cache_pitch.element_size()},
            },
            {
                'name': 'pitchf',
                'shape': list(cache_pitchf.shape),
                'datatype': 'FP32',
                'parameters': {'binary_data_size': cache_pitchf.numel() * cache_pitchf.element_size()},
            },
            {
                'name': 'sid',
                'shape': [B, 1],
                'datatype': 'INT64',
                'parameters': {'binary_data_size': sid.numel() * sid.element_size()},
            },
        ],
        'outputs': [
            {
                'name': 'audio',
                'parameters': {'binary_data': True},
            }
        ]
    }
    inference_header_data = json.dumps(inference_header).encode('UTF-8')
    request_body = (
        inference_header_data +
        feats.numpy().tobytes() +
        p_len.numpy().tobytes() +
        cache_pitch.numpy().tobytes() +
        cache_pitchf.numpy().tobytes() +
        sid.numpy().tobytes()
    )
    print(f'Request body size: {len(request_body)}, inference header size: {len(inference_header_data)}')

    t1 = time.perf_counter()
    async with session.post(URL, data=request_body, headers={'Inference-Header-Content-Length': str(len(inference_header_data))}) as resp:
        if resp.status != 200:
            print(f'Request failed with code {resp.status}: {await resp.text()}')
            return
        resp_inference_header_length = resp.headers.get('Inference-Header-Content-Length')
        result = await resp.read()
    t2 = time.perf_counter()

    assert resp_inference_header_length is not None
    resp_inference_header_length = int(resp_inference_header_length)
    resp_inference_header = json.loads(result[:resp_inference_header_length])
    assert resp_inference_header['model_name'] == MODEL
    assert len(resp_inference_header['outputs']) == 1
    output = resp_inference_header['outputs'][0]
    assert output['name'] == 'audio'
    assert output['datatype'] == 'FP32'
    res_last_dim_size = 8800 if MODEL == 'xiangling-eng' else 8000
    assert output['shape'] == [B, 1, res_last_dim_size]
    assert 'data' not in output
    assert len(result) == resp_inference_header_length + B * res_last_dim_size * 4

    t3 = time.perf_counter()
    print(f'net_g inference [B={B}]: {(t2 - t1) * 1000:.1f} ms, prep: {(t1 - t0) * 1000:.1f} ms, post: {(t3 - t2) * 1000:.1f} ms')

async def main():
    async with aiohttp.ClientSession(
        headers={'Authorization': BEARER_PREFIX + AUTH_TOKEN},
        trace_configs=[request_tracer()],
    ) as session:
        while True:
            await net_g_inference(session)

if __name__ == '__main__':
    asyncio.run(main())
