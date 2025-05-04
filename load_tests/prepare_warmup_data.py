#!/usr/bin/env python3

import torch
from torchfcpe.models_infer import spawn_wav2mel
from torchfcpe.tools import DotDict

# For hubert
C_input_wav_batch = torch.load('data/input_wav_batch.pt').float().cpu().repeat(5, 1)
assert C_input_wav_batch.shape == torch.Size([50, 35840])

# For fcpe
C_input_wav_batch_cropped = torch.load('data/input_wav_batch_cropped.pt').float().cpu().repeat(10, 1)
assert C_input_wav_batch_cropped.shape == torch.Size([100, 3200])

mel_extractor_args = DotDict()
mel_extractor_args.mel = DotDict({'fmax': 8000, 'fmin': 0, 'hop_size': 160, 'n_fft': 1024, 'num_mels': 128, 'sr': 16000, 'win_size': 1024})
mel_extractor = spawn_wav2mel(mel_extractor_args, 'cpu')

# For net_g
P_LEN = 224
C_feats = torch.load('data/feats.pt').float().cpu().repeat(5, 1, 1)
C_p_len = torch.full((50,), P_LEN, dtype=torch.long).cpu().unsqueeze(-1)
C_cache_pitch = torch.load('data/cache_pitch.pt').cpu().repeat(5, 1)
C_cache_pitchf = torch.load('data/cache_pitchf.pt').float().cpu().repeat(5, 1)
C_sid = torch.zeros(50, dtype=torch.long).cpu().unsqueeze(-1)

HUBERT_BATCH_SIZE = 4
FCPE_BATCH_SIZE = 8
NET_G_BATCH_SIZE = 5

def to_api_datatype(dtype: torch.dtype):
    if dtype == torch.float32:
        return 'FP32'
    if dtype == torch.int64:
        return 'INT64'
    raise RuntimeError(f'Unsupported torch data type for API: {dtype}')

def save_tensor_to_binary_file(filename_prefix: str, tensor: torch.Tensor):
    shape_str = '-'.join([str(sz) for sz in tensor.shape])
    with open(f'{filename_prefix}__{to_api_datatype(tensor.dtype)}__{shape_str}.bin', 'wb') as f:
        f.write(tensor.numpy().tobytes())

if __name__ == '__main__':
    save_tensor_to_binary_file('hubert__input_wav', C_input_wav_batch[:HUBERT_BATCH_SIZE])

    mel = mel_extractor(C_input_wav_batch_cropped[:FCPE_BATCH_SIZE], sample_rate=16000)
    save_tensor_to_binary_file('fcpe__mel', mel)

    save_tensor_to_binary_file('net_g__feats', C_feats[:NET_G_BATCH_SIZE])
    save_tensor_to_binary_file('net_g__p_len', C_p_len[:NET_G_BATCH_SIZE])
    save_tensor_to_binary_file('net_g__pitch', C_cache_pitch[:NET_G_BATCH_SIZE])
    save_tensor_to_binary_file('net_g__pitchf', C_cache_pitchf[:NET_G_BATCH_SIZE])
    save_tensor_to_binary_file('net_g__sid', C_sid[:NET_G_BATCH_SIZE])
