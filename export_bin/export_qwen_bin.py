#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_qwen_bin.py (extended)
-----------------------------
Exports Qwen-VL safetensors → single .bin

Layout:
[config header]
[int32 text_weight_bits]
[int32 group_size]
[int32 vision_weight_bits]
[weights stream...]

Supports:
- FP32
- FP16
- INT8 (groupwise symmetric)
- INT4 (groupwise symmetric, packed)
"""

import argparse
import json
import struct
import numpy as np
import torch
from safetensors.torch import safe_open
from typing import Dict, Callable, List

# ----------------------------------------------------------------------
# Config header keys (UNCHANGED)
# ----------------------------------------------------------------------

KEYS = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",

    "vision_hidden_size",
    "vision_depth",
    "vision_patch_size",
    "vision_spatial_merge_size",
    "vision_temporal_patch_size",
    "vision_num_heads",
    "vision_intermediate_size",
    "out_hidden_size",

    "image_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "video_token_id",
]

# ----------------------------------------------------------------------
# Tensor Provider
# ----------------------------------------------------------------------

class TensorProvider:
    def __init__(self, loader: Callable[[], torch.Tensor]):
        self._loader = loader

    def load(self) -> torch.Tensor:
        return self._loader()


def collect_effective_keys(f) -> Dict[str, TensorProvider]:
    out = {}
    for k in f.keys():
        out[k] = TensorProvider(lambda kk=k: f.get_tensor(kk))
    return out


# ----------------------------------------------------------------------
# Ordering (UNCHANGED)
# ----------------------------------------------------------------------

def reorder_keys_for_write(all_keys: List[str], num_layers=28) -> List[str]:
    ordered = []

    if "model.language_model.embed_tokens.weight" in all_keys:
        ordered.append("model.language_model.embed_tokens.weight")

    lm_sub = [
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    ]

    for s in lm_sub:
        for i in range(num_layers):
            k = f"model.language_model.layers.{i}.{s}"
            if k in all_keys:
                ordered.append(k)

    tail = [
        "model.language_model.norm.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight",
    ]

    for k in tail:
        if k in all_keys:
            ordered.append(k)

    for k in sorted(all_keys):
        if k not in ordered:
            ordered.append(k)

    return ordered


# ----------------------------------------------------------------------
# Config Header Writer
# ----------------------------------------------------------------------

def write_config_header(config: dict, fout):
    cfg = {}
    cfg.update(config)
    cfg.update(config.get("text_config", {}))

    vc = config.get("vision_config", {})
    cfg.update({
        "vision_hidden_size": vc.get("hidden_size"),
        "vision_depth": vc.get("depth"),
        "vision_patch_size": vc.get("patch_size"),
        "vision_spatial_merge_size": vc.get("spatial_merge_size"),
        "vision_temporal_patch_size": vc.get("temporal_patch_size"),
        "vision_num_heads": vc.get("num_heads"),
        "vision_intermediate_size": vc.get("intermediate_size"),
        "out_hidden_size": vc.get("out_hidden_size"),
    })

    for k in KEYS:
        v = cfg.get(k)
        if isinstance(v, int):
            fout.write(struct.pack("<i", v))
        elif isinstance(v, float):
            fout.write(struct.pack("<f", v))


# ----------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------

def quantize_groupwise(x: np.ndarray, bits: int, group_size: int):
    assert bits in (4, 8)
    qmax = (1 << (bits - 1)) - 1

    x = x.reshape(-1)
    n = len(x)

    scales = []
    qvals = []

    for i in range(0, n, group_size):
        g = x[i:i + group_size]
        maxv = np.max(np.abs(g)) + 1e-8
        scale = maxv / qmax
        q = np.round(g / scale).astype(np.int8)
        q = np.clip(q, -qmax, qmax)

        scales.append(scale)
        qvals.append(q)

    return np.concatenate(qvals), np.array(scales, dtype=np.float32)


def pack_int4(q: np.ndarray) -> np.ndarray:
    assert q.dtype == np.int8
    q = np.clip(q, -8, 7).astype(np.int8)
    if len(q) % 2 != 0:
        q = np.append(q, 0)

    lo = q[0::2] & 0x0F
    hi = (q[1::2] & 0x0F) << 4
    return (lo | hi).astype(np.uint8)


# ----------------------------------------------------------------------
# Weight Streaming
# ----------------------------------------------------------------------

def is_vision(name: str) -> bool:
    return name.startswith("model.visual")


def write_weights_streaming(
    providers,
    ordered_keys,
    fout,
    text_bits: int,
    vision_bits: int,
    group_size: int,
):
    for name in ordered_keys:
        t = providers[name].load().cpu().float()
        arr = t.numpy().reshape(-1)

        is_vis = is_vision(name)
        bits = vision_bits if is_vis else text_bits

        # FP32
        if bits == 32:
            fout.write(arr.astype(np.float32).tobytes())

        # FP16
        elif bits == 16:
            fout.write(arr.astype(np.float16).tobytes())

        # INT8
        elif bits == 8:
            q, scales = quantize_groupwise(arr, 8, group_size)
            fout.write(scales.tobytes())
            fout.write(q.astype(np.int8).tobytes())

        # INT4
        elif bits == 4:
            q, scales = quantize_groupwise(arr, 4, group_size)
            packed = pack_int4(q)
            fout.write(scales.tobytes())
            fout.write(packed.tobytes())

        else:
            raise ValueError(bits)

        print(f"Wrote {name} bits={bits} shape={t.shape}", flush=True)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--text_bits", type=int, choices=[4, 8, 16, 32], default=32)
    p.add_argument("--vision_bits", type=int, choices=[16, 32], default=32)
    p.add_argument("--group_size", type=int, choices=[32, 64], default=32)

    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    with safe_open(args.input, framework="pt", device="cpu") as f:
        providers = collect_effective_keys(f)
        ordered = reorder_keys_for_write(list(providers.keys()))

        with open(args.output, "wb") as fout:
            # 1️⃣ config header
            write_config_header(config, fout)

            # 2️⃣ quant metadata
            fout.write(struct.pack("<i", args.text_bits))
            fout.write(struct.pack("<i", args.group_size))
            fout.write(struct.pack("<i", args.vision_bits))

            # 3️⃣ weights
            write_weights_streaming(
                providers,
                ordered,
                fout,
                args.text_bits,
                args.vision_bits,
                args.group_size,
            )

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
