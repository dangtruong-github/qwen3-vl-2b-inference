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
import re

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

def reorder_keys_for_write(all_keys: List[str], num_layers: int = 28) -> List[str]:
    """
    Return keys ordered according to the Qwen-like hierarchical structure:
    1. embed_tokens
    2. per-layer stack (0..num_layers-1) in fixed subkey order
    3. all remaining keys alphabetically
    """
    ordered = []

    # 1️⃣ Embedding
    if "model.language_model.embed_tokens.weight" in all_keys:
        ordered.append("model.language_model.embed_tokens.weight")

    # 2️⃣ Per-layer ordering pattern
    lm_subkeys = [
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

    for sub in lm_subkeys:
        for i in range(num_layers):
            prefix = f"model.language_model.layers.{i}."
            name = prefix + sub
            if name in all_keys:
                ordered.append(name)

    next_lm_vision_keys = [
        "model.language_model.norm.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.patch_embed.proj.weight",
        "model.visual.pos_embed.weight"
    ]

    for each_key in next_lm_vision_keys:
        if each_key in all_keys:
            ordered.append(each_key)

    vision_block_subkeys = [
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.qkv.bias",
        "attn.qkv.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc2.bias",
        "mlp.linear_fc2.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight"
    ]

    for sub in vision_block_subkeys:
        for i in range(24):
            prefix = f"model.visual.blocks.{i}."
            name = prefix + sub
            if name in all_keys:
                ordered.append(name)

    deep_stack_merger_subkeys = [
        "linear_fc1.bias",
        "linear_fc1.weight",
        "linear_fc2.bias",
        "linear_fc2.weight",
        "norm.bias",
        "norm.weight"
    ]

    for sub in deep_stack_merger_subkeys:
        for i in range(3):
            prefix = f"model.visual.deepstack_merger_list.{i}."
            name = prefix + sub
            if name in all_keys:
                ordered.append(name)

    merger_subkeys = [
        "linear_fc1.bias",
        "linear_fc1.weight",
        "linear_fc2.bias",
        "linear_fc2.weight",
        "norm.bias",
        "norm.weight"
    ]

    prefix = "model.visual.merger."
    for sub in merger_subkeys:
        name = prefix + sub
        if name in all_keys:
            ordered.append(name)
    
    # 3️⃣ Remaining keys (e.g., vision or lm_head)
    remaining = [k for k in all_keys if k not in ordered]
    ordered.extend(sorted(remaining))

    return ordered

def reorder_keys_by_group(ordered_keys: List[str]):
    new_ordered_keys = []
    tmp_keys = []
    for item in ordered_keys:
        numbers = re.findall(r'\.(\d+)\.', item)
        if not numbers:
            if tmp_keys:
                new_ordered_keys.append(tmp_keys)
            new_ordered_keys.append([item])
            tmp_keys = []
        else:
            if int(numbers[0]) == 0 and tmp_keys:
                new_ordered_keys.append(tmp_keys)
                tmp_keys = []
            tmp_keys.append(item)

    return new_ordered_keys

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
            print(f"Wrote config key {k}={v} (<i)")
        elif isinstance(v, float):
            fout.write(struct.pack("<f", v))
            print(f"Wrote config key {k}={v} (<f)")

# ----------------------------------------------------------------------
# Quantization helpers
# ----------------------------------------------------------------------

def quantize_rowwise(x: np.ndarray, bits: int, shape: tuple):
    """
    Quantize along last dimension (row-wise).
    """
    qmax = (1 << (bits - 1)) - 1

    x = x.reshape(shape)
    last_dim = shape[-1]
    x2 = x.reshape(-1, last_dim)

    maxv = np.max(np.abs(x2), axis=1, keepdims=True) + 1e-8
    scales = maxv / qmax

    q = np.round(x2 / scales).astype(np.int8)
    q = np.clip(q, -qmax, qmax)

    return q.flatten(), scales.flatten().astype(np.float32)

def quantize_groupwise(x: np.ndarray, bits: int, group_size: int):
    qmax = (1 << (bits - 1)) - 1
    
    # Reshape to (number_of_groups, group_size)
    # This assumes x.size is divisible by group_size
    x_reshaped = x.flatten().reshape(-1, group_size)
    
    # Calculate scales for all groups at once
    maxv = np.max(np.abs(x_reshaped), axis=1, keepdims=True) + 1e-8
    scales = maxv / qmax
    
    # Quantize
    q = np.round(x_reshaped / scales).astype(np.int8)
    q = np.clip(q, -qmax, qmax)
    
    return q.flatten(), scales.flatten().astype(np.float32)


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
    group_quantized: bool
):
    # Lists to buffer the bytes in memory before writing
    all_scales_bytes = []
    all_weights_bytes = []

    for name in ordered_keys:
        t = providers[name].load().cpu().float()
        arr = t.numpy().reshape(-1)

        is_vis = is_vision(name)
        bits = vision_bits if is_vis else text_bits

        # FP32 & FP16 (No separate scales)
        if bits == 32:
            all_weights_bytes.append(arr.astype(np.float32).tobytes())
        elif bits == 16:
            all_weights_bytes.append(arr.astype(np.float16).tobytes())

        # INT8
        elif bits == 8:
            if group_quantized or "norm" in name:
                q, scales = quantize_groupwise(arr, 8, group_size)
            else:
                q, scales = quantize_rowwise(arr, 8, t.shape)

            all_scales_bytes.append(scales.tobytes())
            all_weights_bytes.append(q.astype(np.int8).tobytes())

        # INT4
        elif bits == 4:
            if group_quantized or "norm" in name:
                q, scales = quantize_groupwise(arr, 4, group_size)
            else:
                q, scales = quantize_rowwise(arr, 4, t.shape)

            packed = pack_int4(q)
            all_scales_bytes.append(scales.tobytes())
            all_weights_bytes.append(packed.tobytes())

        else:
            raise ValueError(f"Unsupported bit depth: {bits}")

        print(f"Processed {name} (bits={bits}, shape={t.shape})", flush=True)

    # --- WRITING PHASE ---
    if all_scales_bytes:
        print("Writing scales to file...", flush=True)
        for scale_data in all_scales_bytes:
            fout.write(scale_data)

    print("Writing weights to file...", flush=True)
    for weight_data in all_weights_bytes:
        fout.write(weight_data)

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
    p.add_argument("--group_size", type=int, choices=[32, 64, 128], default=32)
    
    p.add_argument("--group_quantized", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        config = json.load(f)

    with safe_open(args.input, framework="pt", device="cpu") as f:
        providers = collect_effective_keys(f)
        ordered = reorder_keys_for_write(list(providers.keys()))

        ordered_new = reorder_keys_by_group(ordered)

        with open(args.output, "wb") as fout:
            # 1️⃣ config header
            write_config_header(config, fout)

            # 2️⃣ quant metadata
            fout.write(struct.pack("<i", args.text_bits))
            fout.write(struct.pack("<i", args.group_size))
            fout.write(struct.pack("<i", args.vision_bits))

            # pack bool as int32 (1=True, 0=False)
            group_flag = 1 if args.group_quantized else 0
            fout.write(struct.pack("<i", group_flag))

            print(f"Wrote config key text_bits={args.text_bits} (<i)")
            print(f"Wrote config key group_size={args.group_size} (<i)")
            print(f"Wrote config key vision_bits={args.vision_bits} (<i)")
            print(f"Wrote config key group_quantized={group_flag} (<i)")

            # 3️⃣ weights
            for ordered_group in ordered_new:
                write_weights_streaming(
                    providers,
                    ordered_group,
                    fout,
                    args.text_bits,
                    args.vision_bits,
                    args.group_size,
                    args.group_quantized,   # NEW
                )

    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
