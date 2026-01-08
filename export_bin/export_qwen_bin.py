#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_qwen_bin.py
------------------
One-shot exporter for Qwen-VL checkpoints (non-quantized):
- Reads a single input safetensors file containing standard FP16/BF16 weights.
- Writes [config header + ordered weights] directly to a single .bin

Usage
-----
# Typical
python export_qwen_bin.py \
  --input model.safetensors \
  --config config.json \
  --output qwen-vl-2b.bin
"""

import argparse
import json
import re
import struct
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from safetensors.torch import safe_open
import torch

# ----------------------------------------------------------------------
# Qwen-Specific Config keys & ordering buckets
# ----------------------------------------------------------------------

# Standard Qwen config parameters to be written to the binary header
# The order of parameters in the binary header
KEYS = [
    # Text Model Params (often in 'text_config')
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "rms_norm_eps",

    # Vision Model Params (often in 'vision_config' or root)
    "vision_hidden_size",      # Mapped from vision_config.hidden_size
    "vision_depth",            # Mapped from vision_config.depth
    "vision_patch_size",       # Mapped from vision_config.patch_size
    "vision_spatial_merge_size", # Mapped from vision_config.spatial_merge_size
    "vision_temporal_patch_size", # Mapped from vision_config.temporal_patch_size
    "vision_num_heads",        # Mapped from vision_config.num_heads
    "vision_intermediate_size",# Mapped from vision_config.intermediate_size
    "out_hidden_size",         # Mapped from vision_config.out_hidden_size

    # Special Tokens (often int)
    "image_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
    "video_token_id",
]

# Qwen layer name suffixes
CATEGORIES = [
    # Global/Vision
    "transformer.wte.weight", # Token embeddings
    "transformer.ln_f.weight", # Final layer norm
    # Attention
    "attn.c_attn.weight",   # QKV projection (linear)
    "attn.c_attn.bias",     # QKV bias (optional)
    "attn.c_proj.weight",   # Attention output projection
    "attn.c_proj.bias",     # Attention output bias (optional)
    # MLP (SwiGLU)
    "mlp.w2.weight",        # Gate projection (w2)
    "mlp.w1.weight",        # Up projection (w1)
    "mlp.c_proj.weight",    # Down projection (c_proj)
    # Layer Norms (RMSNorm typically)
    "ln_1.weight",          # Pre-attention layer norm
    "ln_2.weight",          # Pre-MLP layer norm
]

# Regex to capture the block index in Qwen-style naming (e.g., 'transformer.h.0.attn...')
_BLOCK_RE = re.compile(r"transformer\.h\.(\d+)\.")

# ----------------------------------------------------------------------
# Tensor Provider (Simplified - no dequantization)
# ----------------------------------------------------------------------


class TensorProvider:
    """Holds a callable to load a tensor on demand."""

    def __init__(self, loader: Callable[[], torch.Tensor]):
        self._loader = loader

    def load(self) -> torch.Tensor:
        return self._loader()


def collect_effective_keys(f) -> Dict[str, TensorProvider]:
    """
    Build a mapping: effective_name -> TensorProvider
    All keys are exposed as-is, assuming they are standard weights (FP16/BF16/FP32).
    """
    effective: Dict[str, TensorProvider] = {}
    keys = list(f.keys())

    for key in keys:
        # Plain tensor passthrough
        def make_plain_loader(k=key):
            def _ld():
                # Load the tensor directly from safetensors
                return f.get_tensor(k)
            return _ld

        effective[key] = TensorProvider(make_plain_loader())

    return effective


# ----------------------------------------------------------------------
# Ordering & Writing
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


def write_config_header(config: dict, fout) -> None:
    """
    Write Qwen3-VL config keys in fixed order as <i (int) or <f (float) little-endian.
    Handles nested 'text_config' and 'vision_config'.
    """
    print("--- Writing Config Header ---")
    
    # 1. Prepare a flattened map for easier access
    cfg_map = {}
    cfg_map.update(config)
    cfg_map.update(config.get("text_config", {}))
    
    # Manually map vision keys
    vision_cfg = config.get("vision_config", {})
    cfg_map["vision_hidden_size"] = vision_cfg.get("hidden_size")
    cfg_map["vision_depth"] = vision_cfg.get("depth")
    cfg_map["vision_patch_size"] = vision_cfg.get("patch_size")
    cfg_map["vision_spatial_merge_size"] = vision_cfg.get("spatial_merge_size")
    cfg_map["vision_temporal_patch_size"] = vision_cfg.get("temporal_patch_size")
    cfg_map["vision_num_heads"] = vision_cfg.get("num_heads")
    cfg_map["vision_intermediate_size"] = vision_cfg.get("intermediate_size")
    cfg_map["out_hidden_size"] = vision_cfg.get("out_hidden_size") # <-- Crucial line
    # 'out_hidden_size' is used as-is from vision_config
    
    for key in KEYS:
        # Use .get() on the flattened map
        val = cfg_map.get(key)
        
        # NOTE: 'vocab_size' often needs to be taken from the root or text_config.
        # Ensure we check the core config first if it's missing in the flattened map.
        if val is None and key == "vocab_size":
            val = config.get(key)
        
        # Skip keys that are missing or don't fit the fixed binary header format
        if not isinstance(val, (int, float)):
            # Special handling for optional keys that might be None
            if key in ["num_key_value_heads", "vision_intermediate_size", "vision_num_heads"]:
                print(f"Skipping optional config key {key} (missing/None).")
                continue
            
            # For other critical keys, report and skip
            print(f"Skipping config key {key} (type {type(val)} or missing).")
            continue

        if isinstance(val, int):
            fout.write(struct.pack("<i", val))
            print(f"Wrote config key {key}={val} (<i)")
        elif isinstance(val, float):
            fout.write(struct.pack("<f", val))
            print(f"Wrote config key {key}={val} (<f)")


def write_weights_streaming(
    providers,
    ordered_keys,
    fout,
    out_dtype="float32",
):
    print("\n--- Writing Weights ---")

    for name in ordered_keys:
        tens = providers[name].load().cpu()

        if out_dtype == "float32":
            tens = tens.to(torch.float32)
            np_arr = tens.numpy().astype(np.float32, copy=False)

        elif out_dtype in ("float16", "bfloat16"):
            tens = tens.to(
                torch.float16 if out_dtype == "float16" else torch.bfloat16
            )
            # reinterpret 16-bit payload
            np_arr = tens.numpy().view(np.uint16)

        else:
            raise ValueError(out_dtype)

        fout.write(np_arr.tobytes(order="C"))
        print(f"Wrote {name} shape={tuple(tens.shape)} dtype={out_dtype}")

# ---- CLI ----


def parse_args():
    p = argparse.ArgumentParser(
        description=
        "Export Qwen-VL to single .bin (config + weights) in one step.")
    p.add_argument(
        "--input",
        required=True,
        help="Path to input .safetensors (containing standard weights)")
    p.add_argument("--config", required=True, help="Path to config.json")
    p.add_argument("--output", required=True, help="Output .bin path")
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Output weights dtype (default float32, or bfloat16 for raw bf16 payload)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config JSON
    with open(args.config, "r") as cf:
        config = json.load(cf)

    # Open safetensors once; build providers
    with safe_open(args.input, framework="pt", device="cpu") as f:
        providers = collect_effective_keys(f)
        all_keys = list(providers.keys())
        ordered = reorder_keys_for_write(all_keys)

        """
        for item in ordered:
            print(item)

        import sys
        sys.stdout.flush()
        assert 0 == 1
        """

        # Write binary in one pass
        with open(args.output, "wb") as fout:
            write_config_header(config, fout)
            write_weights_streaming(providers,
                                    ordered,
                                    fout,
                                    out_dtype=args.dtype)

    print(f"\nDone. Wrote {args.output}")


if __name__ == "__main__":
    main()