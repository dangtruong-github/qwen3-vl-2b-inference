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
    "lm_head.weight",         # Unembedding / final linear layer
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


def reorder_keys_for_write(all_keys: List[str]) -> List[str]:
    """Return keys ordered by CATEGORIES suffix + block index, then the rest sorted."""
    buckets = {cat: [] for cat in CATEGORIES}
    others: List[str] = []

    for key in all_keys:
        matched = False
        # Find the category based on the suffix
        for cat in CATEGORIES:
            if key.endswith(cat):
                # Check for block index using the Qwen-specific regex
                m = _BLOCK_RE.search(key)
                block_idx = int(m.group(1)) if m else -1
                buckets[cat].append((block_idx, key))
                matched = True
                break
        if not matched:
            others.append(key)

    ordered: List[str] = []
    # Write by category, sorted by block index (e.g., ln_1.weight block 0, block 1, ...)
    for cat in CATEGORIES:
        for _, k in sorted(buckets[cat], key=lambda x: x[0]):
            ordered.append(k)

    # Add global/non-block keys (sorted)
    ordered.extend(sorted(others))
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
    providers: Dict[str, TensorProvider],
    ordered_keys: List[str],
    fout,
    out_dtype: str = "float32",
) -> None:
    """
    Stream tensors to file in the specified order.
    out_dtype: "float32" (default) or "bfloat16"
    """
    torch_cast = torch.float32 if out_dtype == "float32" else torch.bfloat16

    print("\n--- Writing Weights ---")
    for name in ordered_keys:
        # Load and cast on demand
        tens = providers[name].load().to(torch_cast).cpu()
        
        # Convert to numpy and write bytes
        if torch_cast == torch.bfloat16:
            # For bfloat16, view as uint16 to get the raw 16-bit payload
            np_arr = tens.view(torch.uint16).numpy().astype(np.uint16, copy=False)
        else:
            # For float32
            np_arr = tens.numpy().astype(np.float32, copy=False)

        fout.write(np_arr.tobytes(order="C"))
        print(f"Wrote {name}  shape={tuple(tens.shape)} dtype={torch_cast}")


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
        choices=["float32", "bfloat16"],
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