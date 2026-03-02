# Qwen3-VL-2B C++ Inference Engine (CPU-Optimized)

A lightweight C++ inference engine for Qwen3-VL-2B, built for efficient multimodal inference on low-power CPUs without relying on external deep learning frameworks.

Key Optimizations

🔧 Quantized & Packed GEMM
- Blocked weight prepacking for improved cache locality
- Vision module weight transpose for contiguous SIMD access
- Precomputed weight sums for efficient u8 × s8 asymmetric quantization correction
- INT32 accumulation with fused dequantization epilogue

⚡ Inference Pipeline Design
- Explicit separation of prefill and autoregressive decode
- KV-cache reuse across decoding steps
- Mixed precision execution:
+ Text transformer → INT8 (group=64 scaling)
+ Vision encoder → FP16 weights and activations

Quantization Strategy
- Text branch optimized for memory bandwidth via INT8 weights
- Vision branch preserved in FP16 for numerical stability

Designed with a focus on reducing memory movement, improving cache efficiency, and enabling practical LLM/VLM deployment on ultrabook-class CPUs.

You can see the full optimization at [CPU Branch](https://github.com/truongchu/qwen3-cpp/tree/int8-optimized)

# Incoming optimization
- KV cache quantization
- Pascal-GPU integration

# Installation (incoming)

Install

```
conda install -c conda-forge opencv gxx_linux-64 cmake pkg-config -y
pip install -r requirements.txt
```
