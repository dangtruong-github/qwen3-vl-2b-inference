# Qwen3-VL-2B C++ Inference Engine

A lightweight C++ inference engine for Qwen3-VL-2B, designed for efficient multimodal inference on resource-constrained systems without relying on external deep learning frameworks.

Built with a focus on minimizing memory movement, improving cache locality, and enabling practical LLM/VLM deployment on ultrabook-class CPUs.

## Key Optimizations

### 🔧 Quantized & Packed GEMM

* Blocked weight prepacking for improved cache locality
* Precomputed weight sums for efficient asymmetric quantization correction
* INT32 accumulation with fused dequantization epilogue

### ⚡ Inference Pipeline

* Separate prefill and autoregressive decode
* KV-cache reuse across decoding steps
* Current KV-cache limit: 3072 tokens
* Mixed precision execution:
  * Text transformer → INT8 weights
  * Vision encoder → FP16 weights and activations

### 🧠 Memory & Attention Optimizations

* FP16 KV-cache for reduced memory bandwidth and footprint
* Fused kernel skeletons for reducing intermediate memory traffic
* OpenMP parallel execution for CPU scalability

## Design Goals

The project is designed around:

* Low memory bandwidth usage
* Cache-efficient execution
* Lightweight dependency footprint
* Practical multimodal inference on CPUs

## Runtime Configuration

| Component | Configuration |
|---|---|
| Maximum context length | 3072 tokens |
| KV-cache precision | FP16 |
| Text transformer weights | INT8 |
| Vision encoder weights | FP16 |
| Parallel backend | OpenMP |

## Memory Usage

| Configuration | Approx. Memory Usage |
|---|---|
| Model weights only | ~X GB |
| + 3072-token KV-cache | ~X GB |
| Peak runtime memory | ~X GB |

---

# Incoming Optimizations

* CPU Flash-style attention for long-context inference (vendor-specific)
* Vendor-specific optimized backends (ARM / x86 / CUDA)

## Backend Status

| Backend | Status | Branch |
|---|---|---|
| x86 | ✅ Completed | [x86 Branch](https://github.com/dangtruong-github/qwen3-vl-2b-inference/tree/cpu) |
| ARM | 🚧 In Progress | [ARM Branch](https://github.com/dangtruong-github/qwen3-vl-2b-inference/tree/cpu-arm) |
| CUDA | 🚧 In Progress | [CUDA Branch](https://github.com/dangtruong-github/qwen3-vl-2b-inference/tree/gpu) |

---

# Installation & Usage (Incoming)

## Install

```bash
conda install -c conda-forge opencv gxx_linux-64 cmake pkg-config -y
pip install -r requirements.txt

```
