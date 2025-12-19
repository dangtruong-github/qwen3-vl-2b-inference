# qwen3-vl-2b-inference

Here we use Qwen3-VL model family to infer. We'll start with qwen3-vl-2b

Install

```
conda install -c conda-forge opencv gxx_linux-64 cmake pkg-config -y
```

Baseline (no images):
- Average TTFT: 30.928463 (s)
- Average generated tokens per second: 0.613309 (toks/s)