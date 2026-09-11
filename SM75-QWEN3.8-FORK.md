# SM75 / Qwen3.8-27B vLLM fork

This fork carries the vLLM changes used by
[Aiakos1818/qwen3-8-27b-dual-2080ti-vllm](https://github.com/Aiakos1818/qwen3-8-27b-dual-2080ti-vllm)
to serve Qwen3.8-27B on 2 × RTX 2080 Ti 22GB (Turing / SM75, NVLink, TP=2).

- **Base**: upstream vLLM tag `v0.27.1` = `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`.
- **Branch**: `sm75-qwen3.8-kv`.
- **Commits** (on top of the base):
  1. `sm75/qwen3.8` — SM75 adaptation: FlashQLA legacy GDN prefill, Qwen3.5 MTP
     compatibility, reasoning budget, FlashInfer/sampler compatibility and GPU
     runner adjustments. Originally authored in the
     [zyYuc](https://github.com/zyYuc/qwen3-8-27b-dual-2080ti-vllm) project.
  2. `kv` — long-context multi-session agent KV optimization: session keep-alive
     pin, durable Mamba/GDN anchors, GPU↔RAM/SSD two-tier offload with chunked
     streaming, and Prometheus metrics.

Patch exports, design docs and launch scripts live in the deployment repo above
(`patches/`, `docs/kv-optimization/`, `scripts/`).

## License

Upstream vLLM is Apache-2.0; this fork keeps that license. The `sm75/qwen3.8`
base changes originate from the zyYuc deployment project (MIT) — see that repo
for attribution.
