---
title: "Understanding the KV Cache: The VRAM Bottleneck of LLM Serving"
date: 2026-07-11
template: blog
image: "./cover_image.jpg"
description: "Why do LLM serving nodes run out of VRAM? Explore the mechanics, calculations, and capacity bottlenecks of the Key-Value (KV) Cache."
tags: ["ai", "machine-learning", "performance", "infrastructure"]
---

*AI Inference Deep-Dive Series: &larr; [The Two Pillars of LLM Inference: Prefill vs. Decode](/blog/prefill-vs-decode/) (Previous)*

### Prior Reading Material
Before exploring the memory footprint of serving systems, ensure you understand the execution phases of transformer networks:
*   [The Two Pillars of LLM Inference: Prefill vs. Decode](/blog/prefill-vs-decode/) — Deep-dive into compute-bound prefill vs. memory-bandwidth-bound decode phases, and the theoretical limits of Arithmetic Intensity.
*   [Basics of AI Inference: Demystifying Latency, Throughput, and Serving](/blog/basics-of-ai-inference/) — Tracing the core performance metrics (TTFT, throughput, inter-token latency) and introductory OS-level optimizations.

---

When deploying Large Language Models (LLMs) to production, systems engineers often encounter a puzzling phenomenon: 

You select a GPU with enough VRAM to hold the model parameters (e.g., an 8B model fits comfortably in 16 GB of memory at 16-bit precision). But the moment you start sending concurrent user requests, the server crashes with a dreaded **CUDA Out of Memory (OOM)** error. 

The culprit is almost never the static model parameters. Instead, the VRAM bottleneck is caused by the **Key-Value (KV) Cache**—the dynamic memory allocated to store historical context during text generation.

In this post, we explore why the KV Cache is mathematically necessary, how to calculate its size, and why it forms the primary bottleneck to scaling LLM concurrency.

---

### The Problem: Quadratic Attention Recomputation

As we discussed in [The Two Pillars of LLM Inference](/blog/prefill-vs-decode/), generating text with an LLM is an autoregressive process. To predict token $N$, the model must inspect all previous tokens ($1$ to $N-1$) to compute the attention weights.

The self-attention mechanism works by computing Query (Q), Key (K), and Value (V) projections of the tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In the decode phase, when we generate a new token, the model needs the Key and Value vectors of **all past tokens** to compute attention for the new token. 

If we did not cache these past states, the GPU would have to recompute the Keys and Values for every single historical token, at every single decode step. Generating a 1,000-token output would require running the projection layers $1,000$ times for token 1, $999$ times for token 2, and so on. This turns the generation process into a quadratic $O(N^2)$ compute loop, causing latency to skyrocket.

---

### The Solution: The KV Cache

To avoid this recomputation, we trade memory space for speed: we allocate a dedicated block of VRAM to store the Key and Value vectors of past tokens.

*   **Prefill Phase**: When the user's prompt is processed, we calculate the Keys and Values for all prompt tokens in parallel and save them to the KV Cache.
*   **Decode Phase**: When generating token $N$, we only calculate Q, K, and V for the *new* single input token. We append the new K and V to the cache, and then load the entire historical KV Cache from VRAM to compute the attention matrix.

By caching these states, we keep the compute cost of each decode step linear $O(1)$ relative to the new token, enabling fast token streaming.

---

### Calculating the KV Cache Size

How much VRAM does the KV Cache actually consume? The memory footprint scales linearly with the sequence length, batch size, and model size. 

The formula to calculate the size in bytes of the KV Cache is:

$$\text{KV Cache Size (bytes)} = 2 \times \text{Layers} \times \text{KV Heads} \times \text{HeadDim} \times \text{SeqLen} \times \text{BatchSize} \times \text{PrecisionBytes}$$

Where:
*   **2**: Storing both Keys (K) and Values (V).
*   **Layers**: The number of transformer layers in the model.
*   **KV Heads**: The number of Key-Value attention heads.
*   **HeadDim**: The size of each attention head (typically `Hidden Dimension / Query Heads`).
*   **SeqLen**: The total context sequence length (prompt length + generated tokens).
*   **BatchSize**: The number of concurrent requests being processed.
*   **PrecisionBytes**: The bytes per parameter (2 bytes for FP16/BF16, 1 byte for FP8).

---

### The Architectural Shift: MHA vs. GQA

In older model architectures like Llama 2 or GPT-3, the number of KV Heads was equal to the number of Query Heads. This is called **Multi-Head Attention (MHA)**. 

To reduce the massive memory overhead of MHA, newer models (like Llama 3, Mistral, and Qwen) utilize **Grouped-Query Attention (GQA)**. GQA groups multiple query heads to share a single Key-Value head, reducing the KV head count (typically by a factor of 4 or 8) and directly shrinking the KV Cache size by that same factor.

We can see the impact of this architectural shift by running a simulator script. 

The simulation script is saved at `scripts/kv_cache_calculator.py`. You can run it on your local system:

```bash
# Run the KV cache memory calculator
python scripts/kv_cache_calculator.py
```

Let's compare a Llama 2 7B model (MHA) against a Llama 3 8B model (GQA) serving FP16 precision:

**Simulation Results:**
```text
Model: Llama 2 7B (MHA)
Details: 32 Layers | 32 KV Heads | 128 Head Dim
---------------------------------------------------------------------------
Batch Size   | Context Length  | KV Cache VRAM Footprint  
---------------------------------------------------------------------------
1            | 2048            | 1.000 GB
16           | 8192            | 64.000 GB
32           | 16384           | 256.000 GB

Model: Llama 3 8B (GQA)
Details: 32 Layers | 8 KV Heads | 128 Head Dim
---------------------------------------------------------------------------
Batch Size   | Context Length  | KV Cache VRAM Footprint  
---------------------------------------------------------------------------
1            | 2048            | 0.250 GB
16           | 8192            | 16.000 GB
32           | 16384           | 64.000 GB
```

#### The Takeaway
Under MHA (Llama 2), serving a batch size of 16 at an 8k context length requires **64 GB of VRAM just for the KV Cache**. This exceeds the total VRAM of an A100 (40GB) or H100 (80GB) once you add the model parameters (which consume ~14 GB).

Under GQA (Llama 3), the KV Cache footprint for the exact same workload drops to **16 GB**, enabling much higher throughput on standard hardware.

---

### The Scaling Bottleneck: Throughput vs. Capacity

The KV Cache creates a fundamental systems trade-off between **throughput** and **hardware capacity**:

1.  **To Maximize Throughput**: You want high batch sizes. Processing multiple requests concurrently allows the GPU to reuse weights across requests, bypassing the memory-bandwidth decode bottleneck.
2.  **The Memory Limit**: High batch sizes and long sequence lengths cause the KV Cache to expand exponentially. If the KV cache exceeds the remaining GPU memory, the engine crashes with a CUDA OOM.

This constraint forces serving systems to balance batch size limits dynamically.

---

### Overcoming the Bottleneck

Because the KV Cache is the primary bottleneck to scaling concurrency, the industry has developed several advanced software optimizations:

1.  **PagedAttention**: Dynamically maps KV cache tokens to virtual memory pages (used in vLLM), eliminating VRAM fragmentation and allowing servers to pack more requests into memory.
2.  **Quantization**: Compressing the KV cache to FP8 or INT4 precision, halving memory consumption at the cost of slight degradation in model outputs.
3.  **RadixAttention**: Caching and sharing prompt prefix KV states across requests (used in SGLang) to eliminate redundant memory allocations.

---

### Summary

The KV Cache is a critical performance engine that trades GPU memory space to keep generation times linear. However, as developers build longer context windows (32k, 128k, or 1M tokens), managing the KV Cache VRAM footprint becomes the single most important task in AI systems engineering.

---

### What's Next?

Now that we understand the math and limitations of the KV Cache, our next post will explore **The Landscape of Open Source vs. Enterprise Inference Engines** (rescheduled to July 12th). We will examine how engines like vLLM and SGLang implement PagedAttention and RadixAttention to tame the KV cache memory beast!
