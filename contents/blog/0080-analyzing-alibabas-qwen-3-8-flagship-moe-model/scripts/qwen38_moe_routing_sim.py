#!/usr/bin/env python3
"""
Qwen 3.8 Sparse MoE Routing & API Cost Economics Simulator

Demonstrates:
1. Pure Python standard library implementation of Top-K MoE Expert Gating Router (Top-8 out of 512 Experts).
2. Active vs. Total Parameter scaling calculation (32B Active out of 512B Total).
3. API Token Cost Comparison across Frontier Models (Qwen 3.8, DeepSeek-V3, GPT-4o, Claude 3.5 Sonnet).
"""

import math
import random

def softmax(logits):
    """Computes numerical softmax over a list of floating point values."""
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

def simulate_qwen38_moe_router(token_dim=16, num_experts=512, top_k=8):
    """
    Simulates Top-K Sparse MoE Routing for a single incoming token embedding.
    Gating equation: TopK(Softmax(W_gates * h_t))
    """
    random.seed(42)
    # Dummy token embedding (h_t)
    h_t = [random.gauss(0, 1) for _ in range(token_dim)]

    # Router Gate Weights W_gates (num_experts x token_dim)
    W_gates = [[random.gauss(0, 0.1) for _ in range(token_dim)] for _ in range(num_experts)]

    # Compute raw logits for 512 experts
    logits = []
    for e in range(num_experts):
        logit = sum(h_t[i] * W_gates[e][i] for i in range(token_dim))
        logits.append(logit)

    # Compute Softmax probabilities across all 512 experts
    probs = softmax(logits)

    # Select Top-K Experts
    expert_probs = list(enumerate(probs))
    expert_probs.sort(key=lambda x: x[1], reverse=True)
    top_experts = expert_probs[:top_k]

    # Re-normalize Top-K routing weights
    top_weight_sum = sum(w for _, w in top_experts)
    normalized_top = [(idx, w / top_weight_sum) for idx, w in top_experts]

    return normalized_top

def calculate_api_cost(input_tokens_m, output_tokens_m, input_price_per_m, output_price_per_m):
    """Calculates total workload cost given token volume in Millions."""
    return (input_tokens_m * input_price_per_m) + (output_tokens_m * output_price_per_m)

def run_qwen38_sim():
    print("=" * 85)
    print("1. QWEN 3.8 SPARSE MIXTURE-OF-EXPERTS (MoE) ROUTING SIMULATION")
    print("=" * 85)

    num_experts = 512
    top_k = 8
    total_params = 512  # Billion
    active_params = 32  # Billion

    print(f"Model Specs          : Total Parameters = {total_params}B | Active Params per Token = {active_params}B")
    print(f"Expert Topology      : {num_experts} Total Experts | Top-{top_k} Activated per Token ({top_k/num_experts*100:.1f}% Sparsity)\n")

    top_experts = simulate_qwen38_moe_router(token_dim=16, num_experts=num_experts, top_k=top_k)

    print(f"{'Activated Expert ID':<22} | {'Raw Prob':<12} | {'Normalized Routing Weight':<25}")
    print("-" * 85)
    for idx, norm_w in top_experts:
        raw_p = norm_w * sum(w for _, w in top_experts)
        print(f"Expert #{idx:<15} | {raw_p:10.5f}  | {norm_w:25.4f}")

    print("\n")
    print("=" * 85)
    print("2. FRONTIER MODEL TOKEN PRICING & ECONOMICS BENCHMARK (Per 1M Tokens)")
    print("=" * 85)

    models = {
        "Alibaba Qwen 3.8-Max": {"in": 0.30, "out": 0.90, "active": "32B MoE"},
        "DeepSeek-V3": {"in": 0.27, "out": 1.10, "active": "37B MoE"},
        "OpenAI GPT-4o": {"in": 2.50, "out": 10.00, "active": "Dense 200B+"},
        "Anthropic Claude 3.5 Sonnet": {"in": 3.00, "out": 15.00, "active": "Dense"}
    }

    print(f"{'Model Name':<30} | {'Input ($/1M)':<14} | {'Output ($/1M)':<14} | {'Active Compute':<15}")
    print("-" * 85)
    for name, info in models.items():
        print(f"{name:<30} | ${info['in']:<13.2f} | ${info['out']:<13.2f} | {info['active']:<15}")

    # Simulated Enterprise Workload: 100 Million Input Tokens + 20 Million Output Tokens
    in_m = 100.0
    out_m = 20.0

    print("\n")
    print("=" * 85)
    print(f"3. ENTERPRISE WORKLOAD COST (100M Input Tokens + 20M Output Tokens)")
    print("=" * 85)

    qwen_cost = calculate_api_cost(in_m, out_m, models["Alibaba Qwen 3.8-Max"]["in"], models["Alibaba Qwen 3.8-Max"]["out"])

    for name, info in models.items():
        cost = calculate_api_cost(in_m, out_m, info["in"], info["out"])
        multiplier = cost / qwen_cost
        print(f"• {name:<30}: ${cost:9.2f} ({multiplier:4.1f}x relative to Qwen 3.8)")

if __name__ == "__main__":
    run_qwen38_sim()
