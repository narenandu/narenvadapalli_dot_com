# Conceptual Simulator of LLM Inference Phases: Prefill vs. Decode
# File: scripts/prefill_decode_simulator.py
# Execution: Run in headless mode: `python scripts/prefill_decode_simulator.py`

import time

def simulate_inference_step(batch_size, seq_len, hidden_dim, num_layers, precision_bytes=2):
    """
    Simulates memory read overhead and math operations (FLOPs) for a transformer layer.
    """
    # Number of parameters in a single transformer layer (simplified projections)
    # 4 projection matrices for Self-Attention (Q, K, V, O) + 2 MLP matrices
    # Projection size: hidden_dim * hidden_dim
    # MLP size: 2 * (hidden_dim * 4 * hidden_dim)
    attn_params = 4 * (hidden_dim * hidden_dim)
    mlp_params = 8 * (hidden_dim * hidden_dim)
    layer_params = attn_params + mlp_params
    
    total_weights_bytes = layer_params * precision_bytes
    
    # 1. Prefill Phase
    # The entire prompt sequence (seq_len) is processed in parallel
    # Matrix multiplications: [Batch, SeqLen, HiddenDim] x [HiddenDim, HiddenDim]
    # FLOPS count for matrix multiply: 2 * N * M * K
    prefill_flops = num_layers * (2 * batch_size * seq_len * layer_params)
    
    # Weights are read from VRAM once per layer
    prefill_memory_read = total_weights_bytes * num_layers
    
    # Arithmetic Intensity = FLOPs / Bytes Read
    prefill_intensity = prefill_flops / prefill_memory_read
    
    # 2. Decode Phase (next single token generation)
    # Only 1 new token is processed autoregressively (seq_len = 1)
    # Matrix multiplications: [Batch, 1, HiddenDim] x [HiddenDim, HiddenDim]
    decode_flops = num_layers * (2 * batch_size * 1 * layer_params)
    
    # Weights must still be read from VRAM once per layer to process the single token
    decode_memory_read = total_weights_bytes * num_layers
    
    # Arithmetic Intensity for Decode
    decode_intensity = decode_flops / decode_memory_read
    
    return {
        "params_per_layer_m": layer_params / 1e6,
        "prefill": {
            "flops_gflops": prefill_flops / 1e9,
            "memory_gb": prefill_memory_read / 1e9,
            "intensity": prefill_intensity
        },
        "decode": {
            "flops_gflops": decode_flops / 1e9,
            "memory_gb": decode_memory_read / 1e9,
            "intensity": decode_intensity
        }
    }

if __name__ == "__main__":
    # Model Configurations (e.g., Llama-like ~8B parameters layer projection)
    BATCH_SIZE = 4
    PROMPT_LEN = 1024 # 1024 tokens prompt
    HIDDEN_DIM = 4096
    NUM_LAYERS = 32
    
    results = simulate_inference_step(BATCH_SIZE, PROMPT_LEN, HIDDEN_DIM, NUM_LAYERS)
    
    print("=" * 60)
    print(f"LLM LAYER SIMULATOR (Layer Params: {results['params_per_layer_m']:.2f} M)")
    print(f"Batch Size: {BATCH_SIZE} | Prompt Length: {PROMPT_LEN} | Hidden Dim: {HIDDEN_DIM}")
    print("=" * 60)
    
    print(f"1. PREFILL PHASE:")
    print(f"   - Total FLOPS required: {results['prefill']['flops_gflops']:.2f} GFLOPs")
    print(f"   - Weight memory read:   {results['prefill']['memory_gb']:.3f} GB")
    print(f"   - Arithmetic Intensity: {results['prefill']['intensity']:.2f} FLOPs/byte")
    print("   - Classification:       COMPUTE-BOUND (High ratio of math to memory)")
    print("-" * 60)
    
    print(f"2. DECODE PHASE (Single Token):")
    print(f"   - Total FLOPS required: {results['decode']['flops_gflops']:.2f} GFLOPs")
    print(f"   - Weight memory read:   {results['decode']['memory_gb']:.3f} GB")
    print(f"   - Arithmetic Intensity: {results['decode']['intensity']:.2f} FLOPs/byte")
    print("   - Classification:       MEMORY-BANDWIDTH-BOUND (Low ratio of math to memory)")
    print("=" * 60)
