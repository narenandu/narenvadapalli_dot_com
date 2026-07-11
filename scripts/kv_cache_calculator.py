# KV Cache VRAM Footprint Calculator
# File: scripts/kv_cache_calculator.py
# Execution: Run in headless mode: `python scripts/kv_cache_calculator.py`

def calculate_kv_cache_size(layers, kv_heads, head_dim, seq_len, batch_size, precision_bytes=2):
    """
    Calculates the exact size of the KV Cache in gigabytes.
    Formula: 2 * layers * kv_heads * head_dim * seq_len * batch_size * precision_bytes
    """
    # 2 represents storing both Keys (K) and Values (V)
    total_elements = 2 * layers * kv_heads * head_dim * seq_len * batch_size
    total_bytes = total_elements * precision_bytes
    total_gb = total_bytes / (1024 ** 3) # Convert bytes to Gigabytes
    return total_gb

if __name__ == "__main__":
    # Model configuration examples:
    # 1. Llama 3 8B (uses Grouped-Query Attention - GQA)
    llama3_8b = {
        "name": "Llama 3 8B (GQA)",
        "layers": 32,
        "kv_heads": 8,       # GQA uses 8 KV heads (Q heads is 32)
        "head_dim": 128,     # 4096 hidden_dim / 32 Q heads
    }
    
    # 2. Llama 2 7B (uses Multi-Head Attention - MHA)
    llama2_7b = {
        "name": "Llama 2 7B (MHA)",
        "layers": 32,
        "kv_heads": 32,      # MHA uses equal KV heads to Q heads
        "head_dim": 128,     # 4096 hidden_dim / 32 Q heads
    }
    
    # Serving settings
    batch_sizes = [1, 8, 16, 32]
    context_lengths = [2048, 8192, 16384]
    
    print("=" * 75)
    print("KV CACHE VRAM CAPACITY SIMULATOR (FP16 / BF16 - 2 Bytes per parameter)")
    print("=" * 75)
    
    for model in [llama2_7b, llama3_8b]:
        print(f"\nModel: {model['name']}")
        print(f"Details: {model['layers']} Layers | {model['kv_heads']} KV Heads | {model['head_dim']} Head Dim")
        print("-" * 75)
        print(f"{'Batch Size':<12} | {'Context Length':<15} | {'KV Cache VRAM Footprint':<25}")
        print("-" * 75)
        
        for batch in batch_sizes:
            for ctx in context_lengths:
                size_gb = calculate_kv_cache_size(
                    layers=model["layers"],
                    kv_heads=model["kv_heads"],
                    head_dim=model["head_dim"],
                    seq_len=ctx,
                    batch_size=batch
                )
                print(f"{batch:<12} | {ctx:<15} | {size_gb:.3f} GB")
        print("=" * 75)
