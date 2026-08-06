"""
Scaled Dot-Product & Multi-Head Self-Attention Simulation from Scratch
Author: Narendra Vadapalli
Series: Neural Architecture Evolution Series (Part 3)

This script demonstrates the core mathematical mechanics of the Transformer:
1. Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
2. Multi-Head Self-Attention: Splitting channels into parallel attention heads
3. GPU Parallelization vs. Sequential RNN Loops: Demonstrating O(1) parallel token processing vs O(N) sequential loops
"""

import math
import random
import time

def matrix_multiply(A, B):
    """Computes standard matrix multiplication C = A x B."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"Matrix dimension mismatch: {cols_A} != {rows_B}"

    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            a_ik = A[i][k]
            for j in range(cols_B):
                C[i][j] += a_ik * B[k][j]
    return C

def transpose(M):
    """Computes transpose of matrix M."""
    rows, cols = len(M), len(M[0])
    return [[M[r][c] for r in range(rows)] for c in range(cols)]

def softmax_rowwise(matrix):
    """Applies numerically stable softmax row-by-row."""
    result = []
    for row in matrix:
        max_val = max(row)
        exps = [math.exp(val - max_val) for val in row]
        sum_exps = sum(exps)
        result.append([e / sum_exps for e in exps])
    return result

class ScaledDotProductAttention:
    """Self-Attention module executing Q, K, V matrix math."""
    def __init__(self, d_model: int, d_k: int):
        self.d_model = d_model
        self.d_k = d_k
        self.scale = math.sqrt(d_k)

        random.seed(42)
        # Random projections W_Q, W_K, W_V
        self.W_Q = [[random.uniform(-0.1, 0.1) for _ in range(d_k)] for _ in range(d_model)]
        self.W_K = [[random.uniform(-0.1, 0.1) for _ in range(d_k)] for _ in range(d_model)]
        self.W_V = [[random.uniform(-0.1, 0.1) for _ in range(d_k)] for _ in range(d_model)]

    def forward(self, X):
        """
        X shape: (N_seq, d_model)
        Computes Q = X * W_Q, K = X * W_K, V = X * W_V in PARALLEL.
        """
        # 1. Project input tokens into Q, K, V matrices simultaneously
        Q = matrix_multiply(X, self.W_Q)  # (N, d_k)
        K = matrix_multiply(X, self.W_K)  # (N, d_k)
        V = matrix_multiply(X, self.W_V)  # (N, d_k)

        # 2. Compute raw attention scores: S = Q * K^T / sqrt(d_k)
        K_T = transpose(K)
        scores_raw = matrix_multiply(Q, K_T)  # (N, N)

        scores_scaled = [[val / self.scale for val in row] for row in scores_raw]

        # 3. Softmax row-wise to get attention weights A
        A = softmax_rowwise(scores_scaled)  # (N, N)

        # 4. Multiply attention weights by Value matrix: Output = A * V
        Output = matrix_multiply(A, V)  # (N, d_k)

        return Output, A

def run_simulation():
    seq_len = 8
    d_model = 16
    d_k = 16

    print("=" * 75)
    print("      TRANSFORMER SELF-ATTENTION & GPU PARALLELIZATION SIMULATION      ")
    print("=" * 75)
    print(f"Sequence Length (N) : {seq_len} tokens")
    print(f"Embedding Dim (d)   : {d_model}")
    print(f"Key/Query Dim (d_k) : {d_k}")
    print("-" * 75)

    random.seed(123)
    # Input sequence X: N tokens x d_model features
    X = [[random.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)]

    # Run Scaled Dot-Product Self-Attention
    attention_layer = ScaledDotProductAttention(d_model, d_k)
    output, A = attention_layer.forward(X)

    print("\n[1] Sample Attention Matrix A = Softmax(Q * K^T / sqrt(d_k)):")
    print("    (Row i represents how much Token i attends to Token j)")
    for i in range(min(4, seq_len)):
        row_str = " ".join(f"{A[i][j]:.3f}" for j in range(min(4, seq_len)))
        print(f"    Token {i} weights -> [{row_str} ...]")

    print("\n[2] Output Representation Matrix (N x d_k):")
    for i in range(min(3, seq_len)):
        out_str = " ".join(f"{output[i][j]:.3f}" for j in range(4))
        print(f"    Token {i} output -> [{out_str} ...]")

    print("\n[3] Parallelization Benchmark (N=500 Tokens):")
    large_N = 100
    large_X = [[random.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(large_N)]

    # Simulated Sequential RNN step loop
    t0 = time.perf_counter()
    h = [0.0] * d_model
    for t in range(large_N):
        # Sequential step dependency: h_t depends on h_{t-1}
        h = [math.tanh(h[d] + large_X[t][d]) for d in range(d_model)]
    rnn_time = (time.perf_counter() - t0) * 1000

    # Parallel Transformer Attention matrix multiplication
    t0 = time.perf_counter()
    large_attn = ScaledDotProductAttention(d_model, d_k)
    _, _ = large_attn.forward(large_X)
    transformer_time = (time.perf_counter() - t0) * 1000

    print(f"    - Simulated Sequential RNN (100 steps sequential loop): {rnn_time:.2f} ms")
    print(f"    - Parallel Transformer Matrix Ops (100 tokens batch op): {transformer_time:.2f} ms")
    print("    --> Key Insight: Transformer computes all pairwise token interactions simultaneously")
    print("        in a single matrix multiplication without step-by-step loops!")
    print("=" * 75)

if __name__ == "__main__":
    run_simulation()
