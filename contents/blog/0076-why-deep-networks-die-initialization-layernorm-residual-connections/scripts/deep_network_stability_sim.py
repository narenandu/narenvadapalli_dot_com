"""
Deep Network Numerical Stability: Initialization, LayerNorm, and Residual Connections
Author: Narendra Vadapalli
Series: Neural Architecture Evolution Series (Part 6)

This script demonstrates:
1. Vanishing and Exploding Gradient collapse in a 30-layer un-initialized network.
2. Variance preservation via Kaiming (He) and Xavier (Glorot) Initialization.
3. Layer Normalization (LayerNorm) signal stabilization.
4. Residual Skip Connections (ResNet y = F(x) + x) gradient flow across 30 layers.
"""

import math
import random

def relu(x):
    return max(0.0, x)

def relu_grad(x):
    return 1.0 if x > 0.0 else 0.0

def mean(vals):
    return sum(vals) / len(vals)

def std_dev(vals):
    m = mean(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(var + 1e-8)


# =====================================================================
# 1. UNINITIALIZED VS HE-INITIALIZED LAYER PASS
# =====================================================================

class LinearLayer:
    def __init__(self, in_features, out_features, init_mode='he'):
        self.in_features = in_features
        self.out_features = out_features
        random.seed(42)

        if init_mode == 'naive_large': # Exploding gradients
            scale = 2.5
        elif init_mode == 'naive_small': # Vanishing gradients
            scale = 0.05
        elif init_mode == 'xavier': # Xavier/Glorot: sqrt(2 / (in + out))
            scale = math.sqrt(2.0 / (in_features + out_features))
        elif init_mode == 'he': # Kaiming/He: sqrt(2 / in_features)
            scale = math.sqrt(2.0 / in_features)
        else:
            scale = 1.0

        self.W = [[random.gauss(0, scale) for _ in range(in_features)] for _ in range(out_features)]
        self.b = [0.0] * out_features

    def forward(self, x):
        out = []
        for i in range(self.out_features):
            val = sum(self.W[i][j] * x[j] for j in range(self.in_features)) + self.b[i]
            out.append(val)
        return out


# =====================================================================
# 2. LAYER NORMALIZATION (LAYERNORM)
# =====================================================================

def layer_norm(x, eps=1e-5):
    m = mean(x)
    std = std_dev(x)
    return [(val - m) / (std + eps) for val in x]


# =====================================================================
# 3. 30-LAYER NETWORK SIMULATION
# =====================================================================

def simulate_deep_network(depth=30, dim=64, mode='he', use_norm=False, use_residual=False):
    random.seed(42)
    layers = [LinearLayer(dim, dim, init_mode=mode) for _ in range(depth)]
    
    # Input vector with mean ~0, std ~1
    x = [random.gauss(0, 1.0) for _ in range(dim)]
    
    # Forward Pass tracking activation variance
    activations = [x]
    curr = x
    for l in range(depth):
        out = layers[l].forward(curr)
        out_act = [relu(v) for v in out]

        if use_norm:
            out_act = layer_norm(out_act)

        if use_residual and len(curr) == len(out_act):
            out_act = [a + c for a, c in zip(out_act, curr)] # Residual skip connection y = F(x) + x

        curr = out_act
        activations.append(curr)

    # Backward Gradient Pass tracking gradient magnitude from Layer 30 back to Layer 1
    grad = [1.0] * dim # Incoming gradient from loss
    layer_grads = [std_dev(grad)]
    
    for l in reversed(range(depth)):
        # Backprop through ReLU and Weight matrix transpose
        prev_act = activations[l]
        next_grad = [0.0] * dim
        
        for i in range(dim):
            d_relu = relu_grad(activations[l+1][i])
            g_i = grad[i] * d_relu
            for j in range(dim):
                next_grad[j] += layers[l].W[i][j] * g_i
        
        if use_residual:
            next_grad = [ng + g for ng, g in zip(next_grad, grad)] # Shortcut gradient flow

        grad = next_grad
        layer_grads.append(std_dev(grad))

    layer_grads.reverse()
    return std_dev(activations[0]), std_dev(activations[-1]), layer_grads[0], layer_grads[-1]


def run_simulation():
    print("=" * 80)
    print("        SIMULATING DEEP NETWORK STABILITY ACROSS 30 LAYERS        ")
    print("=" * 80)

    configs = [
        ("1. Naive Small Init (W ~ N(0, 0.05))", "naive_small", False, False),
        ("2. Naive Large Init (W ~ N(0, 2.5))", "naive_large", False, False),
        ("3. Kaiming (He) Init (W ~ N(0, sqrt(2/N)))", "he", False, False),
        ("4. He Init + LayerNorm", "he", True, False),
        ("5. He Init + LayerNorm + Residual Skips (ResNet)", "he", True, True),
    ]

    for title, mode, norm, res in configs:
        print(f"\n{title}:")
        in_std, out_std, g_in, g_out = simulate_deep_network(depth=30, dim=64, mode=mode, use_norm=norm, use_residual=res)
        
        print(f"    Input Activation StdDev  : {in_std:.4f}")
        print(f"    Layer 30 Activation StdDev: {out_std:.4f}")
        print(f"    Layer 30 Gradient Magnitude: {g_in:.4e}")
        print(f"    Layer 1 Gradient Magnitude : {g_out:.4e}")
        
        if math.isnan(g_out) or g_out > 1e6:
            status = "FAILED: Exploding Gradient (NaN / Infinity)"
        elif g_out < 1e-10:
            status = "FAILED: Vanishing Gradient (Signal Died to 0.0000)"
        else:
            status = "PASSED: Stable Gradient Signal Flow Across All 30 Layers!"
        print(f"    STATUS: {status}")

    print("\n" + "=" * 80)
    print("  RESULT: Kaiming Init + LayerNorm + Residual Connections conquered network death!")
    print("=" * 80)

if __name__ == "__main__":
    run_simulation()
