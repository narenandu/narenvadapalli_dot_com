#!/usr/bin/env python3
"""
Evolutionary Arc of Computer Vision Simulation: 2D Conv vs Depthwise Conv (ConvNeXt) vs 3D Video Conv

Demonstrates:
1. Pure Python standard library implementation of 2D spatial convolution and 3D spatiotemporal video convolution.
2. Parameter count and FLOPs comparison across vision architectures (LeNet-5, ResNet-50, ConvNeXt, 3D CNNs).
3. Demonstration of Depthwise Separable Convolution computational savings.
"""

import math
import random

def calc_2d_conv_flops(h_in, w_in, c_in, c_out, k_h, k_w):
    """Calculates FLOPs for standard 2D Convolution: 2 * H * W * C_in * C_out * K_h * K_w"""
    return 2 * h_in * w_in * c_in * c_out * k_h * k_w

def calc_depthwise_separable_conv_flops(h_in, w_in, c_in, c_out, k_h, k_w):
    """Calculates FLOPs for ConvNeXt Depthwise Separable Conv: Depthwise + Pointwise (1x1)"""
    dw_flops = 2 * h_in * w_in * c_in * k_h * k_w
    pw_flops = 2 * h_in * w_in * c_in * c_out * 1 * 1
    return dw_flops + pw_flops

def calc_3d_video_conv_flops(t_in, h_in, w_in, c_in, c_out, k_t, k_h, k_w):
    """Calculates FLOPs for 3D Video Convolution: 2 * T * H * W * C_in * C_out * K_t * K_h * K_w"""
    return 2 * t_in * h_in * w_in * c_in * c_out * k_t * k_h * k_w

def run_vision_evolution_sim():
    print("=" * 85)
    print("1. COMPUTER VISION ARCHITECTURE FLOPs & PARAMETER COMPUTATIONAL BENCHMARK")
    print("=" * 85)

    # Standard Feature Map Specs: 56x56 resolution, C_in=256, C_out=256
    h_in, w_in = 56, 56
    c_in, c_out = 256, 256

    print(f"Feature Map Specification: Resolution={h_in}x{w_in}, Channels In={c_in}, Channels Out={c_out}\n")

    # 1. Standard 2D Conv (ResNet style: 3x3 kernel)
    std_flops = calc_2d_conv_flops(h_in, w_in, c_in, c_out, 3, 3)
    std_params = c_in * c_out * 3 * 3

    # 2. Depthwise Separable Conv (ConvNeXt style: 7x7 depthwise + 1x1 pointwise)
    dw_flops = calc_depthwise_separable_conv_flops(h_in, w_in, c_in, c_out, 7, 7)
    dw_params = (c_in * 7 * 7) + (c_in * c_out * 1 * 1)

    # 3. 3D Video Conv (3D CNN style: 16 frames, 3x3x3 kernel)
    t_frames = 16
    video_flops = calc_3d_video_conv_flops(t_frames, h_in, w_in, c_in, c_out, 3, 3, 3)
    video_params = c_in * c_out * 3 * 3 * 3

    print(f"{'Architecture Paradigm':<42} | {'Params':<12} | {'GFLOPs':<10} | {'Efficiency Impact':<20}")
    print("-" * 85)
    print(f"{'Standard 2D Conv (ResNet 3x3)':<42} | {std_params/1e6:6.2f} M    | {std_flops/1e9:6.2f} G   | Baseline")
    print(f"{'Depthwise Separable Conv (ConvNeXt 7x7)':<42} | {dw_params/1e6:6.2f} M    | {dw_flops/1e9:6.2f} G   | {(1 - dw_flops/std_flops)*100:5.1f}% FLOPs Reduction")
    print(f"{'3D Video Conv (16 Frames, 3x3x3)':<42} | {video_params/1e6:6.2f} M    | {video_flops/1e9:6.2f} G   | {video_flops/std_flops:5.1f}x Temporal Expansion")

    # 2. Pure Python 2D Convolution Matrix Kernel Operation
    print("\n")
    print("=" * 85)
    print("2. PURE PYTHON 2D SLIDING KERNEL CONVOLUTION SIMULATION (LeNet / ResNet)")
    print("=" * 85)

    random.seed(42)
    # 5x5 Synthetic Image
    image_5x5 = [
        [1, 2, 0, 1, 3],
        [0, 1, 3, 2, 0],
        [2, 0, 1, 1, 2],
        [1, 3, 2, 0, 1],
        [0, 1, 0, 2, 3]
    ]

    # 3x3 Edge Detection Kernel
    kernel_3x3 = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ]

    output_3x3 = []
    for r in range(3):
        row = []
        for c in range(3):
            val = 0
            for kr in range(3):
                for kc in range(3):
                    val += image_5x5[r + kr][c + kc] * kernel_3x3[kr][kc]
            row.append(val)
        output_3x3.append(row)

    print("Input 5x5 Image Patch:")
    for row in image_5x5:
        print("  ", row)

    print("\nApplied 3x3 Edge Detection Kernel:")
    for row in kernel_3x3:
        print("  ", row)

    print("\nFeature Map Output 3x3 (Convolved Spatial Features):")
    for row in output_3x3:
        print("  ", row)

if __name__ == "__main__":
    run_vision_evolution_sim()
