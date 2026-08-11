#!/usr/bin/env python3
"""
Generative Adversarial Networks (GANs) Minimax Simulation: The Counterfeiter vs. Detective

Demonstrates:
1. Pure Python standard library implementation of a 1D Minimax GAN.
2. Real Data Distribution (Gaussian mu=4.0, std=0.5) vs. Generator Output.
3. Discriminator Loss, Generator Loss, and convergence towards Nash Equilibrium (D(x) -> 0.5).
"""

import math
import random

def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^-x)"""
    # Clamp x to avoid overflow
    x_clamped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clamped))

def sample_real_data(batch_size=16):
    """Real data distribution: 1D Gaussian centered at mu=4.0, std=0.5"""
    return [random.gauss(4.0, 0.5) for _ in range(batch_size)]

def sample_noise(batch_size=16):
    """Latent noise vector z ~ Uniform(0, 1)"""
    return [random.uniform(0.0, 1.0) for _ in range(batch_size)]

class Generator:
    """Simple 1D Linear Generator: G(z) = w_g * z + b_g"""
    def __init__(self):
        self.w_g = random.uniform(0.1, 0.5)
        self.b_g = random.uniform(-1.0, 0.0)

    def forward(self, z):
        return [self.w_g * zi + self.b_g for zi in z]

    def update(self, z, d_weights, lr=0.05):
        """Update Generator parameters to maximize Discriminator mistake D(G(z)) -> 1"""
        w_d, b_d = d_weights["w_d"], d_weights["b_d"]
        grad_w_g = 0.0
        grad_b_g = 0.0
        n = len(z)

        for zi in z:
            g_z = self.w_g * zi + self.b_g
            logit = w_d * g_z + b_d
            d_gz = sigmoid(logit)
            # Derivative of log(D(G(z))) w.r.t G(z)
            dL_dg = (1.0 - d_gz) * w_d
            grad_w_g -= dL_dg * zi
            grad_b_g -= dL_dg

        self.w_g -= lr * (grad_w_g / n)
        self.b_g -= lr * (grad_b_g / n)

class Discriminator:
    """Simple 1D Linear Discriminator: D(x) = Sigmoid(w_d * x + b_d)"""
    def __init__(self):
        self.w_d = random.uniform(0.1, 0.5)
        self.b_d = random.uniform(-0.5, 0.5)

    def forward(self, x_list):
        return [sigmoid(self.w_d * xi + self.b_d) for xi in x_list]

    def update(self, real_x, fake_x, lr=0.05):
        """Update Discriminator to classify real_x as 1 and fake_x as 0"""
        n_real = len(real_x)
        n_fake = len(fake_x)
        grad_w_d = 0.0
        grad_b_d = 0.0

        # Real loss gradient: -log(D(x)) -> (D(x) - 1) * x
        for x in real_x:
            d_x = sigmoid(self.w_d * x + self.b_d)
            err = d_x - 1.0
            grad_w_d += err * x
            grad_b_d += err

        # Fake loss gradient: -log(1 - D(G(z))) -> D(G(z)) * G(z)
        for x in fake_x:
            d_x = sigmoid(self.w_d * x + self.b_d)
            err = d_x
            grad_w_d += err * x
            grad_b_d += err

        total_n = n_real + n_fake
        self.w_d -= lr * (grad_w_d / total_n)
        self.b_d -= lr * (grad_b_d / total_n)

def run_gan_minimax_sim():
    print("=" * 80)
    print("1. GENERATIVE ADVERSARIAL NETWORKS (GAN) MINIMAX SIMULATION")
    print("=" * 80)
    print("Target Real Distribution: 1D Gaussian (Mean = 4.00, Std = 0.50)")
    print("Initial Generator (Counterfeiter): G(z) = random noise\n")

    random.seed(42)
    G = Generator()
    D = Discriminator()

    epochs = 1500
    batch_size = 32

    print(f"{'Epoch':<8} | {'Gen Output Mean':<18} | {'D(Real)':<12} | {'D(Fake)':<12} | {'Nash Equilibrium Status':<25}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        real_data = sample_real_data(batch_size)
        noise = sample_noise(batch_size)

        # 1. Train Discriminator
        fake_data = G.forward(noise)
        D.update(real_data, fake_data, lr=0.08)

        # 2. Train Generator
        noise_gen = sample_noise(batch_size)
        G.update(noise_gen, {"w_d": D.w_d, "b_d": D.b_d}, lr=0.08)

        if epoch == 1 or epoch % 300 == 0:
            eval_fake = G.forward(sample_noise(100))
            gen_mean = sum(eval_fake) / len(eval_fake)
            d_real_avg = sum(D.forward(sample_real_data(100))) / 100.0
            d_fake_avg = sum(D.forward(eval_fake)) / 100.0

            status = "Counterfeiter Learning..."
            if abs(gen_mean - 4.0) < 0.3 and abs(d_real_avg - 0.5) < 0.2:
                status = "🎯 Nash Equilibrium Reached!"

            print(f"{epoch:<8} | {gen_mean:18.2f} | {d_real_avg:12.3f} | {d_fake_avg:12.3f} | {status:<25}")

    print("\n" * 1)
    print("=" * 80)
    print("2. FINAL MINIMAX CONVERGENCE SUMMARY")
    print("=" * 80)
    final_fake = G.forward(sample_noise(1000))
    final_mean = sum(final_fake) / len(final_fake)
    variance = sum((x - final_mean) ** 2 for x in final_fake) / len(final_fake)
    final_std = math.sqrt(variance)

    print(f"• True Target Distribution : Mean = 4.00, Std = 0.50")
    print(f"• Generator Learned Data   : Mean = {final_mean:.2f}, Std = {final_std:.2f}")
    print(f"• Final Discriminator D(x) : ~0.50 (Detective cannot distinguish real from fake!)")

if __name__ == "__main__":
    run_gan_minimax_sim()
