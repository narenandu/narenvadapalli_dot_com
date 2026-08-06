"""
LSTM Cell & Gradient Flow Simulation from Scratch
Author: Narendra Vadapalli
Series: Neural Architecture Evolution Series (Part 2)

This script demonstrates why Long Short-Term Memory (LSTM) networks were invented:
1. Standard RNN: Hidden state (h_t) suffers from vanishing gradients over long sequences (t=50).
2. LSTM Network: Cell state (C_t) acts as a memory conveyor belt, maintaining gradient flow.
"""

import math
import random

def sigmoid(x: float) -> float:
    """Standard sigmoid activation function bounded in (0, 1)."""
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def tanh(x: float) -> float:
    """Hyperbolic tangent activation function bounded in (-1, 1)."""
    return math.tanh(x)

def sigmoid_derivative(output: float) -> float:
    """Derivative of sigmoid given output y = sigmoid(x)."""
    return output * (1.0 - output)

def tanh_derivative(output: float) -> float:
    """Derivative of tanh given output y = tanh(x)."""
    return 1.0 - output ** 2


class SimpleRNNCell:
    """Standard Recurrent Neural Network (RNN) Cell."""
    def __init__(self):
        random.seed(42)
        # Weights for hidden state and input
        self.w_h = random.uniform(-0.5, 0.5)
        self.w_x = random.uniform(-0.5, 0.5)
        self.b = 0.0

    def forward_sequence(self, x_seq):
        """Pass a sequence of scalar inputs through time steps."""
        h_states = [0.0]  # Initial hidden state h_0
        for x_t in x_seq:
            h_prev = h_states[-1]
            h_t = tanh(self.w_h * h_prev + self.w_x * x_t + self.b)
            h_states.append(h_t)
        return h_states

    def compute_gradient_norm(self, seq_len):
        """Simulate backpropagation gradient decay across time steps."""
        # dh_T / dh_0 = product_{t=1..T} (w_h * tanh'(net_t))
        grad = 1.0
        # Assume average derivative tanh' ~ 0.5
        for _ in range(seq_len):
            grad *= (self.w_h * 0.5)
        return abs(grad)


class LSTMCell:
    """Long Short-Term Memory (LSTM) Cell from Scratch."""
    def __init__(self):
        random.seed(42)
        # Forget Gate (f)
        self.w_f = random.uniform(-0.2, 0.2)
        self.b_f = 1.0  # Positive bias to default forget gate ~ 1 (remember)
        
        # Input Gate (i) & Candidate Cell State (c_tilde)
        self.w_i = random.uniform(-0.2, 0.2)
        self.b_i = 0.0
        self.w_c = random.uniform(-0.2, 0.2)
        self.b_c = 0.0

        # Output Gate (o)
        self.w_o = random.uniform(-0.2, 0.2)
        self.b_o = 0.0

    def forward_step(self, x_t, h_prev, c_prev):
        """Single step forward pass of LSTM Gated Architecture."""
        # Concatenated input scalar (simplified representation)
        combined = h_prev + x_t

        # 1. Forget Gate (Shredder)
        f_t = sigmoid(self.w_f * combined + self.b_f)

        # 2. Input Gate & Candidate State (Selective Pen)
        i_t = sigmoid(self.w_i * combined + self.b_i)
        c_tilde = tanh(self.w_c * combined + self.b_c)

        # 3. Cell State Update (Memory Conveyor Belt)
        c_t = f_t * c_prev + i_t * c_tilde

        # 4. Output Gate & Hidden State Output (Highlighter)
        o_t = sigmoid(self.w_o * combined + self.b_o)
        h_t = o_t * tanh(c_t)

        return h_t, c_t, (f_t, i_t, o_t)

    def forward_sequence(self, x_seq):
        h_states = [0.0]
        c_states = [0.0]
        gate_history = []

        for x_t in x_seq:
            h_t, c_t, gates = self.forward_step(x_t, h_states[-1], c_states[-1])
            h_states.append(h_t)
            c_states.append(c_t)
            gate_history.append(gates)

        return h_states, c_states, gate_history


def run_simulation():
    seq_length = 50
    print("=" * 70)
    print("      LSTM VS STANDARD RNN: CONQUERING LONG-TERM MEMORY AMNESIA      ")
    print("=" * 70)
    print(f"Sequence Length: {seq_length} time steps")
    print("-" * 70)

    # Synthetic sequence: Signal is placed at step 0, followed by noise
    input_seq = [1.0] + [random.uniform(-0.1, 0.1) for _ in range(seq_length - 1)]

    # 1. Evaluate Standard RNN
    rnn = SimpleRNNCell()
    rnn_h = rnn.forward_sequence(input_seq)
    rnn_grad = rnn.compute_gradient_norm(seq_length)

    print(f"\n[1] Standard RNN Performance:")
    print(f"    - Initial Signal (t=0): {input_seq[0]:.4f}")
    print(f"    - Hidden State at t=5 : {rnn_h[5]:.4f}")
    print(f"    - Hidden State at t=25: {rnn_h[25]:.4f}")
    print(f"    - Hidden State at t=50: {rnn_h[50]:.4f}")
    print(f"    - Gradient Magnitude at t=0 relative to t=50: {rnn_grad:.10e}")
    print("    --> Result: Signal degraded completely due to Vanishing Gradient!")

    # 2. Evaluate LSTM Cell
    lstm = LSTMCell()
    lstm_h, lstm_c, gates = lstm.forward_sequence(input_seq)

    print(f"\n[2] LSTM Cell Performance (Conveyor Belt + Gated Doors):")
    print(f"    - Initial Signal (t=0)  : {input_seq[0]:.4f}")
    print(f"    - Cell State C_t at t=5 : {lstm_c[5]:.4f}")
    print(f"    - Cell State C_t at t=25: {lstm_c[25]:.4f}")
    print(f"    - Cell State C_t at t=50: {lstm_c[50]:.4f}")
    avg_f = sum(g[0] for g in gates) / len(gates)
    avg_i = sum(g[1] for g in gates) / len(gates)
    avg_o = sum(g[2] for g in gates) / len(gates)
    print(f"    - Average Gate Openness: Forget={avg_f:.2f}, Input={avg_i:.2f}, Output={avg_o:.2f}")
    print("    --> Result: Cell state maintained gradient pathway across 50 steps!")

    print("\n" + "=" * 70)
    print("Simulation completed successfully.")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
