# scripts/moe_routing_simulator.py
import random

def simulate_moe_routing():
    # List of 8 available domain experts in our FFN layer
    experts = [
        "Code Expert", "Math Expert", "Creative Writer", "Vision Analyzer",
        "Audio Transcriber", "System Architect", "Translator", "General Assistant"
    ]
    
    # Simple input prompts to test routing
    prompts = [
        ("Write a fast sorting function in Rust.", ["Code Expert", "System Architect"]),
        ("Calculate the derivative of x^2 + 5x.", ["Math Expert", "General Assistant"]),
        ("Analyze the audio waveform features.", ["Audio Transcriber", "System Architect"]),
    ]
    
    print("=== STARTING INKLING MoE ROUTER SIMULATOR ===")
    
    for i, (prompt, target_experts) in enumerate(prompts, 1):
        print(f"\n[Token Sequence {i}]: \"{prompt}\"")
        
        # Calculate routing probabilities (adding minor random noise for simulation)
        routed = []
        for exp in experts:
            base_prob = 0.85 if exp in target_experts else 0.05
            prob = min(1.0, max(0.0, base_prob + random.uniform(-0.02, 0.02)))
            routed.append((exp, prob))
        
        # Sort experts by router probability score
        routed.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-2 active experts for execution
        top_2 = routed[:2]
        inactive = routed[2:]
        
        print("🚦 Gating Router Probabilities:")
        for exp, p in top_2:
            print(f"  🟢 Active: {exp:<20} (Routing Score: {p:.2f})")
        for exp, p in inactive[:2]:
            print(f"  🔴 Inactive: {exp:<20} (Routing Score: {p:.2f})")
            
    print("\n=============================================")

if __name__ == "__main__":
    simulate_moe_routing()
