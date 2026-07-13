# Conceptual ReAct (Reasoning + Acting) Agent Simulator
# File: scripts/agent_react_simulator.py
# Execution: Run in headless mode: `python scripts/agent_react_simulator.py`

import time

# Mock external tool APIs
def weather_api(location):
    if "San Francisco" in location:
        return "Rainy, 58°F, wind 12mph"
    return "Sunny, 72°F"

def traffic_api(origin, destination, condition="clear"):
    base_time = 45 # 45 minutes
    if condition == "rain":
        return f"{base_time + 30} minutes (Heavy Rain delays on US-101 S)"
    return f"{base_time} minutes"

def run_react_agent(task):
    print("=" * 75)
    print(f"Agent Task: {task}")
    print("=" * 75)
    
    step = 1
    # Hardcoded simulation of the thought-action loop
    # In a real agent, this is driven dynamically by parsing LLM outputs
    loop_steps = [
        {
            "thought": "I need to check the current weather in San Francisco to see if there are any environmental delays.",
            "tool": "weather_api",
            "args": "San Francisco",
            "execute": weather_api
        },
        {
            "thought": "The weather is rainy. Rain typically causes traffic delays. I need to calculate the commute time from San Francisco to Silicon Valley taking the rainy weather into account.",
            "tool": "traffic_api",
            "args": ("San Francisco", "Silicon Valley", "rain"),
            "execute": lambda args: traffic_api(*args)
        }
    ]
    
    for current_step in loop_steps:
        print(f"\n[STEP {step}]")
        print(f"🤔 Thought: {current_step['thought']}")
        time.sleep(1) # Simulate reasoning latency
        
        print(f"🛠️  Action: Call tool '{current_step['tool']}' with args: {current_step['args']}")
        # Execute the mock tool
        observation = current_step["execute"](current_step["args"])
        time.sleep(0.5)
        
        print(f"👁️  Observation: {observation}")
        step += 1
        
    print(f"\n[STEP {step}]")
    print("🤔 Thought: I now have both the weather condition (rainy) and the adjusted commute time (75 minutes). I can compile the final answer for the user.")
    print("🛠️  Action: Finish")
    
    final_response = "Commute time from SF to Silicon Valley today is 75 minutes due to heavy rain delays."
    print(f"\n[FINAL RESPONSE] -> {final_response}")
    print("=" * 75)

if __name__ == "__main__":
    run_react_agent("Check weather in SF and find commute time to Silicon Valley")
