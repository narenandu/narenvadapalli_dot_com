# scripts/world_model_simulation.py
import dataclasses
from typing import List, Tuple

@dataclasses.dataclass
class Vector3D:
    x: float
    y: float
    z: float

@dataclasses.dataclass
class WorldState:
    """Represents ground-truth 3D physical environment state."""
    timestamp: float
    object_positions: dict[str, Vector3D]
    object_velocities: dict[str, Vector3D]
    gravity: float = 9.81

@dataclasses.dataclass
class Observation:
    """Sensory observation payload projected from WorldState."""
    camera_frame_id: int
    depth_map_shape: Tuple[int, int]
    detected_entities: List[str]

class StateSimulator:
    """State Simulator: Computes physical state evolution S_{t+1}."""
    def step(self, current_state: WorldState, action_vector: Vector3D, dt: float) -> WorldState:
        new_positions = {}
        for obj_name, pos in current_state.object_positions.items():
            vel = current_state.object_velocities.get(obj_name, Vector3D(0, 0, 0))
            # Apply kinematic physics step: x_new = x + v*dt
            new_positions[obj_name] = Vector3D(
                x=pos.x + vel.x * dt + action_vector.x * dt,
                y=pos.y + vel.y * dt + action_vector.y * dt,
                z=pos.z + vel.z * dt - current_state.gravity * (dt ** 2)
            )
        return WorldState(
            timestamp=current_state.timestamp + dt,
            object_positions=new_positions,
            object_velocities=current_state.object_velocities
        )

class ActionPlanner:
    """Action Planner: Computes action a_t given observation o_t and goal."""
    def plan_next_action(self, observation: Observation, goal_entity: str) -> Vector3D:
        if goal_entity in observation.detected_entities:
            print(f"[ActionPlanner] Target '{goal_entity}' detected. Initiating trajectory.")
            return Vector3D(x=0.5, y=0.0, z=0.1)
        print(f"[ActionPlanner] Searching for target '{goal_entity}'...")
        return Vector3D(x=0.0, y=0.0, z=0.0)

# Quick verification run
if __name__ == "__main__":
    initial_state = WorldState(
        timestamp=0.0,
        object_positions={"robot_gripper": Vector3D(0, 1, 0)},
        object_velocities={"robot_gripper": Vector3D(0.1, 0, 0)}
    )
    simulator = StateSimulator()
    planner = ActionPlanner()
    
    obs = Observation(camera_frame_id=101, depth_map_shape=(640, 480), detected_entities=["robot_gripper"])
    action = planner.plan_next_action(obs, goal_entity="robot_gripper")
    next_state = simulator.step(initial_state, action, dt=0.1)
    
    print(f"Updated World State at t={next_state.timestamp:.1f}s: {next_state.object_positions['robot_gripper']}")
