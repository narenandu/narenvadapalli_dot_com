#!/usr/bin/env python3
"""
NVIDIA DRIVE & Alpamayo Autonomous Vehicle Architecture Simulator
===================================================================
A standalone, zero-dependency Python simulation demonstrating:
1. Multi-Camera 360° Surround-View to Bird's-Eye-View (BEV) Spatial Transformation.
2. Alpamayo Foundation Model Multi-Candidate Trajectory Generation & Cost Evaluation.
3. ISO 26262 ASIL-D Safety Barrier Redundancy & Deterministic Emergency Fallback.
"""

import math
import random
from typing import List, Dict, Tuple, Optional

# ============================================================================
# 1. 360° SURROUND-VIEW CAMERA PROJECTION & BEV FUSION ENGINE
# ============================================================================

class CameraSensor:
    def __init__(self, name: str, yaw_deg: float, fov_deg: float, range_m: float):
        self.name = name
        self.yaw_rad = math.radians(yaw_deg)
        self.fov_rad = math.radians(fov_deg)
        self.range_m = range_m

    def project_to_bev(self, detected_u: float, detected_depth: float) -> Tuple[float, float]:
        """
        Projects a 2D camera detection (normalized horizontal coordinate u in [-1, 1], depth)
        into the 2D Ego-Centric Bird's-Eye-View (BEV) Cartesian frame (X_ego: forward, Y_ego: left).
        """
        ray_angle = self.yaw_rad + (detected_u * (self.fov_rad / 2.0))
        x_ego = detected_depth * math.cos(ray_angle)
        y_ego = detected_depth * math.sin(ray_angle)
        return x_ego, y_ego


class BEVOccupancyGrid:
    def __init__(self, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], resolution: float):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.res = resolution
        self.cols = int((self.x_max - self.x_min) / self.res)
        self.rows = int((self.y_max - self.y_min) / self.res)
        self.grid = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]

    def add_obstacle(self, x: float, y: float, radius: float, confidence: float):
        """Adds a probabilistic obstacle footprint to the BEV occupancy grid."""
        for r in range(self.rows):
            cell_y = self.y_min + (r + 0.5) * self.res
            for c in range(self.cols):
                cell_x = self.x_min + (c + 0.5) * self.res
                dist = math.hypot(cell_x - x, cell_y - y)
                if dist <= radius:
                    prob = confidence * math.exp(-0.5 * (dist / (radius + 1e-5)) ** 2)
                    self.grid[r][c] = max(self.grid[r][c], prob)

    def render_ascii_view(self, ego_x: float = 0.0, ego_y: float = 0.0, trajectory: Optional[List[Tuple[float, float]]] = None) -> str:
        """Renders a top-down ASCII Bird's Eye View map of the vehicle surroundings."""
        traj_cells = set()
        if trajectory:
            for tx, ty in trajectory:
                col = int((tx - self.x_min) / self.res)
                row = int((ty - self.y_min) / self.res)
                if 0 <= col < self.cols and 0 <= row < self.rows:
                    traj_cells.add((row, col))

        ego_c = int((ego_x - self.x_min) / self.res)
        ego_r = int((ego_y - self.y_min) / self.res)

        lines = []
        border = "+" + "-" * self.cols + "+"
        lines.append(border)

        # Downsample vertical rendering for clean terminal display (every 2 rows)
        for r in range(self.rows - 1, -1, -2):
            row_str = ["|"]
            for c in range(self.cols):
                if (r, c) == (ego_r, ego_c) or (r - 1, c) == (ego_r, ego_c):
                    row_str.append("🏎️")
                elif (r, c) in traj_cells or (r - 1, c) in traj_cells:
                    row_str.append("•")
                elif self.grid[r][c] > 0.6:
                    row_str.append("█")
                elif self.grid[r][c] > 0.2:
                    row_str.append("░")
                else:
                    row_str.append(" ")
            row_str.append("|")
            lines.append("".join(row_str))

        lines.append(border)
        return "\n".join(lines)


# ============================================================================
# 2. ALPAMAYO END-TO-END TRAJECTORY GENERATOR & COST EVALUATOR
# ============================================================================

class TrajectoryCandidate:
    def __init__(self, name: str, waypoints: List[Tuple[float, float, float]]):
        # Waypoints: list of (x_forward_m, y_lateral_m, speed_mps)
        self.name = name
        self.waypoints = waypoints
        self.tracking_cost = 0.0
        self.comfort_cost = 0.0
        self.safety_cost = 0.0
        self.total_cost = 0.0
        self.is_valid = True


class AlpamayoPlanner:
    def __init__(self, horizon_sec: float = 4.0, dt: float = 0.5):
        self.horizon_sec = horizon_sec
        self.dt = dt
        self.steps = int(horizon_sec / dt)

    def generate_candidates(self, current_speed_mps: float, target_speed_mps: float) -> List[TrajectoryCandidate]:
        """Generates multi-modal trajectory candidate maneuvers (Keep Lane, Left Overtake, Right Nudge, Emergency Brake)."""
        candidates = []

        maneuvers = [
            ("Maintain Lane (Center)", 0.0, target_speed_mps, 0.0),
            ("Aggressive Left Overtake", 2.8, target_speed_mps * 1.1, 1.2),
            ("Conservative Left Nudge", 1.2, target_speed_mps * 0.9, 0.5),
            ("Right Lane Change", -3.2, target_speed_mps, -1.0),
            ("Comfortable Deceleration", 0.0, current_speed_mps * 0.5, 0.0),
        ]

        for name, max_lat_offset, target_v, lat_accel in maneuvers:
            waypoints = []
            x, y, v = 0.0, 0.0, current_speed_mps
            for step in range(1, self.steps + 1):
                t = step * self.dt
                # Smooth polynomial lateral transition
                progress = min(1.0, t / (self.horizon_sec * 0.6))
                s_curve = progress * progress * (3.0 - 2.0 * progress)
                y_step = max_lat_offset * s_curve
                
                # Speed ramp
                v_step = current_speed_mps + (target_v - current_speed_mps) * (t / self.horizon_sec)
                x_step = x + v_step * self.dt
                x = x_step
                waypoints.append((x_step, y_step, v_step))

            candidates.append(TrajectoryCandidate(name, waypoints))

        return candidates

    def evaluate_costs(self, candidates: List[TrajectoryCandidate], obstacles: List[Dict[str, float]], target_y: float = 0.0):
        """Calculates multi-objective loss for each candidate trajectory."""
        for cand in candidates:
            # 1. Lateral lane tracking cost
            cand.tracking_cost = sum((wp[1] - target_y) ** 2 for wp in cand.waypoints) * 0.8

            # 2. Kinematic comfort cost (lateral jerk & speed variance)
            lat_jerks = 0.0
            for i in range(len(cand.waypoints) - 1):
                dy1 = cand.waypoints[i][1]
                dy2 = cand.waypoints[i + 1][1]
                lat_jerks += (dy2 - dy1) ** 2
            cand.comfort_cost = lat_jerks * 2.5

            # 3. Spatial Obstacle Proximity Cost
            cand.safety_cost = 0.0
            for wp_x, wp_y, _ in cand.waypoints:
                for obs in obstacles:
                    dist = math.hypot(wp_x - obs["x"], wp_y - obs["y"])
                    safety_margin = obs["radius"] + 1.8  # Vehicle half-width + buffer
                    if dist < safety_margin:
                        cand.safety_cost += (safety_margin - dist) * 150.0

            cand.total_cost = cand.tracking_cost + cand.comfort_cost + cand.safety_cost


# ============================================================================
# 3. ISO 26262 ASIL-D SAFETY BARRIER & REDUNDANCY SENTINEL
# ============================================================================

class ASILDSafetySentinel:
    def __init__(self, min_safe_distance_m: float = 2.5, max_allowable_lat_accel_g: float = 0.45):
        self.min_safe_dist = min_safe_distance_m
        self.max_lat_accel_mps2 = max_allowable_lat_accel_g * 9.81

    def verify_trajectory(self, trajectory: TrajectoryCandidate, obstacles: List[Dict[str, float]]) -> Tuple[bool, str]:
        """
        Independent deterministic validation checking:
        1. Strict collision boundary violations (Control Barrier Function).
        2. Dynamic physical acceleration limits (Anti-rollover envelope).
        """
        # Step 1: Kinematics Check
        for i in range(len(trajectory.waypoints) - 1):
            wp1 = trajectory.waypoints[i]
            wp2 = trajectory.waypoints[i + 1]
            dy = wp2[1] - wp1[1]
            dt = 0.5
            lat_v = dy / dt
            lat_a = abs(lat_v / dt)
            if lat_a > self.max_lat_accel_mps2:
                return False, f"Kinematic Violation: Lat Accel {lat_a:.2f} m/s² exceeds ASIL-D threshold {self.max_lat_accel_mps2:.2f} m/s²"

        # Step 2: Obstacle Spatial Boundary Check
        for wp_x, wp_y, _ in trajectory.waypoints:
            for obs in obstacles:
                dist = math.hypot(wp_x - obs["x"], wp_y - obs["y"])
                critical_threshold = obs["radius"] + self.min_safe_dist
                if dist < critical_threshold:
                    return False, f"Proximity Breach: Distance {dist:.2f}m < Safety Margin {critical_threshold:.2f}m to obstacle '{obs['name']}'"

        return True, "PASSED: Trajectory strictly compliant with ASIL-D safety invariants."

    def generate_fail_safe_fallback(self, current_speed_mps: float, steps: int = 8, dt: float = 0.5) -> TrajectoryCandidate:
        """Generates a deterministic emergency maximum comfort braking trajectory in the current lane."""
        waypoints = []
        x, y, v = 0.0, 0.0, current_speed_mps
        decel = 4.0  # Safe controlled deceleration m/s²
        for _ in range(steps):
            v = max(0.0, v - decel * dt)
            x += v * dt
            waypoints.append((x, y, v))
        return TrajectoryCandidate("ASIL-D Lockstep Fallback (Controlled Stop)", waypoints)


# ============================================================================
# 4. END-TO-END EXECUTION PIPELINE
# ============================================================================

def run_av_architecture_simulation():
    print("=" * 80)
    print("NVIDIA DRIVE & ALPAMAYO AUTONOMOUS VEHICLE ARCHITECTURE SIMULATOR")
    print("=" * 80)

    # 1. Setup 360° Sensor Suite
    cameras = [
        CameraSensor("Front-Center 120°", yaw_deg=0.0, fov_deg=120.0, range_m=120.0),
        CameraSensor("Front-Left 70°", yaw_deg=45.0, fov_deg=70.0, range_m=80.0),
        CameraSensor("Front-Right 70°", yaw_deg=-45.0, fov_deg=70.0, range_m=80.0),
        CameraSensor("Rear-Center 90°", yaw_deg=180.0, fov_deg=90.0, range_m=60.0),
        CameraSensor("Rear-Left 70°", yaw_deg=135.0, fov_deg=70.0, range_m=70.0),
        CameraSensor("Rear-Right 70°", yaw_deg=-135.0, fov_deg=70.0, range_m=70.0),
    ]

    print("\n[1] 360° SURROUND-VIEW SENSOR CALIBRATION MATRIX:")
    print("-" * 80)
    print(f"{'Camera Sensor':<22} | {'Yaw Angle':<12} | {'Field of View':<14} | {'Max Range (m)':<12}")
    print("-" * 80)
    for cam in cameras:
        print(f"{cam.name:<22} | {math.degrees(cam.yaw_rad):>6.1f}°      | {math.degrees(cam.fov_rad):>6.1f}°        | {cam.range_m:>6.1f} m")

    # 2. Build BEV Occupancy Grid and Populate Dynamic Environment
    bev = BEVOccupancyGrid(x_bounds=(-10.0, 50.0), y_bounds=(-12.0, 12.0), resolution=1.0)

    # Scenario: Highway driving at 25 m/s (~90 km/h) with a slow truck at 35m ahead in center lane and a car in right lane
    obstacles = [
        {"name": "Slow Cargo Truck (Center)", "x": 35.0, "y": 0.0, "radius": 2.0, "confidence": 0.98},
        {"name": "Cruising Sedan (Right Lane)", "x": 20.0, "y": -3.5, "radius": 1.5, "confidence": 0.95},
        {"name": "Highway Guardrail (Far Left)", "x": 40.0, "y": 6.0, "radius": 0.8, "confidence": 0.99},
    ]

    for obs in obstacles:
        bev.add_obstacle(obs["x"], obs["y"], obs["radius"], obs["confidence"])

    # 3. Alpamayo Multi-Candidate Trajectory Generation
    planner = AlpamayoPlanner(horizon_sec=4.0, dt=0.5)
    current_speed = 25.0  # 90 km/h
    target_speed = 25.0
    candidates = planner.generate_candidates(current_speed, target_speed)
    planner.evaluate_costs(candidates, obstacles, target_y=0.0)

    # Sort by total neural cost
    candidates.sort(key=lambda c: c.total_cost)

    print("\n[2] ALPAMAYO FOUNDATION MODEL CANDIDATE TRAJECTORY EVALUATION:")
    print("-" * 80)
    print(f"{'Candidate Maneuver':<30} | {'Tracking':<10} | {'Comfort':<9} | {'Safety Cost':<12} | {'Total Score':<10}")
    print("-" * 80)
    for c in candidates:
        print(f"{c.name:<30} | {c.tracking_cost:>8.2f} | {c.comfort_cost:>7.2f} | {c.safety_cost:>11.2f} | {c.total_cost:>10.2f}")

    # 4. ASIL-D Safety Sentinel Verification
    sentinel = ASILDSafetySentinel(min_safe_distance_m=1.8, max_allowable_lat_accel_g=0.45)
    print("\n[3] ISO 26262 ASIL-D DUAL-LOCKSTEP SAFETY SENTINEL ARBITRATION:")
    print("-" * 80)

    selected_trajectory = None
    for cand in candidates:
        passed, reason = sentinel.verify_trajectory(cand, obstacles)
        status_tag = "✅ VALID" if passed else "❌ REJECTED"
        print(f"• Candidate '{cand.name}': {status_tag}")
        print(f"  Reason: {reason}")
        if passed and selected_trajectory is None:
            selected_trajectory = cand

    if selected_trajectory is None:
        print("\n⚠️ ALL NEURAL CANDIDATES REJECTED! Triggering ASIL-D Fail-Safe Emergency Trajectory...")
        selected_trajectory = sentinel.generate_fail_safe_fallback(current_speed)

    print(f"\n🏆 FINAL ARBITRATED ACTION: {selected_trajectory.name}")
    print(f"  Waypoints (X forward, Y lateral, Velocity m/s):")
    for i, (wx, wy, wv) in enumerate(selected_trajectory.waypoints):
        print(f"    t={i*0.5+0.5:.1f}s: X={wx:>5.1f}m, Y={wy:>5.2f}m, V={wv*3.6:>5.1f} km/h")

    # 5. Top-Down Bird's-Eye-View (BEV) ASCII Map
    print("\n[4] UNIFIED BIRD'S-EYE-VIEW (BEV) OCCUPANCY GRID & TRAJECTORY OVERLAY:")
    print("Legend: 🏎️ Ego Vehicle | █ Dense Obstacle | ░ Probabilistic Margin | • Planned Path")
    bev_traj = [(wx, wy) for wx, wy, _ in selected_trajectory.waypoints]
    print(bev.render_ascii_view(ego_x=0.0, ego_y=0.0, trajectory=bev_traj))
    print("=" * 80)


if __name__ == "__main__":
    run_av_architecture_simulation()
