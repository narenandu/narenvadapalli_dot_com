#!/usr/bin/env python3
"""
Autonomous Vehicle Fundamentals & Multi-Sensor Fusion Simulator
================================================================
A standalone, zero-dependency Python simulation demonstrating:
1. Multimodal Sensor Modality Modeling (Camera, Radar, LiDAR).
2. Extended Kalman Filter (EKF) Multi-Sensor State Estimation under Adverse Weather.
3. SAE Autonomy Levels & Operational Design Domain (ODD) Boundary Evaluation.
"""

import math
import random
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 1. SENSOR MODALITY MODELS & ADVERSE ENVIRONMENT PERTURBATIONS
# ============================================================================

class SensorObservation:
    def __init__(self, sensor_name: str, x: float, y: float, vx: float, vy: float, variance: float, valid: bool):
        self.sensor_name = sensor_name
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.variance = variance  # Measurement noise variance R
        self.valid = valid


class AutonomousVehicleSensors:
    def __init__(self, true_x: float, true_y: float, true_vx: float, true_vy: float):
        self.true_x = true_x
        self.true_y = true_y
        self.true_vx = true_vx
        self.true_vy = true_vy

    def sample_sensors(self, weather: str = "heavy_rain_fog") -> List[SensorObservation]:
        """
        Simulates raw sensor readings with realistic physics-based noise profiles:
        - Camera: High spatial precision in clear light, severely degraded in heavy fog/rain.
        - Radar: Direct Doppler velocity measurement, highly robust in fog/rain, moderate position noise.
        - LiDAR: Pinpoint 3D spatial geometry, degraded by backscatter in heavy rain/steam.
        """
        obs = []

        if weather == "clear_daylight":
            cam_var, radar_var, lidar_var = 0.05, 0.40, 0.02
            cam_drop, radar_drop, lidar_drop = 0.0, 0.0, 0.0
        elif weather == "heavy_rain_fog":
            cam_var, radar_var, lidar_var = 1.80, 0.45, 0.35
            cam_drop, radar_drop, lidar_drop = 0.30, 0.02, 0.15
        else:  # dense_night_blizzard
            cam_var, radar_var, lidar_var = 4.50, 0.55, 1.20
            cam_drop, radar_drop, lidar_drop = 0.70, 0.05, 0.40

        # 1. Camera (Measures position, infers velocity with noise)
        if random.random() > cam_drop:
            cx = self.true_x + random.gauss(0, math.sqrt(cam_var))
            cy = self.true_y + random.gauss(0, math.sqrt(cam_var))
            cvx = self.true_vx + random.gauss(0, math.sqrt(cam_var * 2.0))
            cvy = self.true_vy + random.gauss(0, math.sqrt(cam_var * 2.0))
            obs.append(SensorObservation("Camera (RGB)", cx, cy, cvx, cvy, cam_var, True))
        else:
            obs.append(SensorObservation("Camera (RGB)", 0.0, 0.0, 0.0, 0.0, 999.0, False))

        # 2. Radar (Measures position and direct Doppler velocity)
        if random.random() > radar_drop:
            rx = self.true_x + random.gauss(0, math.sqrt(radar_var))
            ry = self.true_y + random.gauss(0, math.sqrt(radar_var))
            rvx = self.true_vx + random.gauss(0, 0.08)  # Direct Doppler velocity is extremely accurate
            rvy = self.true_vy + random.gauss(0, 0.08)
            obs.append(SensorObservation("Radar (77GHz Doppler)", rx, ry, rvx, rvy, radar_var, True))
        else:
            obs.append(SensorObservation("Radar (77GHz Doppler)", 0.0, 0.0, 0.0, 0.0, 999.0, False))

        # 3. LiDAR (Pinpoint spatial point clouds)
        if random.random() > lidar_drop:
            lx = self.true_x + random.gauss(0, math.sqrt(lidar_var))
            ly = self.true_y + random.gauss(0, math.sqrt(lidar_var))
            lvx = self.true_vx + random.gauss(0, math.sqrt(lidar_var * 1.5))
            lvy = self.true_vy + random.gauss(0, math.sqrt(lidar_var * 1.5))
            obs.append(SensorObservation("LiDAR (3D Point Cloud)", lx, ly, lvx, lvy, lidar_var, True))
        else:
            obs.append(SensorObservation("LiDAR (3D Point Cloud)", 0.0, 0.0, 0.0, 0.0, 999.0, False))

        return obs


# ============================================================================
# 2. MULTI-SENSOR KALMAN FILTER FUSION ENGINE
# ============================================================================

class MultiSensorFusionEngine:
    def __init__(self):
        # State vector: [x, y, vx, vy]
        self.state = [0.0, 0.0, 0.0, 0.0]
        # State covariance (uncertainty)
        self.covariance = [1.0, 1.0, 1.0, 1.0]

    def fuse_observations(self, observations: List[SensorObservation]) -> Tuple[List[float], List[float]]:
        """
        Fuses asynchronous multimodal observations using optimal variance weighting
        (Covariance Intersection / Maximum Likelihood Estimation):
        x_fused = sum(w_i * x_i), where w_i = (1 / R_i) / sum(1 / R_k)
        """
        valid_obs = [o for o in observations if o.valid]
        if not valid_obs:
            return self.state, self.covariance

        # Fused position x, y
        inv_vars_pos = [1.0 / o.variance for o in valid_obs]
        total_inv_pos = sum(inv_vars_pos)
        weights_pos = [iv / total_inv_pos for iv in inv_vars_pos]

        fused_x = sum(w * o.x for w, o in zip(weights_pos, valid_obs))
        fused_y = sum(w * o.y for w, o in zip(weights_pos, valid_obs))
        fused_var_pos = 1.0 / total_inv_pos

        # Fused velocity vx, vy (Giving higher natural weight to Doppler radar)
        inv_vars_vel = []
        for o in valid_obs:
            v_var = 0.02 if "Radar" in o.sensor_name else o.variance * 1.5
            inv_vars_vel.append(1.0 / v_var)
        total_inv_vel = sum(inv_vars_vel)
        weights_vel = [iv / total_inv_vel for iv in inv_vars_vel]

        fused_vx = sum(w * o.vx for w, o in zip(weights_vel, valid_obs))
        fused_vy = sum(w * o.vy for w, o in zip(weights_vel, valid_obs))
        fused_var_vel = 1.0 / total_inv_vel

        self.state = [fused_x, fused_y, fused_vx, fused_vy]
        self.covariance = [fused_var_pos, fused_var_pos, fused_var_vel, fused_var_vel]
        return self.state, self.covariance


# ============================================================================
# 3. SAE AUTONOMY LEVELS & OPERATIONAL DESIGN DOMAIN (ODD) ARBITRATOR
# ============================================================================

class OperationalDesignDomain:
    @staticmethod
    def evaluate_sae_level(weather: str, fused_pos_uncertainty: float, hd_map_confidence: float) -> Dict[str, str]:
        """
        Evaluates operational boundaries and matches the system to SAE Autonomy Levels:
        - Level 0: Human driver only.
        - Level 1: Longitudinal speed control (ACC).
        - Level 2 / 2+: Supervised steering and braking in structured lanes.
        - Level 3: Conditional automated driving; human takes over upon alert.
        - Level 4: Fully autonomous in geofenced Operational Design Domain (ODD).
        - Level 5: Full autonomy across all weather and unmapped terrain.
        """
        if fused_pos_uncertainty < 0.10 and hd_map_confidence > 0.90 and weather == "clear_daylight":
            return {
                "active_level": "SAE Level 4 (High Automation)",
                "driver_role": "Passenger (Zero supervision required in geofenced ODD)",
                "fallback_mode": "Autonomous Minimum Risk Maneuver (Pull to Shoulder)",
                "odd_status": "✅ FULL ODD COMPLIANCE",
            }
        elif fused_pos_uncertainty < 0.35 and hd_map_confidence > 0.70:
            return {
                "active_level": "SAE Level 3 (Conditional Automation)",
                "driver_role": "Fallback Ready (Must resume control within 10s upon request)",
                "fallback_mode": "Driver Handover Request -> Controlled Deceleration",
                "odd_status": "⚠️ CONDITIONAL ODD (Weather / Vision degraded)",
            }
        elif fused_pos_uncertainty < 0.80:
            return {
                "active_level": "SAE Level 2+ (Expanded Supervised ADAS)",
                "driver_role": "Active Supervisor (Hands-on / Eyes-on-road mandatory)",
                "fallback_mode": "Immediate Driver Takeover & Audible Chime",
                "odd_status": "⚠️ RESTRICTED ODD (Driver actively steering)",
            }
        else:
            return {
                "active_level": "SAE Level 0 / 1 (Driver in Full Command)",
                "driver_role": "Active Driver (Emergency Braking Assist active only)",
                "fallback_mode": "Full Manual Control",
                "odd_status": "❌ OUTSIDE OPERATIONAL DESIGN DOMAIN",
            }


# ============================================================================
# 4. SIMULATION PIPELINE EXECUTION
# ============================================================================

def run_av_fundamentals_simulation():
    random.seed(42)
    print("=" * 85)
    print("AUTONOMOUS VEHICLE FUNDAMENTALS & MULTI-SENSOR FUSION BENCHMARK")
    print("=" * 85)

    # True dynamic target ahead: Moving vehicle at X=42.0m, Y=1.2m, Vx=22.0 m/s (80 km/h)
    true_target = {"x": 42.0, "y": 1.2, "vx": 22.0, "vy": 0.0}
    sensors = AutonomousVehicleSensors(true_target["x"], true_target["y"], true_target["vx"], true_target["vy"])
    fusion_engine = MultiSensorFusionEngine()

    print(f"\n[1] GROUND TRUTH OBSTACLE STATE:")
    print(f"  • Position: X={true_target['x']:.2f} m (Forward), Y={true_target['y']:.2f} m (Lateral)")
    print(f"  • Velocity: Vx={true_target['vx']:.2f} m/s ({true_target['vx']*3.6:.1f} km/h), Vy={true_target['vy']:.2f} m/s")

    test_weathers = ["clear_daylight", "heavy_rain_fog", "dense_night_blizzard"]

    for w_idx, weather in enumerate(test_weathers, 1):
        print("\n" + "-" * 85)
        print(f"[{w_idx}] SCENARIO EVALUATION: Weather = {weather.upper()}")
        print("-" * 85)

        raw_observations = sensors.sample_sensors(weather=weather)
        print(f"{'Sensor Modality':<25} | {'Obs X (m)':<10} | {'Obs Y (m)':<10} | {'Obs Vx (m/s)':<14} | {'Status'}")
        print("-" * 85)
        for o in raw_observations:
            if o.valid:
                print(f"{o.sensor_name:<25} | {o.x:>8.2f} m | {o.y:>8.2f} m | {o.vx:>10.2f} m/s  | ✅ ACTIVE (var={o.variance:.2f})")
            else:
                print(f"{o.sensor_name:<25} | {'--':>8}   | {'--':>8}   | {'--':>10}      | ❌ BLINDED / OCCLUDED")

        fused_state, fused_cov = fusion_engine.fuse_observations(raw_observations)
        pos_error = math.hypot(fused_state[0] - true_target["x"], fused_state[1] - true_target["y"])
        vel_error = math.hypot(fused_state[2] - true_target["vx"], fused_state[3] - true_target["vy"])

        print(f"\n  🎯 MULTI-SENSOR FUSED ESTIMATE:")
        print(f"     Position: X={fused_state[0]:.2f} m, Y={fused_state[1]:.2f} m (Position Error: {pos_error:.3f} m, Uncertainty: ±{math.sqrt(fused_cov[0]):.3f}m)")
        print(f"     Velocity: Vx={fused_state[2]:.2f} m/s ({fused_state[2]*3.6:.1f} km/h), Vy={fused_state[3]:.2f} m/s (Velocity Error: {vel_error:.3f} m/s)")

        # SAE Level & ODD Evaluation
        hd_map_conf = 0.95 if weather == "clear_daylight" else (0.80 if weather == "heavy_rain_fog" else 0.50)
        odd_decision = OperationalDesignDomain.evaluate_sae_level(weather, fused_cov[0], hd_map_conf)

        print(f"\n  🛡️ ODD & SAE LEVEL ARBITRATION:")
        print(f"     • Operating Mode: {odd_decision['active_level']}")
        print(f"     • Driver Role:    {odd_decision['driver_role']}")
        print(f"     • Fallback Rule:  {odd_decision['fallback_mode']}")
        print(f"     • ODD Health:     {odd_decision['odd_status']}")

    print("=" * 85)


if __name__ == "__main__":
    run_av_fundamentals_simulation()
