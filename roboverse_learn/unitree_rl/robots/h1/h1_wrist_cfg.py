from __future__ import annotations

from typing import Literal

from metasim.utils import configclass
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg



@configclass
class H1WristCfg(BaseRobotCfg):
    name: str = "h1_wrist"
    num_joints: int = 19  # 19 revolute, 4 fixed
    usd_path: str = "roboverse_learn/unitree_rl/robots/h1/assets/usd/h1_wrist.usd"
    mjcf_path: str = "roboverse_learn/unitree_rl/robots/h1/assets/mjcf/h1.xml"
    urdf_path: str = "roboverse_learn/unitree_rl/robots/h1/assets/urdf/h1_wrist.urdf"
    # urdf_path: str = "/home/panwei/RoboVerse/roboverse_data/robots/h1/urdf/h1_wrist_copy.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = False  # must set false to keep wrist link exist

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "left_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "left_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "right_hip_yaw": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_pitch": BaseActuatorCfg(stiffness=200, damping=5),
        "right_knee": BaseActuatorCfg(stiffness=300, damping=6),
        "right_ankle": BaseActuatorCfg(stiffness=40, damping=2),
        "torso": BaseActuatorCfg(stiffness=300, damping=6),
        "left_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "left_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "left_elbow": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_pitch": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_roll": BaseActuatorCfg(stiffness=100, damping=2),
        "right_shoulder_yaw": BaseActuatorCfg(stiffness=100, damping=2),
        "right_elbow": BaseActuatorCfg(stiffness=100, damping=2),
    }

    torque_limits: dict[str, float] = {
        "left_hip_yaw": 200.0,
        "left_hip_roll": 200.0,
        "left_hip_pitch": 200.0,
        "left_knee": 300.0,
        "left_ankle": 40.0,
        "right_hip_yaw": 200.0,
        "right_hip_roll": 200.0,
        "right_hip_pitch": 200.0,
        "right_knee": 300.0,
        "right_ankle": 40.0,
        "torso": 200.0,
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 18.0,
        "left_elbow": 18.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 18.0,
        "right_elbow": 18.0,
    }

    joint_limits: dict[str, tuple[float, float]] = {
        "left_hip_yaw": (-0.43, 0.43),
        "left_hip_roll": (-0.43, 0.43),
        "left_hip_pitch": (-3.14, 2.53),
        "left_knee": (-0.26, 2.05),
        "left_ankle": (-0.87, 0.52),
        "right_hip_yaw": (-0.43, 0.43),
        "right_hip_roll": (-0.43, 0.43),
        "right_hip_pitch": (-3.14, 2.53),
        "right_knee": (-0.26, 2.05),
        "right_ankle": (-0.87, 0.52),
        "torso": (-2.35, 2.35),
        "left_shoulder_pitch": (-2.87, 2.87),
        "left_shoulder_roll": (-0.34, 3.11),
        "left_shoulder_yaw": (-1.3, 4.45),
        "left_elbow": (-1.25, 2.61),
        "right_shoulder_pitch": (-2.87, 2.87),
        "right_shoulder_roll": (-3.11, 0.34),
        "right_shoulder_yaw": (-4.45, 1.3),
        "right_elbow": (-1.25, 2.61),
    }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        "left_hip_yaw": 0.0,
        "left_hip_roll": 0.0,
        "left_hip_pitch": -0.4,
        "left_knee": 0.8,
        "left_ankle": -0.4,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0,
        "right_hip_pitch": -0.4,
        "right_knee": 0.8,
        "right_ankle": -0.4,
        "torso": 0.0,
        "left_shoulder_pitch": 0.0,
        "left_shoulder_roll": 0,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 0.0,
        "right_shoulder_pitch": 0.0,
        "right_shoulder_roll": 0.0,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        "left_hip_yaw": "effort",
        "left_hip_roll": "effort",
        "left_hip_pitch": "effort",
        "left_knee": "effort",
        "left_ankle": "effort",
        "right_hip_yaw": "effort",
        "right_hip_roll": "effort",
        "right_hip_pitch": "effort",
        "right_knee": "effort",
        "right_ankle": "effort",
        "torso": "effort",
        "left_shoulder_pitch": "effort",
        "left_shoulder_roll": "effort",
        "left_shoulder_yaw": "effort",
        "left_elbow": "effort",
        "right_shoulder_pitch": "effort",
        "right_shoulder_roll": "effort",
        "right_shoulder_yaw": "effort",
        "right_elbow": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = [
        "left_ankle",
        "right_ankle",
    ]
    knee_links: list[str] = [
        "left_knee",
        "right_knee",
    ]
    elbow_links: list[str] = [
        "left_elbow",
        "right_elbow",
    ]
    terminate_after_contacts_on_links: list[str] = [
        "left_elbow",
        "right_elbow",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
    ]
    terminate_contacts_links: list[str] = ["pelvis", "torso", "shoulder", "elbow"]
    penalized_contacts_links: list[str] = ["hip", "knee"]
    wrist_links: list[str] = ["wrist"]
    torso_links: list[str] = ["torso"]

    # joint substrings, to find indices of joints.
    left_yaw_roll_joints = ["left_hip_yaw", "left_hip_roll"]
    right_yaw_roll_joints = ["right_hip_yaw", "right_hip_roll"]
    upper_body_joints = ["shoulder", "elbow", "torso"]
