from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass

from .base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class G1Dex3Cfg(BaseRobotCfg):
    name: str = "g1"
    num_joints: int = 43
    usd_path: str = MISSING
    xml_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_with_hand_rev_1_0.xml"
    urdf_path: str = "roboverse_data/robots/g1/urdf/g1_29dof_with_hand_rev_1_0.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "left_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "left_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "left_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "left_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),

        "right_hip_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_hip_roll_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_hip_yaw_joint": BaseActuatorCfg(stiffness=150, damping=5),
        "right_knee_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "right_ankle_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_ankle_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),

        "waist_yaw_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "waist_roll_joint": BaseActuatorCfg(stiffness=200, damping=5),
        "waist_pitch_joint": BaseActuatorCfg(stiffness=200, damping=5),

        "left_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "left_wrist_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "left_wrist_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "left_wrist_yaw_joint": BaseActuatorCfg(stiffness=20, damping=4),

        "right_shoulder_pitch_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_roll_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_shoulder_yaw_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_elbow_joint": BaseActuatorCfg(stiffness=40, damping=10),
        "right_wrist_roll_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_wrist_pitch_joint": BaseActuatorCfg(stiffness=20, damping=4),
        "right_wrist_yaw_joint": BaseActuatorCfg(stiffness=20, damping=4),

        "left_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "left_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=1),

        "right_hand_thumb_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_thumb_1_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_thumb_2_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_middle_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_middle_1_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_index_0_joint": BaseActuatorCfg(stiffness=5, damping=1),
        "right_hand_index_1_joint": BaseActuatorCfg(stiffness=5, damping=1),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # Hips & legs
        "left_hip_pitch_joint": (-2.5307, 2.8798),
        "left_hip_roll_joint": (-0.5236, 2.9671),
        "left_hip_yaw_joint": (-2.7576, 2.7576),
        "left_knee_joint": (-0.087267, 2.8798),
        "left_ankle_pitch_joint": (-0.87267, 0.5236),
        "left_ankle_roll_joint": (-0.2618, 0.2618),
        "right_hip_pitch_joint": (-2.5307, 2.8798),
        "right_hip_roll_joint": (-2.9671, 0.5236),
        "right_hip_yaw_joint": (-2.7576, 2.7576),
        "right_knee_joint": (-0.087267, 2.8798),
        "right_ankle_pitch_joint": (-0.87267, 0.5236),
        "right_ankle_roll_joint": (-0.2618, 0.2618),

        # Waist
        "waist_yaw_joint": (-2.618, 2.618),
        "waist_roll_joint": (-0.52, 0.52),
        "waist_pitch_joint": (-0.52, 0.52),

        # Shoulders & arms
        "left_shoulder_pitch_joint": (-3.0892, 2.6704),
        "left_shoulder_roll_joint": (-1.5882, 2.2515),
        "left_shoulder_yaw_joint": (-2.618, 2.618),
        "left_elbow_joint": (-1.0472, 2.0944),
        "left_wrist_roll_joint": (-1.972222, 1.972222),
        "left_wrist_pitch_joint": (-1.61443, 1.61443),
        "left_wrist_yaw_joint": (-1.61443, 1.61443),
        "right_shoulder_pitch_joint": (-3.0892, 2.6704),
        "right_shoulder_roll_joint": (-2.2515, 1.5882),
        "right_shoulder_yaw_joint": (-2.618, 2.618),
        "right_elbow_joint": (-1.0472, 2.0944),
        "right_wrist_roll_joint": (-1.972222, 1.972222),
        "right_wrist_pitch_joint": (-1.61443, 1.61443),
        "right_wrist_yaw_joint": (-1.61443, 1.61443),

        # Hands
        "left_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "left_hand_thumb_1_joint": (-0.61086523, 1.04719755),
        "left_hand_thumb_2_joint": (0.0, 1.74532925),
        "left_hand_middle_0_joint": (-1.57079632, 0.0),
        "left_hand_middle_1_joint": (-1.74532925, 0.0),
        "left_hand_index_0_joint": (-1.57079632, 0.0),
        "left_hand_index_1_joint": (-1.74532925, 0.0),
        "right_hand_thumb_0_joint": (-1.04719755, 1.04719755),
        "right_hand_thumb_1_joint": (-1.04719755, 0.61086523),
        "right_hand_thumb_2_joint": (-1.74532925, 0.0),
        "right_hand_middle_0_joint": (0.0, 1.57079632),
        "right_hand_middle_1_joint": (0.0, 1.74532925),
        "right_hand_index_0_joint": (0.0, 1.57079632),
        "right_hand_index_1_joint": (0.0, 1.74532925),
    }

    torque_limits: dict[str, float] = {
        # Hips & legs
        "left_hip_pitch_joint": 88,
        "left_hip_roll_joint": 139,
        "left_hip_yaw_joint": 88,
        "left_knee_joint": 139,
        "left_ankle_pitch_joint": 35,
        "left_ankle_roll_joint": 35,
        "right_hip_pitch_joint": 88,
        "right_hip_roll_joint": 139,
        "right_hip_yaw_joint": 88,
        "right_knee_joint": 139,
        "right_ankle_pitch_joint": 35,
        "right_ankle_roll_joint": 35,

        # Waist
        "waist_yaw_joint": 88,
        "waist_roll_joint": 35,
        "waist_pitch_joint": 35,

        # Shoulders & arms
        "left_shoulder_pitch_joint": 25,
        "left_shoulder_roll_joint": 25,
        "left_shoulder_yaw_joint": 25,
        "left_elbow_joint": 25,
        "left_wrist_roll_joint": 25,
        "left_wrist_pitch_joint": 5,
        "left_wrist_yaw_joint": 5,
        "right_shoulder_pitch_joint": 25,
        "right_shoulder_roll_joint": 25,
        "right_shoulder_yaw_joint": 25,
        "right_elbow_joint": 25,
        "right_wrist_roll_joint": 25,
        "right_wrist_pitch_joint": 5,
        "right_wrist_yaw_joint": 5,

        # Hands
        "left_hand_thumb_0_joint": 2.45,
        "left_hand_thumb_1_joint": 1.4,
        "left_hand_thumb_2_joint": 1.4,
        "left_hand_middle_0_joint": 1.4,
        "left_hand_middle_1_joint": 1.4,
        "left_hand_index_0_joint": 1.4,
        "left_hand_index_1_joint": 1.4,
        "right_hand_thumb_0_joint": 2.45,
        "right_hand_thumb_1_joint": 1.4,
        "right_hand_thumb_2_joint": 1.4,
        "right_hand_middle_0_joint": 1.4,
        "right_hand_middle_1_joint": 1.4,
        "right_hand_index_0_joint": 1.4,
        "right_hand_index_1_joint": 1.4,
    }

    default_joint_positions: dict[str, float] = {
        # Hips & legs
        "left_hip_pitch_joint": -0.4,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.8,
        "left_ankle_pitch_joint": -0.4,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": -0.4,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.8,
        "right_ankle_pitch_joint": -0.4,
        "right_ankle_roll_joint": 0.0,

        # Waist
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,

        # Shoulders & arms
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,

        # Hands
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.0,
        "left_hand_thumb_2_joint": 0.0,
        "left_hand_middle_0_joint": 0.0,
        "left_hand_middle_1_joint": 0.0,
        "left_hand_index_0_joint": 0.0,
        "left_hand_index_1_joint": 0.0,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": 0.0,
        "right_hand_thumb_2_joint": 0.0,
        "right_hand_middle_0_joint": 0.0,
        "right_hand_middle_1_joint": 0.0,
        "right_hand_index_0_joint": 0.0,
        "right_hand_index_1_joint": 0.0,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        # Hips & legs
        "left_hip_pitch_joint": "effort",
        "left_hip_roll_joint": "effort",
        "left_hip_yaw_joint": "effort",
        "left_knee_joint": "effort",
        "left_ankle_pitch_joint": "effort",
        "left_ankle_roll_joint": "effort",

        "right_hip_pitch_joint": "effort",
        "right_hip_roll_joint": "effort",
        "right_hip_yaw_joint": "effort",
        "right_knee_joint": "effort",
        "right_ankle_pitch_joint": "effort",
        "right_ankle_roll_joint": "effort",

        # Waist
        "waist_yaw_joint": "effort",
        "waist_roll_joint": "effort",
        "waist_pitch_joint": "effort",

        # Shoulders & arms
        "left_shoulder_pitch_joint": "effort",
        "left_shoulder_roll_joint": "effort",
        "left_shoulder_yaw_joint": "effort",
        "left_elbow_joint": "effort",
        "left_wrist_roll_joint": "effort",
        "left_wrist_pitch_joint": "effort",
        "left_wrist_yaw_joint": "effort",

        "right_shoulder_pitch_joint": "effort",
        "right_shoulder_roll_joint": "effort",
        "right_shoulder_yaw_joint": "effort",
        "right_elbow_joint": "effort",
        "right_wrist_roll_joint": "effort",
        "right_wrist_pitch_joint": "effort",
        "right_wrist_yaw_joint": "effort",

        # Hands
        "left_hand_thumb_0_joint": "effort",
        "left_hand_thumb_1_joint": "effort",
        "left_hand_thumb_2_joint": "effort",
        "left_hand_middle_0_joint": "effort",
        "left_hand_middle_1_joint": "effort",
        "left_hand_index_0_joint": "effort",
        "left_hand_index_1_joint": "effort",

        "right_hand_thumb_0_joint": "effort",
        "right_hand_thumb_1_joint": "effort",
        "right_hand_thumb_2_joint": "effort",
        "right_hand_middle_0_joint": "effort",
        "right_hand_middle_1_joint": "effort",
        "right_hand_index_0_joint": "effort",
        "right_hand_index_1_joint": "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["ankle_roll"]
    knee_links: list[str] = ["knee"]
    elbow_links: list[str] = ["elbow"]
    wrist_links: list[str] = ["rubber_hand"]
    torso_links: list[str] = ["torso_link"]
    terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    penalized_contacts_links: list[str] = ["hip", "knee"]

    # joint substrings, to find indices of joints.

    left_yaw_roll_joints = ["left_hip_yaw", "left_hip_roll"]
    right_yaw_roll_joints = ["right_hip_yaw", "right_hip_roll"]
    upper_body_joints = ["shoulder", "elbow", "torso"]
