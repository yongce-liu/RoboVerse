from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from metasim.utils import configclass
from metasim.cfg.robots.base_robot_cfg import BaseActuatorCfg, BaseRobotCfg


@configclass
class Go2Cfg(BaseRobotCfg):
    name: str = "go2"
    num_joints: int = 12
    usd_path: str = MISSING
    xml_path: str = MISSING
    urdf_path: str ="roboverse_learn/unitree_rl/robots/go2/assets/urdf/go2.urdf"
    enabled_gravity: bool = True
    fix_base_link: bool = False
    enabled_self_collisions: bool = False
    isaacgym_flip_visual_attachments: bool = False
    collapse_fixed_joints: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        'FL_hip': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RL_hip': BaseActuatorCfg(stiffness=20., damping=0.5),
        'FR_hip': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RR_hip': BaseActuatorCfg(stiffness=20., damping=0.5),

        'FL_thigh': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RL_thigh': BaseActuatorCfg(stiffness=20., damping=0.5),
        'FR_thigh': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RR_thigh': BaseActuatorCfg(stiffness=20., damping=0.5),

        'FL_calf': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RL_calf': BaseActuatorCfg(stiffness=20., damping=0.5),
        'FR_calf': BaseActuatorCfg(stiffness=20., damping=0.5),
        'RR_calf': BaseActuatorCfg(stiffness=20., damping=0.5),
    }

    joint_limits: dict[str, tuple[float, float]] = {
        # rad
        'FL_hip': (-0.837758, 0.837758),
        'RL_hip': (-0.837758, 0.837758),
        'FR_hip': (-0.837758, 0.837758),
        'RR_hip': (-0.837758, 0.837758),

        'FL_thigh': (-3.49066, 1.5708),
        'RL_thigh': (-4.53786, 0.523599),
        'FR_thigh': (-3.49066, 1.5708),
        'RR_thigh': (-4.53786, 0.523599),

        'FL_calf': (-2.72271, -0.837758),
        'RL_calf': (-2.72271, -0.837758),
        'FR_calf': (-2.72271, -0.837758),
        'RR_calf': (-2.72271, -0.837758),
    }

    # torque_limits: dict[str, float] = {
    #     'FL_hip': 88,
    #     'RL_hip': 88,
    #     'FR_hip': 88,
    #     'RR_hip': 88,

    #     'FL_thigh': 88,
    #     'RL_thigh': 88,
    #     'FR_thigh': 88,
    #     'RR_thigh': 88,

    #     'FL_calf': 88,
    #     'RL_calf': 88,
    #     'FR_calf': 88,
    #     'RR_calf': 88,
    # }

    default_joint_positions: dict[str, float] = {  # = target angles [rad] when action = 0.0
        'FL_hip': 0.1,
        'RL_hip': 0.1,
        'FR_hip': -0.1,
        'RR_hip': -0.1,

        'FL_thigh': 0.8,
        'RL_thigh': 1.0,
        'FR_thigh': 0.8,
        'RR_thigh': 1.0,

        'FL_calf': -1.5,
        'RL_calf': -1.5,
        'FR_calf': -1.5,
        'RR_calf': -1.5,
    }

    control_type: dict[str, Literal["position", "effort"]] = {
        'FL_hip': "effort",
        'RL_hip': "effort",
        'FR_hip': "effort",
        'RR_hip': "effort",

        'FL_thigh': "effort",
        'RL_thigh': "effort",
        'FR_thigh': "effort",
        'RR_thigh': "effort",

        'FL_calf': "effort",
        'RL_calf': "effort",
        'FR_calf': "effort",
        'RR_calf': "effort",
    }

    # rigid body name substrings, to find indices of different rigid bodies.
    feet_links: list[str] = ["foot"]
    # knee_links: list[str] = ["knee"]
    # elbow_links: list[str] = ["elbow"]
    # wrist_links: list[str] = ["rubber_hand"]
    # torso_links: list[str] = ["torso_link"]
    # terminate_contacts_links = ["pelvis", "torso", "waist", "shoulder", "elbow", "wrist"]
    # penalized_contacts_links: list[str] = ["hip", "knee"]

    # # joint substrings, to find indices of joints.

    # left_yaw_roll_joints = ["left_hip_yaw", "left_hip_roll"]
    # right_yaw_roll_joints = ["right_hip_yaw", "right_hip_roll"]
    # upper_body_joints = ["shoulder", "elbow", "torso"]
