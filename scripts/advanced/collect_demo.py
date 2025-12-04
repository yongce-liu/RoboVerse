"""Demo collection script with domain randomization support.

Collects demonstration data by replaying trajectories with optional domain randomization.

Randomization Levels:
- Level 0: No randomization
- Level 1: Scene + Material randomization
- Level 2: Level 1 + Lighting randomization
- Level 3: Level 2 + Camera randomization

Scene Modes:
- Mode 0: Manual geometry
- Mode 1: USD Table + Manual environment
- Mode 2: USD Scene (Kujiale) + USD Table
- Mode 3: Full USD (Scene + Table + Desktop objects)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import tyro
from loguru import logger as log
from rich.logging import RichHandler

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.render import RenderCfg


@dataclass
class Args:
    render: RenderCfg = field(default_factory=RenderCfg)
    """Renderer options"""
    task: str = "pick_butter"
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "isaacsim", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"
    """Simulator backend"""
    demo_start_idx: int | None = None
    """The index of the first demo to collect, None for all demos"""
    num_demo_success: int | None = None
    """Target number of successful demos to collect"""
    retry_num: int = 0
    """Number of retries for a failed demo"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    tot_steps_after_success: int = 20
    """Maximum number of steps to collect after success, or until run out of demo"""
    split: Literal["train", "val", "test", "all"] = "all"
    """Split to collect"""
    cust_name: str | None = None
    """Custom name for the dataset"""
    custom_save_dir: str | None = None
    """Custom base path for saving demos. If None, use default structure."""
    scene: str | None = None
    """Scene name"""
    run_all: bool = True
    """Rollout all trajectories, overwrite existing demos"""
    run_unfinished: bool = False
    """Rollout unfinished trajectories"""
    run_failed: bool = False
    """Rollout unfinished and failed trajectories"""
    renderer: Literal["isaaclab", "mujoco", "isaacgym", "genesis", "pybullet", "sapien2", "sapien3"] = "mujoco"

    # Domain randomization options
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"""
    scene_mode: Literal[0, 1, 2, 3] = 0
    """Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    def __post_init__(self):
        assert self.run_all or self.run_unfinished or self.run_failed, (
            "At least one of run_all, run_unfinished, or run_failed must be True"
        )
        if self.num_demo_success is None:
            self.num_demo_success = 100
        if self.demo_start_idx is None:
            self.demo_start_idx = 0

        log.info(f"Args: {self}")

        # Log randomization settings
        if self.level > 0:
            mode_names = {0: "Manual", 1: "USD Table", 2: "USD Scene", 3: "Full USD"}
            log.info("=" * 60)
            log.info("DOMAIN RANDOMIZATION CONFIGURATION")
            log.info(f"  Level: {self.level}")
            log.info(f"  Scene Mode: {self.scene_mode} ({mode_names[self.scene_mode]})")
            log.info("  Randomization:")
            log.info("    Level 1+: Scene + Material")
            log.info("    Level 2+: + Lighting")
            log.info("    Level 3+: + Camera")
            log.info(f"  Seed: {self.randomization_seed if self.randomization_seed else 'Random'}")
            log.info("=" * 60)


args = tyro.cli(Args)

import multiprocessing as mp
import os

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import rootutils
import torch
from tqdm.rich import tqdm_rich as tqdm

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DiskLightCfg, SphereLightCfg
from metasim.scenario.robot import RobotCfg
from metasim.sim import BaseSimHandler
from metasim.task.registry import get_task_class
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot
from metasim.utils.state import state_tensor_to_nested
from metasim.utils.tensor_util import tensor_to_cpu

rootutils.setup_root(__file__, pythonpath=True)

# Import randomization components
try:
    from metasim.randomization import (
        # Randomizers
        CameraRandomizer,
        # Scene Configuration
        EnvironmentLayerCfg,
        # Light configuration
        LightColorRandomCfg,
        LightIntensityRandomCfg,
        LightOrientationRandomCfg,
        LightPositionRandomCfg,
        LightRandomCfg,
        LightRandomizer,
        ManualGeometryCfg,
        MaterialPresets,
        MaterialRandomizer,
        # Core (usually transparent)
        ObjectsLayerCfg,
        SceneRandomCfg,
        SceneRandomizer,
        USDAssetPoolCfg,
        WorkspaceLayerCfg,
    )
    from metasim.randomization.presets.scene_presets import ScenePresets, SceneUSDCollections

    RANDOMIZATION_AVAILABLE = True
except ImportError as e:
    log.warning(f"Randomization components not available: {e}")
    RANDOMIZATION_AVAILABLE = False


class DomainRandomizationManager:
    """Manages domain randomization for demo collection.

    Architecture:
    - Static Objects: Handler-managed (Robot, task objects, Camera, Light)
    - Dynamic Objects: SceneRandomizer-managed (Floor, Table, Distractors)
    - Level system: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera
    - Mode system: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
    """

    def __init__(self, args: Args, scenario, handler, init_states: list):
        self.args = args
        self.scenario = scenario
        self.handler = handler
        self.init_states = init_states
        self.randomizers = {}

        # Store original camera positions BEFORE any randomization
        self.original_camera_positions = {}
        for camera in self.handler.cameras:
            self.original_camera_positions[camera.name] = {
                "pos": tuple(camera.pos),
                "look_at": tuple(camera.look_at),
            }

        # Store original positions for ALL demos (before any modification)
        # This ensures we always have the true original positions from the trajectory file
        self.original_positions = {}
        for demo_idx, init_state in enumerate(init_states):
            demo_key = f"demo_{demo_idx}"
            self.original_positions[demo_key] = {}

            if "objects" in init_state:
                for obj_name, obj_state in init_state["objects"].items():
                    self.original_positions[demo_key][f"obj_{obj_name}"] = {
                        "x": float(obj_state["pos"][0]),
                        "y": float(obj_state["pos"][1]),
                        "z": float(obj_state["pos"][2]),
                    }

            if "robots" in init_state:
                for robot_name, robot_state in init_state["robots"].items():
                    self.original_positions[demo_key][f"robot_{robot_name}"] = {
                        "x": float(robot_state["pos"][0]),
                        "y": float(robot_state["pos"][1]),
                        "z": float(robot_state["pos"][2]),
                    }

        # Early validation
        if not self._validate_setup():
            return

        log.info("=" * 50)
        log.info("DOMAIN RANDOMIZATION SETUP: Initializing randomizers")
        self._setup_randomizers()
        log.info(f"Setup complete: Randomizers ready (Level {args.level}, Mode {args.scene_mode})")

    def _validate_setup(self) -> bool:
        """Validate if randomization can be set up."""
        if self.args.level == 0:
            log.info("Domain randomization disabled (level=0)")
            return False

        if not RANDOMIZATION_AVAILABLE:
            log.warning("Domain randomization requested but components not available")
            return False

        return True

    def _setup_randomizers(self):
        """Initialize all randomizers based on level and mode."""
        seed = self.args.randomization_seed
        self._setup_reproducibility(seed)

        self.randomizers = {
            "scene": None,
            "material_dynamic": [],
            "light": [],
            "camera": [],
        }

        # Scene Randomizer
        self._setup_scene_randomizer(seed)

        # Material Randomization
        self._setup_material_randomizers(seed)

        # Light Randomization (Level 2+)
        if self.args.level >= 2:
            self._setup_light_randomizers(seed)

        # Camera Randomization (Level 3+)
        if self.args.level >= 3:
            self._setup_camera_randomizers(seed)

    def _setup_reproducibility(self, seed: int | None):
        """Setup global reproducibility if seed is provided."""
        if seed is not None:
            log.info(f"Setting up reproducible randomization with seed: {seed}")
            torch.manual_seed(seed)
            import random

            import numpy as np

            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def _setup_scene_randomizer(self, seed: int | None):
        """Setup SceneRandomizer based on scene_mode."""
        mode = self.args.scene_mode
        level = self.args.level

        log.info(f"\nScene Creation (Mode {mode})")
        log.info("-" * 50)

        # Environment Layer
        if mode >= 2:
            # USD Scene
            scene_paths, scene_configs = SceneUSDCollections.kujiale_scenes(return_configs=True)
            log.info(f"Environment: Kujiale USD ({len(scene_paths)} scenes)")
            env_element = USDAssetPoolCfg(
                name="kujiale_scene",
                usd_paths=scene_paths,
                per_path_overrides=scene_configs,
                selection_strategy="random" if level >= 1 else "sequential",
            )
            environment_layer = EnvironmentLayerCfg(elements=[env_element])
        else:
            # Manual Scene
            log.info("Environment: Manual geometry (10m x 10m x 5m)")
            base_cfg = ScenePresets.empty_room(room_size=10.0, wall_height=5.0)
            environment_layer = base_cfg.environment_layer

        # Workspace Layer
        if mode >= 1:
            # USD Table
            table_paths, table_configs = SceneUSDCollections.table785(return_configs=True)
            log.info(f"Workspace: Table785 USD ({len(table_paths)} tables)")
            workspace_element = USDAssetPoolCfg(
                name="table",
                usd_paths=table_paths,
                per_path_overrides=table_configs,
                selection_strategy="random" if level >= 1 else "sequential",
                add_collision=True,
            )
            workspace_layer = WorkspaceLayerCfg(elements=[workspace_element])
        else:
            # Manual Table
            log.info("Workspace: Manual table (Plywood default, randomized in level 1+)")
            workspace_layer = WorkspaceLayerCfg(
                elements=[
                    ManualGeometryCfg(
                        name="table",
                        geometry_type="cube",
                        size=(1.8, 1.8, 0.1),
                        position=(0.0, 0.0, 0.7 - 0.05),
                        default_material="roboverse_data/materials/arnold/Wood/Plywood.mdl",
                        add_collision=True,
                    )
                ]
            )

        # Objects Layer
        if mode >= 3:
            object_paths, object_configs = SceneUSDCollections.desktop_supplies(return_configs=True)
            log.info(f"Objects: Desktop supplies ({len(object_paths)} items, placing 3)")
            objects_layer = ObjectsLayerCfg(
                elements=[
                    USDAssetPoolCfg(
                        name=f"desktop_object_{i + 1}",
                        usd_paths=object_paths,
                        per_path_overrides=object_configs,
                        selection_strategy="random" if level >= 1 else "sequential",
                        add_collision=True,
                    )
                    for i in range(3)
                ]
            )
        else:
            objects_layer = None

        # Create SceneRandomizer
        scene_cfg = SceneRandomCfg(
            environment_layer=environment_layer,
            workspace_layer=workspace_layer,
            objects_layer=objects_layer,
        )

        scene_rand = SceneRandomizer(scene_cfg, seed=seed)
        scene_rand.bind_handler(self.handler)
        self.randomizers["scene"] = scene_rand
        log.info("SceneRandomizer created")

    def _setup_material_randomizers(self, seed: int | None):
        """Setup material randomizers for dynamic objects (environment)."""
        mode = self.args.scene_mode
        level = self.args.level

        if level == 0:
            return

        log.info("\nMaterial Randomization")
        log.info("-" * 50)

        # Dynamic Objects (Manual geometry only)
        if mode == 0:
            # Manual table
            table_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("table", family=("wood", "metal")),
                seed=seed + 2 if seed is not None else None,
            )
            table_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(table_mat)
            log.info("  Dynamic Object: table (Manual)")

        # Manual environment (mode < 2 and level >= 1)
        if mode < 2 and level >= 1:
            # Floor
            floor_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("floor", family=("carpet", "wood", "stone")),
                seed=seed + 101 if seed is not None else None,
            )
            floor_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(floor_mat)

            # Walls
            wall_seed = seed + 102 if seed is not None else None
            for wall_name in ["wall_front", "wall_back", "wall_left", "wall_right"]:
                wall_mat = MaterialRandomizer(
                    MaterialPresets.mdl_family_object(wall_name, family=("masonry", "architecture")),
                    seed=wall_seed,
                )
                wall_mat.bind_handler(self.handler)
                self.randomizers["material_dynamic"].append(wall_mat)

            # Ceiling
            ceiling_mat = MaterialRandomizer(
                MaterialPresets.mdl_family_object("ceiling", family=("architecture", "wall_board")),
                seed=seed + 103 if seed is not None else None,
            )
            ceiling_mat.bind_handler(self.handler)
            self.randomizers["material_dynamic"].append(ceiling_mat)

            log.info("  Dynamic Objects: floor + 4 walls + ceiling")

    def _setup_light_randomizers(self, seed: int | None):
        """Setup light randomizers (Level 2+)."""
        from metasim.scenario.lights import DiskLightCfg, DomeLightCfg, SphereLightCfg

        log.info("\nLight Randomization")
        log.info("-" * 50)

        lights = getattr(self.scenario, "lights", [])
        if not lights:
            log.info("  No lights found")
            return

        # Determine intensity ranges based on render mode
        if hasattr(self.args.render, "mode") and self.args.render.mode == "pathtracing":
            main_range = (22000.0, 40000.0)
            corner_range = (10000.0, 18000.0)
        else:
            main_range = (16000.0, 30000.0)
            corner_range = (6000.0, 12000.0)

        for i, light in enumerate(lights):
            light_name = getattr(light, "name", f"light_{i}")

            if isinstance(light, DiskLightCfg):
                # Main ceiling light with orientation
                light_rand = LightRandomizer(
                    LightRandomCfg(
                        light_name=light_name,
                        intensity=LightIntensityRandomCfg(intensity_range=main_range, enabled=True),
                        color=LightColorRandomCfg(
                            temperature_range=(3000.0, 6000.0), use_temperature=True, enabled=True
                        ),
                        orientation=LightOrientationRandomCfg(
                            angle_range=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
                            relative_to_origin=True,
                            distribution="uniform",
                            enabled=True,
                        ),
                    ),
                    seed=seed + 4 + i if seed is not None else None,
                )
            elif isinstance(light, SphereLightCfg):
                # Corner lights with position and color
                light_rand = LightRandomizer(
                    LightRandomCfg(
                        light_name=light_name,
                        intensity=LightIntensityRandomCfg(intensity_range=corner_range, enabled=True),
                        color=LightColorRandomCfg(
                            temperature_range=(2700.0, 5500.0), use_temperature=True, enabled=True
                        ),
                        position=LightPositionRandomCfg(
                            position_range=((-0.5, 0.5), (-0.5, 0.5), (-0.3, 0.3)),
                            relative_to_origin=True,
                            distribution="uniform",
                            enabled=True,
                        ),
                    ),
                    seed=seed + 5 + i if seed is not None else None,
                )
            elif isinstance(light, DomeLightCfg):
                # Dome light (ambient)
                from metasim.randomization.presets.light_presets import LightPresets

                config = LightPresets.dome_ambient(light_name)
                light_rand = LightRandomizer(config, seed=seed + 4 + i if seed else None)
            else:
                log.warning(f"  Unknown light type: {light_name}")
                continue

            light_rand.bind_handler(self.handler)
            self.randomizers["light"].append(light_rand)
            log.info(f"  Light: {light_name}")

    def _setup_camera_randomizers(self, seed: int | None):
        """Setup camera randomizers (Level 3+)."""
        from metasim.randomization import (
            CameraIntrinsicsRandomCfg,
            CameraPositionRandomCfg,
            CameraRandomCfg,
        )
        from metasim.randomization.presets.camera_presets import CameraProperties

        log.info("\nCamera Randomization")
        log.info("-" * 50)

        cameras = getattr(self.scenario, "cameras", [])
        if not cameras:
            log.info("  No cameras found")
            return

        for camera in cameras:
            camera_name = getattr(camera, "name", "camera")

            # Orbit camera configuration (no roll to keep camera horizontal)
            # Z range kept positive to ensure camera stays above table (no upward view)
            cam_config = CameraRandomCfg(
                camera_name=camera_name,
                position=CameraPositionRandomCfg(
                    delta_range=((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.2)),  # Z: only upward movement
                    use_delta=True,
                    distribution="uniform",
                    enabled=True,
                ),
                intrinsics=CameraIntrinsicsRandomCfg(
                    fov_range=CameraProperties.FOV_NORMAL,
                    use_fov=True,
                    distribution="uniform",
                    enabled=True,
                ),
            )

            cam_rand = CameraRandomizer(cam_config, seed=seed + 10 if seed is not None else None)
            cam_rand.bind_handler(self.handler)
            self.randomizers["camera"].append(cam_rand)
            log.info(f"  Camera: {camera_name}")

    def apply_randomization(self, demo_idx: int, is_initial: bool = False):
        """Apply randomization with global deferred visual flush.

        Args:
            demo_idx: Demo index for logging
            is_initial: Whether this is the initial call (always creates scene)
        """
        if self.args.level == 0 or not self.randomizers:
            return

        log.info("=" * 50)
        log.info(f"DOMAIN RANDOMIZATION: Demo {demo_idx}")

        # Enable global defer flag
        if self.handler:
            self.handler._defer_all_visual_flushes = True

        try:
            # Scene creation/switching
            if self.randomizers["scene"]:
                if is_initial or self.args.level >= 1:
                    scene_rand = self.randomizers["scene"]
                    original_auto_flush = scene_rand.cfg.auto_flush_visuals
                    scene_rand.cfg.auto_flush_visuals = False
                    scene_rand()
                    scene_rand.cfg.auto_flush_visuals = original_auto_flush
                    log.info("  Applied SceneRandomizer")

            # Level 1+: Material randomization (environment only)
            if self.args.level >= 1:
                for mat_rand in self.randomizers["material_dynamic"]:
                    mat_rand()
                if self.randomizers["material_dynamic"]:
                    log.info(f"  Applied MaterialRandomizers ({len(self.randomizers['material_dynamic'])})")

            # Level 2+: Lighting
            if self.args.level >= 2:
                for light_rand in self.randomizers["light"]:
                    light_rand()
                if self.randomizers["light"]:
                    log.info(f"  Applied LightRandomizers ({len(self.randomizers['light'])})")

            # Level 3+: Camera
            if self.args.level >= 3:
                for cam_rand in self.randomizers["camera"]:
                    cam_rand()
                if self.randomizers["camera"]:
                    log.info(f"  Applied CameraRandomizers ({len(self.randomizers['camera'])})")

        finally:
            # Disable global defer and flush once
            if self.handler:
                self.handler._defer_all_visual_flushes = False
                if hasattr(self.handler, "flush_visual_updates"):
                    try:
                        self.handler.flush_visual_updates(wait_for_materials=True, settle_passes=2)
                    except Exception as e:
                        log.debug(f"Failed to flush visual updates: {e}")

    def update_camera_look_at(self, env_id: int = 0):
        """Update camera position and look_at to focus on table after scene switch.

        Adjusts both camera position and look-at point to maintain the same relative
        viewing angle but account for the table's height.
        """
        if not self.randomizers.get("scene"):
            return

        table_bounds = self.randomizers["scene"].get_table_bounds(env_id=env_id)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        table_height = table_bounds["height"]

        # Use original camera positions stored in __init__ (before any randomization)
        clearance = 0.05
        target_look_at_z = table_height + clearance

        for camera in self.handler.cameras:
            orig = self.original_camera_positions[camera.name]
            orig_look_at_z = orig["look_at"][2]
            orig_pos_z = orig["pos"][2]

            # Compute Z offset needed
            z_offset = target_look_at_z - orig_look_at_z

            # Apply same offset to both position and look_at
            camera.pos = (orig["pos"][0], orig["pos"][1], orig_pos_z + z_offset)
            camera.look_at = (orig["look_at"][0], orig["look_at"][1], target_look_at_z)

        if hasattr(self.handler, "_update_camera_pose"):
            self.handler._update_camera_pose()

    def update_positions_to_table(self, demo_idx: int, env_id: int = 0):
        """Update object positions to align with current table after scene switch.

        Maintains relative positions of all objects and robots (rigid body translation).
        The entire system is translated such that the original ground level aligns with the table surface.
        """
        if not self.randomizers.get("scene"):
            return

        # Get current state
        if demo_idx >= len(self.init_states):
            return

        init_state = self.init_states[demo_idx]

        # Get this demo's original positions (stored in __init__)
        demo_key = f"demo_{demo_idx}"
        if demo_key not in self.original_positions:
            log.warning(f"No original positions found for demo {demo_idx}")
            return

        demo_original_positions = self.original_positions[demo_key]

        # Get table bounds
        table_bounds = self.randomizers["scene"].get_table_bounds(env_id=env_id)
        if not table_bounds or abs(table_bounds.get("height", 0)) > 100:
            return

        table_height = table_bounds["height"]
        table_center_x = (table_bounds["x_min"] + table_bounds["x_max"]) / 2
        table_center_y = (table_bounds["y_min"] + table_bounds["y_max"]) / 2

        # Compute system center (XY)
        all_x = [demo_original_positions[k]["x"] for k in demo_original_positions]
        all_y = [demo_original_positions[k]["y"] for k in demo_original_positions]
        system_center_x = sum(all_x) / len(all_x)
        system_center_y = sum(all_y) / len(all_y)

        # Compute XY offset (to center system on table)
        offset_x = table_center_x - system_center_x
        offset_y = table_center_y - system_center_y

        # Compute Z offset: move the reference plane from ground to table
        # Find the ground level (minimum Z in original trajectory)
        all_z = [demo_original_positions[k]["z"] for k in demo_original_positions]
        ground_level = min(all_z)

        # The reference plane (ground) moves to table surface
        # All objects and robots maintain their relative height from this plane
        z_offset = table_height - ground_level

        # Apply same offset to everything (rigid body translation)
        for obj_name, obj_state in init_state["objects"].items():
            orig = demo_original_positions[f"obj_{obj_name}"]
            obj_state["pos"][0] = orig["x"] + offset_x
            obj_state["pos"][1] = orig["y"] + offset_y
            obj_state["pos"][2] = orig["z"] + z_offset

        for robot_name, robot_state in init_state["robots"].items():
            orig = demo_original_positions[f"robot_{robot_name}"]
            robot_state["pos"][0] = orig["x"] + offset_x
            robot_state["pos"][1] = orig["y"] + offset_y
            robot_state["pos"][2] = orig["z"] + z_offset


def get_actions(all_actions, env, demo_idxs: list[int], robot: RobotCfg):
    action_idxs = env._episode_steps

    actions = []
    for env_id, (demo_idx, action_idx) in enumerate(zip(demo_idxs, action_idxs)):
        if action_idx < len(all_actions[demo_idx]):
            action = all_actions[demo_idx][action_idx]
        else:
            action = all_actions[demo_idx][-1]

        actions.append(action)

    return actions


def get_run_out(all_actions, env, demo_idxs: list[int]) -> list[bool]:
    action_idxs = env._episode_steps
    run_out = [action_idx >= len(all_actions[demo_idx]) for demo_idx, action_idx in zip(demo_idxs, action_idxs)]
    return run_out


def save_demo_mp(save_req_queue: mp.Queue, robot_cfg: RobotCfg, task_desc: str):
    from metasim.utils.save_util import save_demo

    while (save_request := save_req_queue.get()) is not None:
        demo = save_request["demo"]
        save_dir = save_request["save_dir"]
        log.info(f"Received save request, saving to {save_dir}")
        save_demo(save_dir, demo, robot_cfg=robot_cfg, task_desc=task_desc)


def ensure_clean_state(handler, expected_state=None):
    """Ensure environment is in clean initial state with intelligent validation."""
    prev_state = None
    stable_count = 0
    max_steps = 10
    min_steps = 2

    for step in range(max_steps):
        handler.simulate()
        current_state = handler.get_states()

        if step >= min_steps:
            if prev_state is not None:
                is_stable = True
                if hasattr(current_state, "objects") and hasattr(prev_state, "objects"):
                    for obj_name, obj_state in current_state.objects.items():
                        if obj_name in prev_state.objects:
                            curr_dof = getattr(obj_state, "dof_pos", None)
                            prev_dof = getattr(prev_state.objects[obj_name], "dof_pos", None)
                            if curr_dof is not None and prev_dof is not None:
                                if not torch.allclose(curr_dof, prev_dof, atol=1e-5):
                                    is_stable = False
                                    break

                if is_stable and expected_state is not None:
                    is_correct_state = _validate_state_correctness(current_state, expected_state)
                    if not is_correct_state:
                        log.debug(f"State stable but incorrect at step {step}, continuing simulation...")
                        stable_count = 0
                        is_stable = False

                if is_stable:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0

            prev_state = current_state

    if expected_state is not None:
        final_state = handler.get_states()
        is_final_correct = _validate_state_correctness(final_state, expected_state)
        if not is_final_correct:
            log.warning(f"State validation failed after {max_steps} steps - reset may not have taken full effect")

    handler.get_states()


def _validate_state_correctness(current_state, expected_state):
    """Validate that current state matches expected initial state for critical objects."""
    if not hasattr(current_state, "objects") or not hasattr(expected_state, "objects"):
        return True

    critical_objects = []
    for obj_name, expected_obj in expected_state.objects.items():
        if hasattr(expected_obj, "dof_pos") and getattr(expected_obj, "dof_pos", None) is not None:
            critical_objects.append(obj_name)

    if not critical_objects:
        return True

    tolerance = 5e-3

    for obj_name in critical_objects:
        if obj_name not in current_state.objects:
            continue

        expected_obj = expected_state.objects[obj_name]
        current_obj = current_state.objects[obj_name]

        expected_dof = getattr(expected_obj, "dof_pos", None)
        current_dof = getattr(current_obj, "dof_pos", None)

        if expected_dof is not None and current_dof is not None:
            if not torch.allclose(current_dof, expected_dof, atol=tolerance):
                diff = torch.abs(current_dof - expected_dof).max().item()
                log.debug(f"DOF mismatch for {obj_name}: max diff = {diff:.6f} (tolerance = {tolerance})")
                return False

    return True


def force_reset_to_state(env, state, env_id):
    """Force reset environment to specific state with validation."""
    env.reset(states=[state], env_ids=[env_id])
    ensure_clean_state(env.handler, expected_state=state)
    if hasattr(env, "_episode_steps"):
        env._episode_steps[env_id] = 0


global global_step, tot_success, tot_give_up
tot_success = 0
tot_give_up = 0
global_step = 0


class DemoCollector:
    def __init__(self, handler, robot_cfg, task_desc="", demo_start_idx=0):
        assert isinstance(handler, BaseSimHandler)
        self.handler = handler
        self.robot_cfg = robot_cfg
        self.task_desc = task_desc
        self.cache: dict[int, list[dict]] = {}
        self.save_request_queue = mp.Queue()
        self.save_proc = mp.Process(target=save_demo_mp, args=(self.save_request_queue, robot_cfg, task_desc))
        self.save_proc.start()

        TaskName = args.task
        if args.custom_save_dir:
            self.base_save_dir = args.custom_save_dir
        else:
            additional_str = f"-{args.cust_name}" if args.cust_name else ""
            self.base_save_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    def _get_max_demo_index(self, status: str) -> int:
        status_dir = os.path.join(self.base_save_dir, status)
        if not os.path.exists(status_dir):
            return 0

        max_idx = -1
        for item in os.listdir(status_dir):
            if item.startswith("demo_") and os.path.isdir(os.path.join(status_dir, item)):
                try:
                    idx = int(item.split("_")[1])
                    max_idx = max(max_idx, idx)
                except (ValueError, IndexError):
                    continue

        return max_idx + 1

    def create(self, demo_idx: int, data_dict: dict):
        assert demo_idx not in self.cache
        assert isinstance(demo_idx, int)
        self.cache[demo_idx] = [data_dict]

    def add(self, demo_idx: int, data_dict: dict):
        if data_dict is None:
            log.warning("Skipping adding obs to DemoCollector because obs is None")
        assert demo_idx in self.cache
        self.cache[demo_idx].append(deepcopy(tensor_to_cpu(data_dict)))

    def save(self, demo_idx: int, status: str):
        assert demo_idx in self.cache
        assert status in ["success", "failed"], f"Invalid status: {status}"

        continuous_idx = demo_idx

        save_dir = os.path.join(self.base_save_dir, status, f"demo_{continuous_idx:04d}")
        if os.path.exists(os.path.join(save_dir, "status.txt")):
            os.remove(os.path.join(save_dir, "status.txt"))

        os.makedirs(save_dir, exist_ok=True)
        log.info(f"Saving demo {demo_idx} as {continuous_idx:04d} to {save_dir}")

        from metasim.utils.save_util import save_demo

        save_demo(save_dir, self.cache[demo_idx], self.robot_cfg, self.task_desc)

        if status == "failed":
            with open(os.path.join(save_dir, "status.txt"), "w") as f:
                f.write(status)

    def delete(self, demo_idx: int):
        assert demo_idx in self.cache
        del self.cache[demo_idx]

    def final(self):
        self.save_request_queue.put(None)  # signal to save_demo_mp to exit
        self.save_proc.join()
        assert self.cache == {}


def should_skip(log_dir: str, demo_idx: int):
    demo_name = f"demo_{demo_idx:04d}"
    success_path = os.path.join(log_dir, "success", demo_name, "status.txt")
    failed_path = os.path.join(log_dir, "failed", demo_name, "status.txt")

    if args.run_unfinished:
        if not os.path.exists(success_path) and not os.path.exists(failed_path):
            return False
        return True

    if args.run_all:
        return False

    if args.run_failed:
        if os.path.exists(success_path):
            return is_status_success(log_dir, demo_idx)
        return False

    return True


def is_status_success(log_dir: str, demo_idx: int) -> bool:
    demo_name = f"demo_{demo_idx:04d}"
    status_path = os.path.join(log_dir, "success", demo_name, "status.txt")

    if os.path.exists(status_path):
        return open(status_path).read().strip() == "success"
    return False


class DemoIndexer:
    def __init__(self, save_root_dir: str, start_idx: int, end_idx: int, pbar: tqdm):
        self.save_root_dir = save_root_dir
        self._next_idx = start_idx
        self.end_idx = end_idx
        self.pbar = pbar
        self._skip_if_should()

    @property
    def next_idx(self):
        return self._next_idx

    def _skip_if_should(self):
        while should_skip(self.save_root_dir, self._next_idx):
            global global_step, tot_success, tot_give_up
            if is_status_success(self.save_root_dir, self._next_idx):
                tot_success += 1
            else:
                tot_give_up += 1
            self.pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
            self.pbar.update(1)
            log.info(f"Demo {self._next_idx} already exists, skipping...")
            self._next_idx += 1

    def move_on(self):
        self._next_idx += 1
        self._skip_if_should()


def main():
    global global_step, tot_success, tot_give_up
    task_cls = get_task_class(args.task)

    if args.task in {"stack_cube", "pick_cube", "pick_butter"}:
        dp_camera = True
    else:
        dp_camera = args.task != "close_box"

    is_libero_dataset = "libero_90" in args.task

    if is_libero_dataset:
        dp_pos = (2.0, 0.0, 2)
    elif dp_camera:
        dp_pos = (1.0, 0.0, 0.75)
    else:
        dp_pos = (1.5, 0.0, 1.5)

    camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=dp_pos, look_at=(0.0, 0.0, 0.0))

    # Lighting setup
    if args.render.mode == "pathtracing":
        ceiling_main = 18000.0
        ceiling_corners = 8000.0
    else:
        ceiling_main = 12000.0
        ceiling_corners = 5000.0

    lights = [
        DiskLightCfg(
            name="ceiling_main",
            intensity=ceiling_main,
            color=(1.0, 1.0, 1.0),
            radius=1.2,
            pos=(0.0, 0.0, 2.8),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        SphereLightCfg(
            name="ceiling_ne", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_nw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, 1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_sw", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(-1.0, -1.0, 2.5)
        ),
        SphereLightCfg(
            name="ceiling_se", intensity=ceiling_corners, color=(1.0, 1.0, 1.0), radius=0.6, pos=(1.0, -1.0, 2.5)
        ),
    ]

    scenario = task_cls.scenario.update(
        robots=[args.robot],
        scene=args.scene,
        cameras=[camera],
        lights=lights,
        render=args.render,
        simulator=args.sim,
        renderer=args.renderer,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    robot = get_robot(args.robot)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = task_cls(scenario, device=device)

    ## Data
    assert os.path.exists(env.traj_filepath), f"Trajectory file does not exist: {env.traj_filepath}"
    init_states, all_actions, all_states = get_traj(env.traj_filepath, robot, env.handler)

    # Initialize domain randomization manager
    randomization_manager = DomainRandomizationManager(args, scenario, env.handler, init_states)

    tot_demo = len(all_actions)
    if args.split == "train":
        init_states = init_states[: int(tot_demo * 0.9)]
        all_actions = all_actions[: int(tot_demo * 0.9)]
        all_states = all_states[: int(tot_demo * 0.9)]
    elif args.split == "val" or args.split == "test":
        init_states = init_states[int(tot_demo * 0.9) :]
        all_actions = all_actions[int(tot_demo * 0.9) :]
        all_states = all_states[int(tot_demo * 0.9) :]

    n_demo = len(all_actions)
    log.info(f"Collecting from {args.split} split, {n_demo} out of {tot_demo} demos")

    ########################################################
    ## Main
    ########################################################
    max_demo = n_demo
    try_num = args.retry_num + 1

    ## Demo collection state machine:
    ## CollectingDemo -> Success -> FinalizeDemo -> NextDemo
    ## CollectingDemo -> Timeout -> Retry/GiveUp -> NextDemo

    ## Setup
    task_desc = getattr(env, "task_desc", "")
    collector = DemoCollector(env.handler, robot, task_desc)
    pbar = tqdm(total=args.num_demo_success, desc="Collecting successful demos")

    ## State variables
    failure_count = [0] * env.handler.num_envs
    steps_after_success = [0] * env.handler.num_envs
    finished = [False] * env.handler.num_envs
    TaskName = args.task

    if args.cust_name is not None:
        additional_str = f"-{args.cust_name}"
    else:
        additional_str = ""

    if args.custom_save_dir:
        save_root_dir = args.custom_save_dir
    else:
        save_root_dir = f"roboverse_demo/demo_{args.sim}/{TaskName}{additional_str}/robot-{args.robot}"

    demo_indexer = DemoIndexer(
        save_root_dir=save_root_dir,
        start_idx=args.demo_start_idx,
        end_idx=max_demo,
        pbar=pbar,
    )
    demo_idxs = []
    for demo_idx in range(env.handler.num_envs):
        demo_idxs.append(demo_indexer.next_idx)
        demo_indexer.move_on()
    log.info(f"Initialize with demo idxs: {demo_idxs}")

    ## Apply initial randomization (create scene and update positions)
    for env_id, demo_idx in enumerate(demo_idxs):
        randomization_manager.apply_randomization(demo_idx, is_initial=True)
        randomization_manager.update_positions_to_table(demo_idx, env_id)
        randomization_manager.update_camera_look_at(env_id)

    ## Reset to initial states (after position adjustment)
    obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])

    ## Wait for environment to stabilize after reset
    ensure_clean_state(env.handler)

    ## Reset episode step counters after stabilization
    if hasattr(env, "_episode_steps"):
        for env_id in range(env.handler.num_envs):
            env._episode_steps[env_id] = 0

    ## Record the clean, stabilized initial state
    obs = env.handler.get_states()
    obs = state_tensor_to_nested(env.handler, obs)

    for env_id, demo_idx in enumerate(demo_idxs):
        log.info(f"Starting Demo {demo_idx} in Env {env_id}")
        collector.create(demo_idx, obs[env_id])

    ## Main Loop
    stop_flag = False

    while not all(finished):
        if tot_success >= args.num_demo_success:
            log.info(f"Reached target number of successful demos ({args.num_demo_success}). Stopping collection.")
            break

        if demo_indexer.next_idx >= max_demo:
            log.warning(f"Reached maximum demo index ({max_demo}). Stopping collection.")
            break

        pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")
        actions = get_actions(all_actions, env, demo_idxs, robot)
        obs, reward, success, time_out, extras = env.step(actions)
        obs = state_tensor_to_nested(env.handler, obs)
        run_out = get_run_out(all_actions, env, demo_idxs)

        for env_id in range(env.handler.num_envs):
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            collector.add(demo_idx, obs[env_id])

        for env_id in success.nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            if steps_after_success[env_id] == 0:
                log.info(f"Demo {demo_idx} in Env {env_id} succeeded!")
                tot_success += 1
                pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

            if not run_out[env_id] and steps_after_success[env_id] < args.tot_steps_after_success:
                steps_after_success[env_id] += 1
            else:
                steps_after_success[env_id] = 0
                collector.save(demo_idx, status="success")
                collector.delete(demo_idx)

                if (not stop_flag) and (demo_indexer.next_idx < max_demo):
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    log.info(f"Transitioning Env {env_id}: Demo {demo_idx} to Demo {new_demo_idx}")

                    randomization_manager.apply_randomization(new_demo_idx, is_initial=False)
                    randomization_manager.update_positions_to_table(new_demo_idx, env_id)
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                    run_out[env_id] = False
                else:
                    finished[env_id] = True

        for env_id in (time_out | torch.tensor(run_out, device=time_out.device)).nonzero().squeeze(-1).tolist():
            if finished[env_id]:
                continue

            demo_idx = demo_idxs[env_id]
            log.info(f"Demo {demo_idx} in Env {env_id} timed out!")
            collector.save(demo_idx, status="failed")
            collector.delete(demo_idx)
            failure_count[env_id] += 1

            if failure_count[env_id] < try_num:
                log.info(f"Demo {demo_idx} failed {failure_count[env_id]} times, retrying...")
                randomization_manager.apply_randomization(demo_idx, is_initial=False)
                randomization_manager.update_positions_to_table(demo_idx, env_id)
                randomization_manager.update_camera_look_at(env_id)
                force_reset_to_state(env, init_states[demo_idx], env_id)

                obs = env.handler.get_states()
                obs = state_tensor_to_nested(env.handler, obs)
                collector.create(demo_idx, obs[env_id])
            else:
                log.error(f"Demo {demo_idx} failed too many times, giving up")
                failure_count[env_id] = 0
                tot_give_up += 1
                # pbar.update(1)
                pbar.set_description(f"Frame {global_step} Success {tot_success} Giveup {tot_give_up}")

                if demo_indexer.next_idx < max_demo:
                    new_demo_idx = demo_indexer.next_idx
                    demo_idxs[env_id] = new_demo_idx
                    randomization_manager.apply_randomization(new_demo_idx, is_initial=False)
                    randomization_manager.update_positions_to_table(new_demo_idx, env_id)
                    randomization_manager.update_camera_look_at(env_id)
                    force_reset_to_state(env, init_states[new_demo_idx], env_id)

                    obs = env.handler.get_states()
                    obs = state_tensor_to_nested(env.handler, obs)
                    collector.create(new_demo_idx, obs[env_id])
                    demo_indexer.move_on()
                else:
                    finished[env_id] = True

        global_step += 1

    log.info("Finalizing")
    collector.final()
    env.close()


if __name__ == "__main__":
    main()
