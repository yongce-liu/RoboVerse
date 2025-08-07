import os

import torch
from loguru import logger as log

from metasim.cfg.lights import BaseLightCfg, CylinderLightCfg, DistantLightCfg
from metasim.cfg.objects import (
    ArticulationObjCfg,
    BaseObjCfg,
    PrimitiveCubeCfg,
    PrimitiveCylinderCfg,
    PrimitiveFrameCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
)
from metasim.cfg.robots import BaseRobotCfg
from metasim.cfg.sensors import BaseCameraCfg, BaseSensorCfg, ContactForceSensorCfg, PinholeCameraCfg
from metasim.utils.math import convert_camera_frame_orientation_convention

try:
    from .empty_env import EmptyEnv
except:
    pass


def _add_object(env: "EmptyEnv", obj: BaseObjCfg) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

    assert isinstance(obj, BaseObjCfg)
    prim_path = f"/World/envs/env_.*/{obj.name}"

    ## Articulation object
    if isinstance(obj, ArticulationObjCfg):
        env.scene.articulations[obj.name] = Articulation(
            ArticulationCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(usd_path=obj.usd_path, scale=obj.scale),
                actuators={},
            )
        )
        return

    if obj.fix_base_link:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True)
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg()
    if obj.collision_enabled:
        collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    else:
        collision_props = None

    ## Primitive object
    if isinstance(obj, PrimitiveCubeCfg):
        env.scene.rigid_objects[obj.name] = RigidObject(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.MeshCuboidCfg(
                    size=obj.size,
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                    ),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                ),
            )
        )
        return
    if isinstance(obj, PrimitiveSphereCfg):
        env.scene.rigid_objects[obj.name] = RigidObject(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.MeshSphereCfg(
                    radius=obj.radius,
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                    ),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                ),
            )
        )
        return
    if isinstance(obj, PrimitiveCylinderCfg):
        env.scene.rigid_objects[obj.name] = RigidObject(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.MeshCylinderCfg(
                    radius=obj.radius,
                    height=obj.height,
                    mass_props=sim_utils.MassPropertiesCfg(mass=obj.mass),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(obj.color[0], obj.color[1], obj.color[2])
                    ),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                ),
            )
        )
        return
    if isinstance(obj, PrimitiveFrameCfg):
        env.scene.rigid_objects[obj.name] = RigidObject(
            RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path="metasim/data/quick_start/assets/COMMON/frame/usd/frame.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),  # fixed
                    collision_props=None,  # no collision
                    scale=obj.scale,
                ),
            )
        )
        return

    ## Rigid object
    if isinstance(obj, RigidObjCfg):
        usd_file_cfg = sim_utils.UsdFileCfg(
            usd_path=obj.usd_path,
            rigid_props=rigid_props,
            collision_props=collision_props,
            scale=obj.scale,
        )
        if isinstance(obj, RigidObjCfg):
            env.scene.rigid_objects[obj.name] = RigidObject(RigidObjectCfg(prim_path=prim_path, spawn=usd_file_cfg))
            return

    raise ValueError(f"Unsupported object type: {type(obj)}")


def add_objects(env: "EmptyEnv", objects: list[BaseObjCfg]) -> None:
    for obj in objects:
        _add_object(env, obj)


def _add_robot(env: "EmptyEnv", robot: BaseRobotCfg) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import Articulation, ArticulationCfg
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=robot.usd_path,
            activate_contact_sensors=True,  # TODO: only activate when contact sensor is added
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=robot.fix_base_link),
        ),
        actuators={
            jn: ImplicitActuatorCfg(
                joint_names_expr=[jn],
                stiffness=actuator.stiffness,
                damping=actuator.damping,
            )
            for jn, actuator in robot.actuators.items()
        },
    )
    cfg.prim_path = f"/World/envs/env_.*/{robot.name}"
    cfg.spawn.usd_path = os.path.abspath(robot.usd_path)
    cfg.spawn.rigid_props.disable_gravity = not robot.enabled_gravity
    cfg.spawn.articulation_props.enabled_self_collisions = robot.enabled_self_collisions
    for joint_name, actuator in robot.actuators.items():
        cfg.actuators[joint_name].velocity_limit = actuator.velocity_limit

    robot_inst = Articulation(cfg)
    env.scene.articulations[robot.name] = robot_inst
    env.robots.append(robot_inst)


def add_robots(env: "EmptyEnv", robots: list[BaseRobotCfg]) -> None:
    env.robots = []
    for robot in robots:
        _add_robot(env, robot)


def _add_light(env: "EmptyEnv", light: BaseLightCfg, prim_path: str) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim.spawners import spawn_light
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners import spawn_light

    if isinstance(light, DistantLightCfg):
        spawn_light(
            prim_path,
            sim_utils.DistantLightCfg(
                intensity=light.intensity,
            ),
            orientation=light.quat,
        )
    elif isinstance(light, CylinderLightCfg):
        spawn_light(
            prim_path,
            sim_utils.CylinderLightCfg(
                intensity=light.intensity,
                length=light.length,
                radius=light.radius,
            ),
            translation=light.pos,
            orientation=light.rot,
        )
    else:
        raise ValueError(f"Unsupported light type: {type(light)}")


def add_lights(env: "EmptyEnv", lights: list[BaseLightCfg]) -> None:
    for i, light in enumerate(lights):
        if light.is_global:
            _add_light(env, light, f"/World/lights/light_{i}")
        else:
            _add_light(env, light, f"/World/envs/env_0/lights/light_{i}")


def _add_contact_force_sensor(env: "EmptyEnv", sensor: ContactForceSensorCfg) -> None:
    try:
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
    except ModuleNotFoundError:
        import isaacsim.core.utils.prims as prim_utils
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

    if isinstance(sensor, ContactForceSensorCfg):
        _base_prim_regex_path = (
            f"/World/envs/env_0/{sensor.base_link}"
            if isinstance(sensor.base_link, str)
            else f"/World/envs/env_0/{sensor.base_link[0]}/.*{sensor.base_link[1]}"  # TODO: improve the regex
        )
        _base_prim_paths = prim_utils.find_matching_prim_paths(_base_prim_regex_path)
        if len(_base_prim_paths) == 0:
            log.error(f"Base link {sensor.base_link} of cotact force sensor not found")
            return
        if len(_base_prim_paths) > 1:
            log.warning(
                f"Multiple base links found for contact force sensor {sensor.name}, using the first one: {_base_prim_paths[0]}"
            )
        base_prim_path = _base_prim_paths[0]
        log.info(f"Base prim path: {base_prim_path}")
        if sensor.source_link is not None:
            _source_prim_regex_path = (
                f"/World/envs/env_0/{sensor.source_link}"
                if isinstance(sensor.source_link, str)
                else f"/World/envs/env_0/{sensor.source_link[0]}/.*{sensor.source_link[1]}"  # TODO: improve the regex
            )
            _source_prim_paths = prim_utils.find_matching_prim_paths(_source_prim_regex_path)
            if len(_source_prim_paths) == 0:
                log.error(f"Source link {sensor.source_link} of cotact force sensor not found")
                return
            if len(_source_prim_paths) > 1:
                log.warning(
                    f"Multiple source links found for contact force sensor {sensor.name}, using the first one: {_source_prim_paths[0]}"
                )
            source_prim_path = _source_prim_paths[0]
        else:
            source_prim_path = None

        env.scene.sensors[sensor.name] = ContactSensor(
            ContactSensorCfg(
                prim_path=base_prim_path.replace("env_0", "env_.*"),  # HACK: this is so hacky
                filter_prim_paths_expr=[source_prim_path.replace("env_0", "env_.*")]  # HACK: this is so hacky
                if source_prim_path is not None
                else [],
                history_length=6,  # XXX: hard-coded
                update_period=0.0,  # XXX: hard-coded
            )
        )


def add_sensors(env: "EmptyEnv", sensors: list[BaseSensorCfg]) -> None:
    for sensor in sensors:
        if isinstance(sensor, ContactForceSensorCfg):
            _add_contact_force_sensor(env, sensor)


def _add_pinhole_camera(env: "EmptyEnv", camera: PinholeCameraCfg) -> None:
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
    except ModuleNotFoundError:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg

    ## See https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#tile-rendered-usd-camera
    data_type_map = {
        "rgb": "rgb",
        "depth": "depth",
        "instance_seg": "instance_segmentation_fast",
        "instance_id_seg": "instance_id_segmentation_fast",
    }
    if camera.mount_to is None:
        prim_path = f"/World/envs/env_.*/{camera.name}"
        offset = TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world")
    else:
        prim_path = f"/World/envs/env_.*/{camera.mount_to}/{camera.mount_link}/{camera.name}"
        offset = TiledCameraCfg.OffsetCfg(pos=camera.mount_pos, rot=camera.mount_quat, convention="world")

    env.scene.sensors[camera.name] = TiledCamera(
        TiledCameraCfg(
            prim_path=prim_path,
            offset=offset,
            data_types=[data_type_map[dt] for dt in camera.data_types],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=camera.focal_length,
                focus_distance=camera.focus_distance,
                horizontal_aperture=camera.horizontal_aperture,
                clipping_range=camera.clipping_range,
            ),
            width=camera.width,
            height=camera.height,
            colorize_instance_segmentation=False,
            colorize_instance_id_segmentation=False,
        )
    )


def add_cameras(env: "EmptyEnv", cameras: list[BaseCameraCfg]) -> None:
    for camera in cameras:
        if isinstance(camera, PinholeCameraCfg):
            _add_pinhole_camera(env, camera)
        else:
            raise ValueError(f"Unsupported camera type: {type(camera)}")


def get_pose(
    env: "EmptyEnv", obj_name: str, obj_subpath: str | None = None, env_ids: list[int] | None = None
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    try:
        from omni.isaac.core.prims import RigidPrimView

        ISAACLAB_VERSION = 1
    except ModuleNotFoundError:
        from isaacsim.core.prims import RigidPrim as RigidPrimView

        ISAACLAB_VERSION = 2

    if env_ids is None:
        env_ids = list(range(env.num_envs))

    if obj_name in env.scene.rigid_objects:
        obj_inst = env.scene.rigid_objects[obj_name]
    elif obj_name in env.scene.articulations:
        obj_inst = env.scene.articulations[obj_name]
    else:
        raise ValueError(f"Object {obj_name} not found")

    if obj_subpath is None:
        pos = obj_inst.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
        rot = obj_inst.data.root_quat_w[env_ids]
    else:
        ## TODO: Following code has bug with IsaacLab2.0 (IsaacSim 4.5)
        if ISAACLAB_VERSION == 1:
            view = RigidPrimView(
                obj_inst.cfg.prim_path + "/" + obj_subpath,
                name=f"{obj_name}_{obj_subpath}_view",
                reset_xform_properties=False,
            )
            pos, rot = view.get_world_poses(indices=env_ids)
            pos = pos - env.scene.env_origins[env_ids]
        else:
            log.warning("IsaacLab2.0 (IsaacSim 4.5) does not support creating RigidPrimView, so we return zeros.")
            pos = torch.zeros((len(env_ids), 3), device=env.device)
            rot = torch.zeros((len(env_ids), 4), device=env.device)

    assert pos.shape == (len(env_ids), 3)
    assert rot.shape == (len(env_ids), 4)
    return pos, rot


def joint_is_implicit_actuator(joint_name: str, obj_inst) -> bool:
    try:
        from omni.isaac.lab.actuators import ImplicitActuatorCfg
        from omni.isaac.lab.assets import Articulation
    except ModuleNotFoundError:
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation

    assert isinstance(obj_inst, Articulation)
    actuators = [actuator for actuator in obj_inst.actuators.values() if joint_name in actuator.joint_names]
    if len(actuators) == 0:
        log.error(f"Joint {joint_name} could not be found in actuators of {obj_inst.cfg.prim_path}")
        return False
    if len(actuators) > 1:
        log.warning(f"Joint {joint_name} is found in multiple actuators of {obj_inst.cfg.prim_path}")
    actuator = actuators[0]
    return isinstance(actuator, ImplicitActuatorCfg)


def _update_tiled_camera_pose(env: "EmptyEnv", cameras: list[BaseCameraCfg]):
    for camera in cameras:
        camera_inst = env.scene.sensors[camera.name]
        pos, quat = camera_inst._view.get_world_poses()
        camera_inst._data.pos_w = pos
        camera_inst._data.quat_w_world = convert_camera_frame_orientation_convention(
            quat, origin="opengl", target="world"
        )
