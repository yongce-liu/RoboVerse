from __future__ import annotations
from typing import Dict

import torch

from metasim.sim.base import BaseSimHandler, BaseQueryType
from metasim.utils import configclass
from metasim.utils.math import quat_apply, convert_quat
import numpy as np
import warnings

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass
class ContactForces(BaseQueryType):
    """Optional query to fetch per-body net contact forces for each robot.

    - For IsaacGym: uses the native net-contact tensor and maps it per-robot in handler indexing order.
    - For IsaacSim: returns a zero tensor fallback per-robot (hook is in place; replace with real source when available).
    """
    def __init__(self):
        super().__init__()

    def bind_handler(self, handler:BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        if self.simulator in ["isaacgym", "mujoco"]:
            self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
        elif self.simulator == "isaacsim":
            sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
            self.body_ids_reindex = torch.tensor([self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names], dtype=torch.int, device=self.handler.device)
        else:
            raise NotImplementedError
        self.initialize()

    def initialize(self):
        if self.simulator == "isaacgym":
            self.contact_forces = isaacgym.gymtorch.wrap_tensor(self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim))
        elif self.simulator == "isaacsim":
            self.contact_forces = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self.contact_forces = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError

    def _get_contact_forces_mujoco(self) -> torch.Tensor:
        """
        Compute net contact forces on each body.
        Returns:
            torch.Tensor: shape (nbody, 3), contact forces for each body
        """
        import mujoco
        nbody = self.handler.physics.model.nbody
        contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

        for i in range(self.handler.physics.data.ncon):
            contact = self.handler.physics.data.contact[i]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
            f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

            body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
            body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

            contact_forces[body1] += f_contact
            contact_forces[body2] -= f_contact

        return contact_forces


    def __call__(self):
        if self.simulator == "isaacgym":
            self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
        elif self.simulator == "isaacsim":
            self.contact_forces = self.handler.contact_sensor.data.net_forces_w
        elif self.simulator == "mujoco":
            self.contact_forces = self._get_contact_forces_mujoco()
        else:
            raise NotImplementedError
        return {self.robots[0].name: self.contact_forces.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]}


class LidarPointCloud(BaseQueryType):
    """Optional query that produces a LiDAR point cloud using LidarSensor + Warp.

    Notes
    - Supports IsaacGym and MuJoCo via common state interface; raycasting is done against a generated terrain mesh.
    - Robot self-geometry is not included in the mesh to keep this query generic and lightweight.
    - Requires packages: LidarSensor, warp, trimesh. If unavailable, returns None payload when enabled.
    - Quaternions: handler states use (w,x,y,z). LidarSensor expects (x,y,z,w). Conversion is handled internally.
    """

    def __init__(
        self,
        link_name: str = "mid360_link",
        sensor_type: str = "mid360",
        apply_optical_center_offset: bool = True,
        optical_center_offset_z: float = 0.03503,
        enabled: bool = False,
    ):
        super().__init__()
        self.link_name = link_name
        self.sensor_type = sensor_type
        self.apply_optical_center_offset = apply_optical_center_offset
        self.optical_center_offset_z = optical_center_offset_z
        self.enabled = enabled

    def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator = handler.scenario.simulator
        self.handler = handler
        self.num_envs = handler.scenario.num_envs
        self.robots = handler.robots
        self.device = str(handler.device) # warp only accepts str device
        self._init_backend()

    def _init_backend(self):
        try:
            import warp as wp  # type: ignore
            import trimesh  # type: ignore
            from LidarSensor.lidar_sensor import LidarSensor  # type: ignore
            from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig  # type: ignore
            from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain  # type: ignore
            from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg  # type: ignore
        except Exception as e:
            warnings.warn(f"LidarPointCloud init failed due to missing deps: {e}")
            self._backend_ready = False
            return

        self.wp = wp
        self.trimesh = trimesh
        self.LidarSensor = LidarSensor
        self.LidarConfig = LidarConfig
        self.Terrain = Terrain
        self.Terrain_cfg = Terrain_cfg

        try:
            self.wp.init()
        except Exception:
            # wp.init() may be already called; ignore
            pass

        # Build terrain mesh once
        self.terrain_cfg = self.Terrain_cfg()
        self.terrain = self.Terrain(self.terrain_cfg, self.num_envs)
        terrain_mesh = self.trimesh.Trimesh(vertices=self.terrain.vertices, faces=self.terrain.triangles)
        # translate so (0,0) aligns to sim origin similar to example
        border = float(getattr(self.terrain_cfg, "border_size", 0.0))
        translation = self.trimesh.transformations.translation_matrix(np.array([-border, -border, 0.0]))
        terrain_mesh.apply_transform(translation)

        vertices = self.handler._ground_mesh_vertices
        triangles = self.handler._ground_mesh_triangles

        import torch

        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        faces_wp_int32_array = self.wp.from_numpy(triangles.reshape(-1), dtype=self.wp.int32, device=self.device)
        vertex_vec3_array = self.wp.from_torch(vertex_tensor, dtype=self.wp.vec3)
        self.wp_mesh = self.wp.Mesh(points=vertex_vec3_array, indices=faces_wp_int32_array)
        self.mesh_ids = self.wp.array([self.wp_mesh.id], dtype=self.wp.uint64, device=self.device)

        # Prepare sensor config and buffers
        self.sensor_cfg = self.LidarConfig()
        self.sensor_cfg.sensor_type = self.sensor_type

        num_envs = self.num_envs
        num_sensors = int(getattr(self.sensor_cfg, "num_sensors", 1))
        v_lines = int(getattr(self.sensor_cfg, "vertical_line_num", 128))
        h_lines = int(getattr(self.sensor_cfg, "horizontal_line_num", 512))

        self.lidar_tensor = torch.zeros((num_envs, num_sensors, v_lines, h_lines, 3), device=self.device)
        self.sensor_dist_tensor = torch.zeros((num_envs, num_sensors, v_lines, h_lines), device=self.device)
        self.sensor_pos_tensor = torch.zeros((num_envs, 3), device=self.device)
        # LidarSensor expects XYZW ordering
        self.sensor_quat_tensor_xyzw = torch.zeros((num_envs, 4), device=self.device)

        self.warp_tensor_dict = {
            "sensor_dist_tensor": self.sensor_dist_tensor,
            "device": str(self.device),
            "num_envs": num_envs,
            "num_sensors": num_sensors,
            "sensor_pos_tensor": self.sensor_pos_tensor,
            "sensor_quat_tensor": self.sensor_quat_tensor_xyzw,
            "mesh_ids": self.mesh_ids,
        }

        self.sensor = self.LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self._backend_ready = True

    def __call__(self):
        if not self.enabled:
            return {self.robots[0].name: None}

        robot_name = self.robots[0].name

        # Obtain the lidar link pose directly from the simulator backend
        # to avoid re-entering handler.get_states() (which calls queries again).
        sim_type = self.simulator
        if sim_type == "isaacgym":
            # Refresh tensors and read rigid body state tensor (xyzw from IsaacGym)
            self.handler.gym.refresh_rigid_body_state_tensor(self.handler.sim)
            rb_states = self.handler._rigid_body_states  # (N_total_bodies, 13)

            # Resolve global rigid body indices for the target link once
            if not hasattr(self, "_gym_link_gidxs"):
                gidxs = []
                for i in range(self.num_envs):
                    gidx = self.handler._env_rigid_body_global_indices[i]["robot"][self.link_name]
                    gidxs.append(gidx)
                self._gym_link_gidxs = gidxs

            link_states = rb_states[self._gym_link_gidxs, :]
            pos_w = link_states[:, 0:3]
            quat_xyzw = link_states[:, 3:7]
            # Convert to (w,x,y,z)
            quat_wxyz = convert_quat(quat_xyzw, to="wxyz")

        elif sim_type == "mujoco":
            # Resolve body id using cached names from handler (avoid mj_name2id signature mismatch)
            if not hasattr(self, "_mj_link_bid"):
                bid = None
                body_names = self.handler.body_names
                for i, bn in enumerate(body_names):
                    if bn == self.link_name or bn.endswith("/" + self.link_name) or bn.split("/")[-1] == self.link_name:
                        bid = i
                        break
                if bid is None:
                    warnings.warn(f"LidarPointCloud: link '{self.link_name}' not found in MuJoCo body names.")
                    return {robot_name: None}
                self._mj_link_bid = int(bid)

            # MuJoCo xquat is (w,x,y,z)
            pos_np = self.handler.physics.data.xpos[self._mj_link_bid]
            quat_np = self.handler.physics.data.xquat[self._mj_link_bid]
            pos_w = torch.as_tensor(pos_np, device=self.device, dtype=torch.float32).view(1, 3)
            quat_wxyz = torch.as_tensor(quat_np, device=self.device, dtype=torch.float32).view(1, 4)

        else:
            # IsaacSim or other backends not yet supported here
            warnings.warn(f"LidarPointCloud: simulator '{sim_type}' not supported for LiDAR pose fetch.")
            return {robot_name: None}

        # Apply optical center offset in the sensor's local +Z (after model-specific mounting rotations)
        if self.apply_optical_center_offset and self.optical_center_offset_z != 0.0:
            offset_local = torch.tensor([0.0, 0.0, self.optical_center_offset_z], device=pos_w.device).view(-1, 3)
            if offset_local.shape[0] != pos_w.shape[0]:
                offset_local = offset_local.repeat(pos_w.shape[0], 1)
            pos_w = pos_w + quat_apply(quat_wxyz, offset_local)

        # Update sensor pose buffers (LidarSensor expects XYZW)
        self.sensor_pos_tensor[:, :pos_w.shape[1]] = pos_w
        quat_xyzw = convert_quat(quat_wxyz, to="xyzw")
        self.sensor_quat_tensor_xyzw[:, :quat_xyzw.shape[1]] = quat_xyzw

        # Run LiDAR update
        lidar_tensor_local, dist_tensor = self.sensor.update()

        # Compute world coordinates from local points
        # lidar_tensor_local: (E, S, V, H, 3)
        E, S, V, H, _ = lidar_tensor_local.shape
        pts_local = lidar_tensor_local.view(E, -1, 3)
        # Expand pose to match points
        quat_rep = quat_wxyz.unsqueeze(1).repeat(1, pts_local.shape[1], 1).view(-1, 4)
        vec = pts_local.view(-1, 3)
        rot = quat_apply(quat_rep, vec).view(E, -1, 3)
        pos_rep = pos_w.unsqueeze(1).repeat(1, pts_local.shape[1], 1)
        pts_world = rot + pos_rep
        pts_world = pts_world.view(E, S, V, H, 3)

        return {
            robot_name: {
                "points_local": lidar_tensor_local,
                "points_world": pts_world,
                "dist": dist_tensor,
                "link": self.link_name,
            }
        }

@configclass
class SensorsCfg:
    contact_forces: ContactForces = ContactForces()
    # Disabled by default to avoid extra overhead/missing-link issues unless explicitly enabled by user
    lidar_pointcloud: LidarPointCloud = LidarPointCloud(enabled=False)
