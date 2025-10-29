from __future__ import annotations
from typing import Literal

import torch

from metasim.sim.base import BaseSimHandler, BaseQueryType
from metasim.utils.math import sample_uniform

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

try:
    import mujoco  # noqa: F401
except ImportError:
    pass


class MaterialRandomizer(BaseQueryType):
    def __init__(self,
                obj_name: str,
                body_names: list[str] | str | None = None,
                static_friction_range: list | tuple = (1.0, 1.0),
                dynamic_friction_range: list | tuple = (1.0, 1.0),
                restitution_range: list | tuple = (0.0, 0.0),
                num_buckets: int = 1,
                make_consistent: bool = False):
        super().__init__()
        self.obj_name = obj_name
        self.set_body_names = [body_names] if isinstance(body_names, str) else body_names
        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.num_buckets = num_buckets
        self.make_consistent = make_consistent

    def bind_handler(self, handler:BaseSimHandler, *args, **kwargs):
        super().bind_handler(handler, *args, **kwargs)
        self.simulator_name = handler.scenario.simulator
        self.initialize()

    def __call__(self, env_ids=None):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(self.handler.num_envs, device="cpu")
        else:
            env_ids = torch.tensor(env_ids).cpu()
        self.randomize(env_ids)

    def initialize(self):
        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [self.static_friction_range, self.dynamic_friction_range, self.restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = sample_uniform(ranges[:, 0], ranges[:, 1], (self.num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        if self.make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        self.body_names = self.handler.get_body_names(self.obj_name, sort=False)
        self.set_body_ids = torch.tensor([self.body_names.index(_name) for _name in self.set_body_names], dtype=torch.int, device="cpu") if self.set_body_names is not None else torch.arange(len(self.body_names), dtype=torch.int, device="cpu")

        self.num_shapes_per_body = None
        if self.simulator_name == "isaacsim":
            if self.obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[self.obj_name]
                # obtain number of shapes per body (needed for indexing the material properties correctly)
                # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
                #  per body. We use the physics simulation view to obtain the number of shapes per body.
                self.num_shapes_per_body = []
                for link_path in obj_inst.root_physx_view.link_paths[0]:
                    link_physx_view = obj_inst._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                    self.num_shapes_per_body.append(link_physx_view.max_shapes)
                # ensure the parsing is correct
                num_shapes = sum(self.num_shapes_per_body)
                expected_shapes = obj_inst.root_physx_view.max_shapes
                if num_shapes != expected_shapes:
                    raise ValueError(
                        "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                        f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                    )

    def randomize(self, env_ids: torch.Tensor):
        if self.simulator_name == "isaacsim":
            if self.obj_name in self.handler.scene.articulations:
                obj_inst = self.handler.scene.articulations[self.obj_name]
            elif self.obj_name in self.handler.scene.rigid_objects:
                obj_inst = self.handler.scene.rigid_objects[self.obj_name]
            else:
                raise ValueError(f"Randomization term 'randomize_rigid_body_material' not supported for asset: {self.obj_name}.")

            # retrieve material buffer from the physics simulation
            materials = obj_inst.root_physx_view.get_material_properties()
            # randomly assign material IDs to the geometries
            total_num_shapes = obj_inst.root_physx_view.max_shapes
            bucket_ids = torch.randint(0, self.num_buckets, (len(env_ids), total_num_shapes), device="cpu")
            material_samples = self.material_buckets[bucket_ids]
            # update material buffer with new samples
            if self.num_shapes_per_body is not None:
                # sample material properties from the given ranges
                for body_id in self.set_body_ids:
                    # obtain indices of shapes for the body
                    start_idx = sum(self.num_shapes_per_body[:body_id])
                    end_idx = start_idx + self.num_shapes_per_body[body_id]
                    # assign the new materials
                    # material samples are of shape: num_env_ids x total_num_shapes x 3
                    materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
            else:
                # assign all the materials
                materials[env_ids] = material_samples[:]
            # apply to simulation
            obj_inst.root_physx_view.set_material_properties(materials, env_ids)


# class MassRandomizer(BaseQueryType):
#     def __init__(self,
#                 obj_name: str,
#                 body_names: list[str] | str | None = None,
#                 mass_distribution_params: list | tuple = (-1.0, 3.0),
#                 operation: Literal["add", "scale", "abs"] = "add",
#                 distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
#                 recompute_inertia: bool = True,
#                 ):
#         super().__init__()
#         self.obj_name = obj_name
#         self.set_body_names = [body_names] if isinstance(body_names, str) else body_names
#         self.mass_distribution_params = mass_distribution_params
#         self.operation = operation
#         self.distribution = distribution
#         self.recompute_inertia = recompute_inertia

#     def bind_handler(self, handler:BaseSimHandler, *args, **kwargs):
#         super().bind_handler(handler, *args, **kwargs)
#         self.simulator_name = handler.scenario.simulator
#         self.initialize()

#     def initialize(self):
#         # extract the used quantities (to enable type-hinting)
#         self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
#         self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
#         # check for valid operation
#         if self.operation == "scale":
#             if "mass_distribution_params" in cfg.params:
#                 _validate_scale_range(
#                     self.mass_distribution_params, "mass_distribution_params", allow_zero=False
#                 )
#         elif self.operation not in ("abs", "add"):
#             raise ValueError(
#                 "Randomization term 'randomize_rigid_body_mass' does not support operation:"
#                 f" '{self.operation}'."
#             )

#     def __call__(self, env_ids: torch.Tensor | None):
#         # resolve environment ids
#         if env_ids is None:
#             env_ids = torch.arange(self.handler.num_envs, device="cpu")
#         else:
#             env_ids = torch.tensor(env_ids).cpu()

#         # resolve environment ids
#         if env_ids is None:
#             env_ids = torch.arange(env.scene.num_envs, device="cpu")
#         else:
#             env_ids = env_ids.cpu()

#         # resolve body indices
#         if self.asset_cfg.body_ids == slice(None):
#             body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
#         else:
#             body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

#         # get the current masses of the bodies (num_assets, num_bodies)
#         masses = self.asset.root_physx_view.get_masses()

#         # apply randomization on default values
#         # this is to make sure when calling the function multiple times, the randomization is applied on the
#         # default values and not the previously randomized values
#         masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

#         # sample from the given range
#         # note: we modify the masses in-place for all environments
#         #   however, the setter takes care that only the masses of the specified environments are modified
#         masses = _randomize_prop_by_op(
#             masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
#         )

#         # set the mass into the physics simulation
#         self.asset.root_physx_view.set_masses(masses, env_ids)

#         # recompute inertia tensors if needed
#         if recompute_inertia:
#             # compute the ratios of the new masses to the initial masses
#             ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
#             # scale the inertia tensors by the the ratios
#             # since mass randomization is done on default values, we can use the default inertia tensors
#             inertias = self.asset.root_physx_view.get_inertias()
#             if isinstance(self.asset, Articulation):
#                 # inertia has shape: (num_envs, num_bodies, 9) for articulation
#                 inertias[env_ids[:, None], body_ids] = (
#                     self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
#                 )
#             else:
#                 # inertia has shape: (num_envs, 9) for rigid object
#                 inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios
#             # set the inertia tensors into the physics simulation
#             self.asset.root_physx_view.set_inertias(inertias, env_ids)


#     def randomize(self):
#         pass



##########################################################################################
# Private Helper Functions
# FROM NVIDIA ISAAC LAB
##########################################################################################
def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")
##########################################################################################
# Private Helper Functions
# FROM NVIDIA ISAAC LAB
##########################################################################################
