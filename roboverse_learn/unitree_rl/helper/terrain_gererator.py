from __future__ import annotations

import math

import numpy as np

import roboverse_learn.unitree_rl.helper.terrain_utils as terrain_utils
from roboverse_learn.unitree_rl.configs.base_terrain import (
    BaseTerrainConfig,
    GapConfig,
    ObstacleConfig,
    PitConfig,
    SlopeConfig,
    StairConfig,
    StoneConfig,
    TerrainConfig,
)


class TerrainGenerator:
    """Abstract base class for backend-specific terrain implementation."""

    def __init__(self, config: TerrainConfig = None):
        if config is not None:
            self._parse_cfg(config)

    def _parse_cfg(self, config: TerrainConfig):
        """Parse the terrain configuration."""
        self.config = config
        self.height_mat = np.zeros((config.num_rows, config.num_cols), dtype=np.int16)
        self.horizontal_scale = config.horizontal_scale
        self.vertical_scale = config.vertical_scale
        self.margin = config.margin

    def _make_sub_terrain(self, config: BaseTerrainConfig):
        terrain = terrain_utils.SubTerrain(
            config.type,
            width=math.ceil(config.size[0] / self.horizontal_scale),
            length=math.ceil(config.size[1] / self.horizontal_scale),
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )
        return terrain

    def _make_slope(self, config: SlopeConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pyramid_sloped_terrain(
            terrain,
            slope=config.slope,
            platform_size=config.platform_size,
        )
        if config.random:
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=2.0 * self.horizontal_scale
            )
        return config.origin, terrain

    def _make_stair(self, config: StairConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pyramid_stairs_terrain(
            terrain,
            step_width=config.step[0],
            step_height=config.step[1],
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_obstacle(self, config: ObstacleConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.discrete_obstacles_terrain(
            terrain,
            max_height=config.max_height,
            min_size=config.rectangle_params[0],
            max_size=config.rectangle_params[1],
            num_rects=config.rectangle_params[2],
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_stone(self, config: StoneConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.stepping_stones_terrain(
            terrain,
            stone_size=config.stone_params[0],
            stone_distance=config.stone_params[1],
            max_height=config.max_height,
            platform_size=config.platform_size,
        )
        return config.origin, terrain

    def _make_gap(self, config: GapConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.gap_terrain(terrain, gap_size=config.gap_size, platform_size=min(config.size))
        return config.origin, terrain

    def _make_pit(self, config: PitConfig):
        terrain = self._make_sub_terrain(config)
        terrain_utils.pit_terrain(terrain, depth=config.depth, platform_size=min(config.size))
        return config.origin, terrain

    def _add_terrain_to_map(self, origin, terrain: terrain_utils.SubTerrain):
        start_row = math.floor(origin[0] / self.horizontal_scale)
        start_col = math.floor(origin[1] / self.horizontal_scale)
        end_row = start_row + terrain.width
        end_col = start_col + terrain.length
        self.height_mat[start_row:end_row, start_col:end_col] = terrain.height_field_raw

    def generate_terrain(self, config: TerrainConfig = None):
        """Generate terrain based on the specified type and parameters."""
        if config is not None:
            self._parse_cfg(config)

        assert hasattr(self, "config"), "Terrain configuration must be set before generating terrain."
        for t in self.config.elements.keys():
            func_name = f"_make_{t}"
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                for cfg in self.config.elements[t]:
                    origin, terrain = func(cfg)
                    self._add_terrain_to_map(origin, terrain)
            else:
                raise NotImplementedError(f"Terrain type '{t}' is not implemented in {self.__class__.__name__}")
        row_padding_size = self.config.margin_num_rows
        col_padding_size = self.config.margin_num_cols
        self.height_mat_pad = np.pad(
            self.height_mat,
            ((row_padding_size, row_padding_size), (col_padding_size, col_padding_size)),
            mode="constant",
            constant_values=0,
        )
        vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
            height_field_raw=self.height_mat_pad,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=0.1,
        )

        return vertices, triangles

    @property
    def height_measure(self):
        """Get the height map of the generated terrain."""
        return self.height_mat * self.vertical_scale
