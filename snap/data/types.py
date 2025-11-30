# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful types and processing/configuration dataclasses for SNAP (PyTorch-ready)."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, Optional, Tuple

# Note: the original JAX/TFDS code used etils.array_types.BoolArray and TFDS BuilderConfig.
# For a PyTorch conversion we use plain Python typing. If you want more precise array typing,
# replace `Any` with e.g. `numpy.ndarray` or `torch.Tensor` wherever appropriate.
DataDict = Dict[str, Any]  # data for a given segment, scene, camera, etc.
SegmentsDict = Dict[str, DataDict]  # a collection of segments
RastersDict = Dict[str, Any]  # e.g. {"class_name": boolean_array_of_shape_[H, W]}


# A valid plane height is always positive w.r.t. the scene coordinate system.
INVALID_GROUND_PLANE_HEIGHT = -1.0

# Semantic classes
AERIAL_BUILDING_CLASSES = ["buildings_raw", "buildings_contoured"]
SURFEL_ROAD_CLASSES = [
    "crosswalk",
    "sidewalk",
    "pavedroad",
    "stopline",
    "line",
    "otherlanemarking",
]


class DataMode(str, enum.Enum):
    SINGLE_SCENE = "single_scene"
    PAIR_SCENES = "pair_scenes"
    PAIR_SCENE_VIEW = "pair_scene_view"


@dataclasses.dataclass
class SceneConfig:
    """Configuration for scene (grid and view) selection."""

    grid_size: Tuple[int, int, int] = (24, 32, 12)
    grid_z_offset: int = 4
    center_grid_around_reference: bool = True
    num_views: int = 10
    min_distance_between_views: float = 1.5
    max_distance_between_views: float = 15
    only_views_in_grid: bool = True
    reference_cameras: Tuple[str, ...] = ("side_left", "side_right")
    reference_vehicles: Tuple[str, ...] = ("CAR",)
    constrain_all_cameras: bool = True
    single_segment_add_front_rear_cameras: bool = True
    single_segment_add_front_rear_cameras_every: Optional[int] = 3
    streetview_hfov_deg: float = 72.0
    camera_frustum_depth: float = 16.0


@dataclasses.dataclass
class PairingConfig:
    """Configuration for pairing scenes."""

    min_overlap: float = 0.3
    max_overlap: float = 0.7
    min_distance_to_scene_views: Optional[float] = None
    max_elevation_diff: float = 2.0
    num_queries_per_scene: Optional[int] = None
    ratio_trekker: float = 0.5


@dataclasses.dataclass
class ProcessingConfig:
    """Configuration for the entire data processing pipeline.

    NOTE:
      - Original code inherited from `tfds.core.BuilderConfig`. For PyTorch usage we use a
        plain dataclass. If you previously relied on TFDS-specific features, reintroduce them
        where needed or adapt your dataset code accordingly.
      - This class references `RastersConfig` and `lidar_config` in some methods. The original
        file defined/expected those elsewhere in the codebase. When integrating, make sure to
        provide / import your project's `RastersConfig` and `lidar_config` objects into the
        runtime environment where ProcessingConfig.from_dict is used.
    """

    # paths
    data_path: Optional[str] = None
    scenes_sstable_path: str = ""
    frames_sstable_path: str = ""
    s2_cell_list_path: str = ""

    # processing options
    image_downsampling_factor: Optional[int] = None
    pose_tag: Optional[str] = None

    split_by_s2_cell: bool = True
    generate_training_split: bool = True
    max_total_area_km2: Optional[float] = None
    evaluation_num_cells: Optional[int] = 5
    evaluation_s2_level: int = 14
    evaluation_max_num_examples: Optional[int] = None

    scene_types: Tuple[str, ...] = ("OUTDOOR",)
    vehicle_types: Tuple[str, ...] = ("CAR", "TREKKER")
    vehicle_types_for_map: Optional[Tuple[str, ...]] = ("CAR",)
    bin_level: int = 18

    single_segment_per_scene: bool = True
    min_num_runs_per_scene: int = 2
    min_num_segments_per_vehicle: int = 1
    scene_config: SceneConfig = dataclasses.field(default_factory=SceneConfig)

    mode: DataMode = DataMode.SINGLE_SCENE
    pairing_config: PairingConfig = dataclasses.field(default_factory=PairingConfig)

    # NOTE: the original code assumes attributes like `rasters_config` and `lidar_config`
    # exist on instances; they are set in `from_dict` below. You may want to add them as
    # dataclass fields if you prefer explicitness.
    # rasters_config: RastersConfig = dataclasses.field(default_factory=RastersConfig)
    # lidar_config: LidarConfig = dataclasses.field(default_factory=LidarConfig)

    def need_lidar_semantics(self) -> bool:
        """Return whether ground-truth semantic rasters / lidar semantics are required.

        The original implementation used a property expecting `rasters_config` and
        `lidar_config` attributes with `add_gt_semantics` fields. Here we keep the same
        semantics but guard against missing attributes.
        """
        add_from_rasters = False
        add_from_lidar = False
        rasters = getattr(self, "rasters_config", None)
        lidar = getattr(self, "lidar_config", None)
        if rasters is not None:
            add_from_rasters = bool(getattr(rasters, "add_gt_semantics", False))
        if lidar is not None:
            add_from_lidar = bool(getattr(lidar, "add_gt_semantics", False))
        return add_from_rasters or add_from_lidar

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProcessingConfig":
        """Create ProcessingConfig from a plain dict.

        Mimics the original logic:
         - handles legacy 'pair_scenes' key by mapping to `mode`
         - constructs SceneConfig, RastersConfig, and PairingConfig sub-objects from dicts

        Important: this expects a `RastersConfig` symbol to be importable in the runtime.
        In the original JAX code `RastersConfig` is defined elsewhere in the repository. If
        you keep the same module layout, just import it before calling this method.
        """
        cfg = dict(config_dict)  # shallow copy to avoid mutating caller dict

        # legacy flag
        if cfg.pop("pair_scenes", False):
            cfg["mode"] = DataMode.PAIR_SCENES
        elif "mode" in cfg:
            # allow string or DataMode
            cfg["mode"] = DataMode(cfg["mode"]) if not isinstance(cfg["mode"], DataMode) else cfg["mode"]

        # instantiate nested configs
        # SceneConfig
        scene_cfg_in = cfg.get("scene_config", {})
        if not isinstance(scene_cfg_in, SceneConfig):
            cfg["scene_config"] = SceneConfig(**scene_cfg_in)

        # PairingConfig
        pairing_cfg_in = cfg.get("pairing_config", {})
        if not isinstance(pairing_cfg_in, PairingConfig):
            cfg["pairing_config"] = PairingConfig(**pairing_cfg_in)

        # RastersConfig: this type is defined elsewhere in original SNAP repo.
        # Try to build it if a dict is provided. If the symbol isn't available, keep the raw dict.
        rasters_cfg_in = cfg.get("rasters_config", {})
        if not isinstance(rasters_cfg_in, dict):
            cfg["rasters_config"] = rasters_cfg_in
        else:
            # Lazy import / resolution for RastersConfig to avoid circular imports.
            try:
                # Adjust import path to your project layout if needed.
                from .rasters import RastersConfig  # type: ignore
            except Exception:
                # If RastersConfig isn't available, just keep the raw dict; caller should
                # handle conversion or set the attribute manually.
                cfg["rasters_config"] = rasters_cfg_in
            else:
                cfg["rasters_config"] = RastersConfig(**rasters_cfg_in)

        # Construct final ProcessingConfig (extras are accepted as attributes on the instance)
        proc_cfg = cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

        # Attach any remaining keys (like rasters_config) that aren't explicit dataclass fields
        # as attributes on the instance to preserve original behavior.
        for k, v in cfg.items():
            if k not in cls.__dataclass_fields__:
                setattr(proc_cfg, k, v)

        return proc_cfg
