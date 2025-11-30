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

"""Training and evaluation loader adapted to PyTorch / iGibson F³Loc dataset.

This is a minimal-change translation of the JAX/TFDS `loader.py`.  It replaces TF
and TFDS parts with PyTorch datasets and dataloaders adapted to the iGibson dataset
layout described in the project's General Rules document.

Key differences vs original:
- Uses torch.utils.data.Dataset and DataLoader instead of tf.data / tfds builders.
- Expects an on-disk layout like:
    Dataset/
      gibson_f/   <-- dataset split directories
        SceneName1/
          rgb/
            00000.png
            00001.png
            ...
          poses.txt
          map.png
          depth40.txt
          depth160.txt
      desdf/
        SceneName1/desdf.npy
- Returns a `DatasetTuple(train_iter, eval_iter, test_iter, meta_data)` similar to
  original dataset_utils.Dataset. Iterators yield dictionaries with the same keys
  expected by the rest of the code (T_view2scene, camera intrinsics, images, rasters...).
"""

from __future__ import annotations

import collections
import functools
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Keep original imports from snap package (unchanged).
import snap.configs.defaults as default_configs
from snap.data import types
from snap.data.types import DataDict
from snap.utils import geometry
from snap.utils import grids
from tqdm import tqdm
# ---------------------------------------------------------------------
# Lightweight compatibility: DatasetTuple (similar to scenic.dataset_lib.Dataset)
# ---------------------------------------------------------------------
@dataclass
class DatasetTuple:
  train: Optional[Iterator[DataDict]]
  eval: Optional[Iterator[DataDict]]
  test: Optional[Iterator[DataDict]]
  meta: Dict[str, object]


# ---------------------------------------------------------------------
# Helpers for iGibson dataset format
# ---------------------------------------------------------------------
IGIBSON_INTRINSICS = np.array([[240.0, 0.0, 320.0], [0.0, 240.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
MAP_METER_PER_PIXEL = 0.01  # map.png resolution 0.01 m/pix (from General Rules)
DESDF_RESOLUTION = 0.1  # desdf is 0.1 m/pix

def _read_poses_txt(path: str) -> np.ndarray:
  """Read poses.txt with lines x y yaw -> returns array [N, 3]."""
  poses = []
  with open(path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      parts = line.split()
      if len(parts) < 3:
        raise ValueError(f'Bad pose line in {path}: {line}')
      x, y, yaw = float(parts[0]), float(parts[1]), float(parts[2])
      poses.append([x, y, yaw])
  return np.array(poses, dtype=np.float32)


def _read_depth_file(path: str) -> List[List[float]]:
  """Read depth*.txt files: each line has ray depth values left->right."""
  entries = []
  with open(path, 'r') as f:
    for line in f:
      tokens = line.strip().split()
      if not tokens:
        entries.append([])
      else:
        entries.append([float(x) for x in tokens])
  return entries


def _load_desdf(scene_desdf_path: str):
  """Loads desdf/<SceneName>/desdf.npy which stores {'l','t','desdf'}."""
  if not os.path.exists(scene_desdf_path):
    return None
  data = np.load(scene_desdf_path, allow_pickle=True).item()
  return data


def _world_to_map(x_world: float, y_world: float, map_w: int, map_h: int) -> Tuple[int, int]:
  """Map transform from world meters to map pixel coordinates (map centered)."""
  x_map = int(x_world / MAP_METER_PER_PIXEL + (map_w / 2.0))
  y_map = int(y_world / MAP_METER_PER_PIXEL + (map_h / 2.0))
  return x_map, y_map


# ---------------------------------------------------------------------
# PyTorch Dataset implementation reading iGibson scenes
# ---------------------------------------------------------------------
class IGibsonSceneDataset(Dataset):
  """Dataset reading a single split (gibson_f/g/g_t) and yielding "scene examples".

  Each index corresponds to one view/frame. The dataset yields dicts compatible
  with `process_example` and `process_scene_example` expectations:
    - 'views': { 'T_camera2scene': {...}, 'intrinsics': {...}, 'color_image': HxWx3 uint8 }
    - 'scene_id': scene_name
    - 'vehicle_type': choose 'AGENT' (placeholder)
    - 'coordinates': { 'center_latlng': ... }  # kept for compatibility; we set None
    - 'rasters': { 'rgb': map_image (H_map x W_map x 3) } if requested
    - 'point_cloud': optional, not provided by iGibson default
  """

  def __init__(self,
               root_dir: str,
               split_dir: str,
               scene_names: List[str],
               add_images: bool = True,
               add_rasters: bool = True,
               add_lidar_rays: bool = False,
               image_loader: Optional[Callable[[str], np.ndarray]] = None):
    """
    Args:
      root_dir: base dataset directory (contains desdf/ and gibson_f/ etc.).
      split_dir: e.g. 'gibson_f' (folder inside root_dir)
      scene_names: list of scene folder names (subdirs of root_dir/split_dir).
      add_images: if True, dataset will read RGB images.
      add_rasters: if True, dataset will read map.png into rasters['rgb'].
      add_lidar_rays: unsupported for iGibson default; kept for API.
      image_loader: callable(path) -> HxWx3 numpy uint8. If None, uses PIL.
    """
    self.root_dir = root_dir
    self.split_dir = split_dir
    self.split_path = os.path.join(root_dir, split_dir)
    self.scene_names = list(scene_names)
    self.add_images = add_images
    self.add_rasters = add_rasters
    self.add_lidar_rays = add_lidar_rays
    self.image_loader = image_loader or self._default_image_loader

    # Build index: a list of (scene_name, frame_idx, image_path)
    self.index = []  # entries: dict with scene, idx, image_path, pose, depths
    for scene in tqdm(self.scene_names):
      scene_dir = os.path.join(self.split_path, scene)
      if not os.path.isdir(scene_dir):
        print('Scene %s not found at %s, skipping.', scene, scene_dir)
        continue

      rgb_dir = os.path.join(scene_dir, 'rgb')
      poses_path = os.path.join(scene_dir, 'poses.txt')
      map_path = os.path.join(scene_dir, 'map.png')
      depth40_path = os.path.join(scene_dir, 'depth40.txt')
      depth160_path = os.path.join(scene_dir, 'depth160.txt')
      desdf_path = os.path.join(self.root_dir, 'desdf', scene, 'desdf.npy')

      poses = _read_poses_txt(poses_path) if os.path.exists(poses_path) else np.zeros((0, 3), dtype=np.float32)
      depth40 = _read_depth_file(depth40_path) if os.path.exists(depth40_path) else [None] * len(poses)
      depth160 = _read_depth_file(depth160_path) if os.path.exists(depth160_path) else [None] * len(poses)
      desdf = _load_desdf(desdf_path)
      map_img = None
      if self.add_rasters and os.path.exists(map_path):
        from PIL import Image
        map_img = np.array(Image.open(map_path).convert('RGB'))

      # Collect image files sorted by name (matches poses order by convention)
      if os.path.isdir(rgb_dir):
        files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
      else:
        files = []

      # If poses exist, prefer their length; otherwise use number of files.
      n_examples = max(len(poses), len(files))
      for i in range(n_examples):
        image_path = os.path.join(rgb_dir, files[i]) if i < len(files) else None
        pose = poses[i] if i < len(poses) else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        d40 = depth40[i] if i < len(depth40) else None
        d160 = depth160[i] if i < len(depth160) else None
        self.index.append({
            'scene': scene,
            'scene_dir': scene_dir,
            'frame_idx': i,
            'image_path': image_path,
            'pose': pose,
            'map': map_img,
            'desdf': desdf,
            'depth40': d40,
            'depth160': d160,
        })

  def _default_image_loader(self, path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert('RGB')
    return np.array(img)

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, idx: int) -> Dict:
    info = self.index[idx]
    out: Dict = {}
    # Build view-level fields expected by process_scene_example
    # T_camera2scene: we convert [x,y,yaw] into Transform3D-like dict:
    x, y, yaw = info['pose']
    # Minimal Transform3D representation: translation + rotation quaternion or matrix.
    # For compatibility, we provide a dict with 'translation' and 'rotation' (yaw around z).
    translation = [float(x), float(y), 0.0]
    # rotation as yaw angle (we pack it as a 3x3 because geometry.Transform3D.from_dict may accept it)
    # Keep compatibility simple: provide a 4x4 homogeneous matrix as list if needed by geometry.Transform3D
    c, s = math.cos(yaw), math.sin(yaw)
    rot3 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    T_camera2scene = {
        'R': rot3,  # 3x3
        't': np.asarray(translation, dtype=np.float32),
    }

    intrinsics = {
        'K': IGIBSON_INTRINSICS.copy(),  # keep same key naming as original code expects
        'width': 640,
        'height': 480,
    }

    views = {
        'T_camera2scene': T_camera2scene,
        'intrinsics': intrinsics,
    }

    if self.add_images and info['image_path'] is not None:
      image = self.image_loader(info['image_path']).astype(np.uint8)
      views['color_image'] = image

    out['views'] = views
    out['scene_id'] = info['scene']
    out['vehicle_type'] = 'AGENT'
    out['coordinates'] = {'center_latlng': None}
    if self.add_rasters and info['map'] is not None:
      out['rasters'] = {'rgb': info['map']}
    # iGibson doesn't come with point clouds by default; leave absent unless required.
    return out


# ---------------------------------------------------------------------
# Processing functions (mirrors original names)
# ---------------------------------------------------------------------
def pad_lidar_rays(rays: Dict[str, np.ndarray], num_target: int) -> DataDict:
  """Pad lidar rays to a fixed number (numpy version)."""
  num = rays['points'].shape[0]
  num_sampled = min(num, num_target)
  indices = np.random.permutation(num)[:num_sampled]
  points = rays['points'][indices]
  origins = rays['origins'][indices]
  missing = num_target - num_sampled
  rays_padded = {
      'points': np.pad(points, ((0, missing), (0, 0))),
      'origins': np.pad(origins, ((0, missing), (0, 0))),
      'mask': np.concatenate([np.ones(num_sampled, dtype=bool), np.zeros(missing, dtype=bool)]),
  }
  if 'semantics' in rays:
    semantics = rays['semantics'][indices]
    rays_padded['semantics'] = np.pad(semantics, ((0, missing),))
  return rays_padded


def process_scene_example(
    example: Dict,
    config: types.ProcessingConfig,
    dtype=np.float32,
    is_single_view: bool = False,
) -> Dict:
  """Process one on-disk example (numpy dict) into model-ready dict.

  This mirrors the TF version but operates on numpy structures and iGibson fields.
  """
  ret = {
      'T_view2scene': example['views']['T_camera2scene'],
      'camera': example['views']['intrinsics'],
      'scene_id': example['scene_id'],
      'vehicle_type': example['vehicle_type'],
      'latlng': example['coordinates']['center_latlng'],
  }
  if config.image_downsampling_factor is None:
    scale = 1
  else:
    scale = config.image_downsampling_factor

  if (config.add_images if hasattr(config, 'add_images') else True) or is_single_view:
    if 'color_image' in example['views']:
      images = example['views']['color_image'].astype(np.float32) / 255.0
      # Optionally downsample
      if scale != 1:
        import cv2
        h, w = images.shape[:2]
        images = cv2.resize(images, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
      ret['images'] = images.astype(dtype)
  if getattr(config, 'add_lidar_rays', False) and not is_single_view:
    if 'point_cloud' in example and 'rays' in example['point_cloud']:
      ret['lidar_rays'] = pad_lidar_rays(example['point_cloud']['rays'], config.num_rays if hasattr(config, 'num_rays') else 2048)
  if getattr(config, 'add_rasters', True) and not is_single_view:
    rasters = example.get('rasters', None) if hasattr(example, 'rasters') else None
    if rasters is not None:
      ret['rasters'] = {}
      if 'rgb' in rasters:
        rgb = rasters['rgb'].astype(np.float32) / 255.0
        if scale != 1:
          import cv2
          h, w = rgb.shape[:2]
          rgb = cv2.resize(rgb, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        ret['rasters']['rgb'] = rgb.astype(dtype)
  return ret


def process_example(
    example: Dict,
    config: types.ProcessingConfig,
    dtype=np.float32,
) -> Dict:
  """Wrapper that matches original mode selection semantics."""
  mode = config.mode if hasattr(config, 'mode') else types.DataMode.SINGLE_SCENE
  if isinstance(mode, str):
    mode = types.DataMode(mode)
  if mode == types.DataMode.SINGLE_SCENE:
    return process_scene_example(example, config, dtype)
  else:
    # Pairing modes are not implemented for iGibson by default in this translation.
    # Keep minimal behavior: raise if user requests pairing.
    raise NotImplementedError('PAIR_SCENES and PAIR_SCENE_VIEW modes are not implemented in the iGibson loader.')


def process_scene_batch(batch: Dict) -> Dict:
  """Convert camera dicts into geometry objects (numpy -> geometry.Transform3D)."""
  # Expect batch['camera'] contains intrinsics dicts and batch['T_view2scene'] is rotation+translation dict.
  cameras = geometry.FisheyeCamera.from_dict(batch['camera'])
  tfm_view2scene = geometry.Transform3D(**batch['T_view2scene'])
  batch.update({
      'T_view2scene': tfm_view2scene,
      'camera': cameras,
      'scene_id': np.asarray(batch['scene_id']).astype(str),
      'vehicle_type': np.asarray(batch['vehicle_type']).astype(str),
  })
  return batch


def process_batch(batch: Dict, config: types.ProcessingConfig) -> Dict:
  """Process a batch (numpy-based) into ready-to-use types for model training/inference.

  For now, only SINGLE_SCENE is supported similar to process_example above.
  """
  mode = config.mode if hasattr(config, 'mode') else types.DataMode.SINGLE_SCENE
  if isinstance(mode, str):
    mode = types.DataMode(mode)
  if mode == types.DataMode.SINGLE_SCENE:
    batch = process_scene_batch(batch)
  else:
    raise NotImplementedError('Pairing batch processing not implemented for iGibson loader.')
  return batch


# ---------------------------------------------------------------------
# DataLoader iterator helpers
# ---------------------------------------------------------------------
def _numpy_collate(batch_list: List[Dict]) -> Dict:
  """Collate list of examples into a single batch dict.

  Collation is conservative: arrays of same shape are stacked; missing keys are skipped.
  """
  out = {}
  keys = set().union(*(set(x.keys()) for x in batch_list))
  for k in keys:
    vals = [b.get(k, None) for b in batch_list]
    # If values are dicts, try to collate recursively.
    if all(isinstance(v, dict) or v is None for v in vals):
      # Collate dict entries
      sub_keys = set().union(*(set(v.keys()) if isinstance(v, dict) else set() for v in vals))
      out[k] = {}
      for sk in sub_keys:
        sub_vals = [v.get(sk) if isinstance(v, dict) else None for v in vals]
        # Try to stack numpy arrays when shapes match.
        if all(isinstance(sv, np.ndarray) for sv in sub_vals if sv is not None):
          try:
            out[k][sk] = np.stack([sv for sv in sub_vals], axis=0)
          except Exception:
            out[k][sk] = sub_vals
        else:
          out[k][sk] = sub_vals
    else:
      # Try to stack numpy arrays
      if all(isinstance(v, np.ndarray) for v in vals if v is not None):
        try:
          out[k] = np.stack([v for v in vals], axis=0)
        except Exception:
          out[k] = vals
      else:
        out[k] = vals
  return out


def _dataloader_iterator(loader: DataLoader, process_batch_fn: Optional[Callable] = None) -> Iterator[Dict]:
  """Yields batches from a PyTorch DataLoader, converting tensors/ndarrays to numpy and calling process_batch_fn."""
  for batch in loader:
    # DataLoader will give us numpy arrays or lists (because dataset returns numpy).
    if isinstance(batch, dict):
      batch_np = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    else:
      batch_np = batch
    # If collate produced nested dicts, keep them.
    if process_batch_fn is not None:
      batch_np = process_batch_fn(batch_np)
    yield batch_np


# ---------------------------------------------------------------------
# High-level dataset construction APIs (replace TFDS builders)
# ---------------------------------------------------------------------
def list_scenes_in_split(root_dir: str, split_name: str) -> List[str]:
  """Return sorted list of scene folder names under root_dir/<split_name>/."""
  split_path = os.path.join(root_dir, split_name)
  if not os.path.isdir(split_path):
    raise FileNotFoundError(f'Split {split_name} not found at {split_path}')
  scenes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
  return scenes


def dataset_iterator_from_folder(
    root_dir: str,
    split_name: str,
    batch_size: int,
    is_training: bool = True,
    process_example_fn: Optional[Callable[[Dict], Dict]] = None,
    process_batch_fn: Optional[Callable[[Dict], Dict]] = None,
    shuffle_seed: int = 0,
    num_workers: int = 4,
    drop_last: bool = True,
) -> Iterator[Dict]:
  """Create a PyTorch DataLoader iterator reading from the iGibson folder layout.

  Returns an iterator yielding processed numpy batches (not torch tensors).
  """
  scenes = list_scenes_in_split(root_dir, split_name)
  dataset = IGibsonSceneDataset(root_dir, split_name, scenes,
                                add_images=True, add_rasters=True, add_lidar_rays=False)
  # Optionally wrap dataset with a map transform applying process_example_fn per example.
  if process_example_fn is not None:
    class MappedDataset(Dataset):
      def __init__(self, base_ds, fn):
        self.base = base_ds
        self.fn = fn
      def __len__(self): return len(self.base)
      def __getitem__(self, i):
        ex = self.base[i]
        return self.fn(ex)
    ds = MappedDataset(dataset, process_example_fn)
  else:
    ds = dataset

  loader = DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=is_training,
                      num_workers=num_workers,
                      collate_fn=_numpy_collate,
                      drop_last=drop_last,
                      pin_memory=True)

  return _dataloader_iterator(loader, process_batch_fn)


# ---------------------------------------------------------------------
# Main entry: get_dataset (keeps same signature as original)
# ---------------------------------------------------------------------
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    num_shards: int,  # ignored in PyTorch loader; kept for API compatibility
    dataset_configs: types.ProcessingConfig,
    dtype_str: str = 'float32',
    shuffle_seed: int = 0,
    rng: Optional[np.ndarray] = None,
    dataset_service_address: Optional[str] = None,
) -> DatasetTuple:
  """Returns generators for the train and validation sets using PyTorch DataLoader.

  Args:
    batch_size: training batch size
    eval_batch_size: evaluation batch size
    num_shards: kept for API compatibility; PyTorch data parallelism handled by DataLoader/Lightning
    dataset_configs: ProcessingConfig-like object or dict describing locations etc.
  """
  del rng
  assert dataset_service_address is None
  assert dataset_configs is not None

  # dataset_configs is expected to have 'data_path' or 'data_dir' and split names in `locations`.
  root_dir = getattr(dataset_configs, 'data_path', None) or getattr(dataset_configs, 'data_dir', None)
  if root_dir is None:
    raise ValueError('dataset_configs must specify data_path or data_dir pointing to dataset root.')

  # Determine train and eval split names (e.g. gibson_f/gibson_g/gibson_t)
  train_location = dataset_configs.locations.training
  eval_location = dataset_configs.locations.evaluation or train_location

  print('Loading train split %s from %s', train_location, root_dir)
  # Create iterators
  example_fn = functools.partial(process_example, config=dataset_configs, dtype=np.float32)
  batch_fn = functools.partial(process_batch, config=dataset_configs)

  train_iter = None
  training_size = 0
  try:
    train_iter = dataset_iterator_from_folder(root_dir, train_location,
                                              batch_size=batch_size,
                                              is_training=True,
                                              process_example_fn=example_fn,
                                              process_batch_fn=batch_fn,
                                              shuffle_seed=shuffle_seed,
                                              num_workers=dataset_configs.num_workers if hasattr(dataset_configs, 'num_workers') else 4,
                                              drop_last=True)
    # Estimate training size by counting frames
    scenes = list_scenes_in_split(root_dir, train_location)
    training_size = sum(len(os.listdir(os.path.join(root_dir, train_location, s, 'rgb')))
                        for s in scenes if os.path.isdir(os.path.join(root_dir, train_location, s, 'rgb')))
  except Exception as e:
    print('Could not create train iterator: %s', e)
    train_iter = None
    training_size = 0

  print('Loading eval split %s from %s', eval_location, root_dir)
  eval_iter = None
  evaluation_size = 0
  try:
    eval_iter = dataset_iterator_from_folder(root_dir, eval_location,
                                             batch_size=eval_batch_size,
                                             is_training=False,
                                             process_example_fn=example_fn,
                                             process_batch_fn=batch_fn,
                                             shuffle_seed=shuffle_seed,
                                             num_workers=dataset_configs.num_workers if hasattr(dataset_configs, 'num_workers') else 4,
                                             drop_last=False)
    scenes_eval = list_scenes_in_split(root_dir, eval_location)
    evaluation_size = sum(len(os.listdir(os.path.join(root_dir, eval_location, s, 'rgb')))
                          for s in scenes_eval if os.path.isdir(os.path.join(root_dir, eval_location, s, 'rgb')))
  except Exception as e:
    print('Could not create eval iterator: %s', e)
    eval_iter = None
    evaluation_size = 0

  # Dummy batch function for model init: create single batch by pulling one from train_iter (if available).
  def get_dummy_batch(batch_size_local=batch_size):
    if train_iter is None:
      raise RuntimeError('No train iterator available for dummy batch.')
    return next(train_iter)

  # Reconstruct grid and other metadata from config if possible.
  voxel_size = getattr(dataset_configs, 'voxel_size', 0.1)
  grid_size_meters = getattr(dataset_configs, 'scene_config', types.SceneConfig()).grid_size
  grid = grids.Grid3D.from_extent_meters(grid_size_meters, voxel_size)

  meta_data = {
      'grid': grid,
      'build_config': getattr(dataset_configs, 'build_config', None),
      'grid_size_meters': grid_size_meters,
      'num_train_examples': training_size,
      'num_eval_examples': evaluation_size,
      'get_dummy_batch_fn': get_dummy_batch,
      # semantic map classes: try to read from dataset_configs or leave empty list
      'semantic_map_classes': getattr(dataset_configs, 'semantic_map_classes', []),
      'semantic_classes_gt': getattr(dataset_configs, 'semantic_classes_gt', []),
  }

  return DatasetTuple(train_iter, eval_iter, None, meta_data)
