# ----------------------------------------------------------------------
#  grids.py — PyTorch/Numpy port of SNAP's grid utilities
# ----------------------------------------------------------------------

import dataclasses
import numpy as np
from typing import Optional, Tuple, TypeVar, Type, Union

try:
    # Preferred: SciPy map_coordinates (same as original SNAP)
    from scipy.ndimage import map_coordinates
    _SCIPY_AVAILABLE = True
except ImportError:
    # Fallback: simple nearest-neighbor
    _SCIPY_AVAILABLE = False


Point = np.ndarray     # (..., N)
Index = np.ndarray     # (..., N)
ID = np.ndarray
AnyGrid = TypeVar("AnyGrid", bound="GridND")


# ======================================================================
#                               GridND
# ======================================================================

@dataclasses.dataclass(frozen=True)
class GridND:
    """
    N-dimensional regular grid.

    extent: tuple of ints (number of cells along each dimension)
    cell_size: float (physical meters per cell)
    """
    extent: Tuple[int, ...]
    cell_size: float

    # --------------------------------------------------------------

    @classmethod
    def from_extent_meters(
        cls: Type[AnyGrid],
        extent_meters: Tuple[float, ...],
        cell_size: float
    ) -> AnyGrid:
        extent = tuple(e / cell_size for e in extent_meters)

        if not all(abs(e - round(e)) < 1e-6 for e in extent):
            raise ValueError(
                f"Metric grid extent {extent_meters} is not divisible "
                f"by cell size {cell_size}"
            )

        return cls(tuple(int(round(e)) for e in extent), cell_size)

    # --------------------------------------------------------------

    def xyz_to_index(self, xyz: Point) -> Index:
        """Convert metric coordinates to integer grid index."""
        return np.floor(xyz / self.cell_size).astype(np.int32)

    def index_to_xyz(self, idx: Index) -> Point:
        """Convert grid index to metric coordinates (cell center)."""
        return (idx + 0.5) * self.cell_size

    # --------------------------------------------------------------

    def index_to_id(self, idx: Index) -> ID:
        """Flatten multi-index to single linear index."""
        idx = np.moveaxis(idx, -1, 0)
        return np.ravel_multi_index(idx, self.extent, mode="clip")

    def id_to_index(self, ids: ID) -> Index:
        """Convert flattened index back to multi-axis index."""
        return np.stack(np.unravel_index(ids, self.extent), axis=-1)

    # --------------------------------------------------------------

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.extent))

    @property
    def extent_meters(self) -> np.ndarray:
        return np.asarray(self.extent, dtype=np.float32) * self.cell_size

    # --------------------------------------------------------------

    def index_in_grid(self, idx: Index) -> np.ndarray:
        idx = np.asarray(idx)
        return np.logical_and(idx >= 0, idx < np.asarray(self.extent)).all(axis=-1)

    def xyz_in_grid(self, xyz: Point) -> np.ndarray:
        xyz = np.asarray(xyz)
        return np.logical_and(xyz >= 0, xyz < self.extent_meters).all(axis=-1)

    # --------------------------------------------------------------

    def grid_index(self) -> np.ndarray:
        """Returns array of indices with shape (e1, e2, ..., eN, N)."""
        grid = np.meshgrid(
            *[np.arange(e) for e in self.extent],
            indexing="ij"
        )
        return np.stack(grid, axis=-1).astype(np.int32)


# ======================================================================
#                               Grid2D
# ======================================================================

@dataclasses.dataclass(frozen=True)
class Grid2D(GridND):
    extent: Tuple[int, int]


# ======================================================================
#                               Grid3D
# ======================================================================

@dataclasses.dataclass(frozen=True)
class Grid3D(GridND):
    extent: Tuple[int, int, int]

    def bev(self) -> Grid2D:
        """Birds-eye-view grid (drops Z)."""
        return Grid2D(self.extent[:2], self.cell_size)


# ======================================================================
#                     N–Dimensional Interpolation
# ======================================================================

def interpolate_nd(
    array: np.ndarray,
    points: np.ndarray,
    valid_array: Optional[np.ndarray] = None,
    order: int = 1,
    mode: str = "nearest",
):
    """
    Interpolate an N-D grid-valued array at floating point `points`.

    array: (..., D) where D = number of channels
    points: (K, N) with N dimensional coordinates

    Returns:
        values: (K, D)
        valid:  (K,) boolean
    """

    array = np.asarray(array)
    points = np.asarray(points)

    dims = array.shape[:-1]
    D = array.shape[-1]
    K = points.shape[0]

    # --- validity: coordinates in range -----------------------------
    valid = np.logical_and(points >= 0, points < np.asarray(dims)).all(axis=-1)

    # SNAP indexing: center of voxel is at +0.5 offset
    pts = (points - 0.5).T  # shape (N, K)

    # --------------------------------------------------------------
    # Interpolate each channel independently (same as JAX vmap)
    # --------------------------------------------------------------

    if _SCIPY_AVAILABLE:
        values = np.zeros((K, D), dtype=array.dtype)
        for ch in range(D):
            values[:, ch] = map_coordinates(
                array[..., ch], pts,
                order=order,
                mode=mode,
                cval=np.nan
            )
    else:
        # Fallback = nearest neighbor
        nn_idx = np.clip(np.round(points).astype(np.int32), 0, np.asarray(dims) - 1)
        values = array[tuple(nn_idx.T)]

    # --------------------------------------------------------------
    # Masking using valid_array (same as original SNAP)
    # --------------------------------------------------------------

    if valid_array is not None:
        valid_array = np.asarray(valid_array, dtype=bool)

        if _SCIPY_AVAILABLE:
            nan_mask = np.where(valid_array, 0.0, np.nan)
            nan_pts = map_coordinates(
                nan_mask, pts,
                order=order,
                mode=mode,
                cval=np.nan
            )
            valid &= ~np.isnan(nan_pts)
        else:
            # Nearest-neighbor fallback
            nn_idx = np.clip(np.round(points).astype(np.int32), 0, np.asarray(dims) - 1)
            valid &= valid_array[tuple(nn_idx.T)]

    return values, valid


# ======================================================================
#                    argmax_nd + expectation_nd
# ======================================================================

def argmax_nd(scores: np.ndarray, grid: GridND) -> Index:
    """
    argmax over last N grid dims.
    Returns index with shape (..., N)
    """
    n = len(grid.extent)
    flat = scores.reshape(*scores.shape[:-n], -1)
    idx = np.argmax(flat, axis=-1)
    return grid.id_to_index(idx)


def expectation_nd(pdf: np.ndarray, grid: GridND) -> np.ndarray:
    """
    Expected value (index expectation) over last N dims.
    pdf must sum to 1 over grid dimensions.
    """
    grid_idx = grid.grid_index()  # shape (..., N)
    # Sum over N last axes
    for _ in range(len(grid.extent)):
        pdf = np.expand_dims(pdf, axis=-1)

    # Reduce over grid dims:
    axes = tuple(range(len(grid.extent), len(grid.extent) * 2))
    return np.sum(grid_idx * pdf, axis=axes)
