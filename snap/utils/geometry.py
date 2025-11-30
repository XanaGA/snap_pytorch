# Copyright 2023 Google LLC
#
# Rewritten for PyTorch/Numpy (no JAX, no dataclass_array)
#
# This version provides:
#   - Transform3D
#   - Transform2D
#   - Camera
#   - FisheyeCamera
#
# Fully compatible with the new PyTorch-based SNAP loader + dataset.


import numpy as np
from typing import Any, Dict, Tuple, Union


# ----------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------
Points2D = np.ndarray
Points3D = np.ndarray
RotationMatrix3D = np.ndarray
RotationMatrix2D = np.ndarray
Angle = Union[float, np.ndarray]
Point3D = np.ndarray
Point2D = np.ndarray


# ======================================================================
#                           Transform3D
# ======================================================================

class Transform3D:
    """
    SE(3) transformation with rotation R (3×3) and translation t (3×1).

    Wrapper compatible with original SNAP API:
        - from_Rt
        - to_4x4matrix
        - inv
        - magnitude
        - transform
        - compose
        - matmul over points or transforms
    """

    def __init__(self, R: RotationMatrix3D, t: Point3D):
        self.R = np.asarray(R, dtype=np.float32)
        self.t = np.asarray(t, dtype=np.float32)

    @classmethod
    def from_Rt(cls, R, t):
        return cls(R, t)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """
        Accepts dicts produced by the PyTorch loader:
            {
                'rotation': 3x3 matrix (numpy),
                'translation': (3,)
            }
        """
        R = np.asarray(d["rotation"], dtype=np.float32)
        t = np.asarray(d["translation"], dtype=np.float32)
        return cls(R, t)

    # --------------------------------------------------------------

    def to_4x4matrix(self) -> np.ndarray:
        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    # --------------------------------------------------------------

    @property
    def inv(self):
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform3D(R_inv, t_inv)

    # --------------------------------------------------------------

    def magnitude(self) -> Tuple[float, float]:
        """
        Computes (rotation_deg, translation_norm).
        """
        trace = np.trace(self.R)
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        dr = np.degrees(abs(np.arccos(cos_angle)))
        dt = np.linalg.norm(self.t)
        return dr, dt

    # --------------------------------------------------------------

    def transform(self, p3d: Points3D) -> Points3D:
        """
        p' = R p + t
        """
        return (self.R @ p3d.T).T + self.t

    # --------------------------------------------------------------

    def compose(self, other: "Transform3D") -> "Transform3D":
        """
        this ∘ other  (applies other, then this)
        """
        R = self.R @ other.R
        t = self.R @ other.t + self.t
        return Transform3D(R, t)

    # --------------------------------------------------------------

    def __matmul__(self, other: Union[Points3D, "Transform3D"]):
        if isinstance(other, np.ndarray):
            return self.transform(other)
        elif isinstance(other, Transform3D):
            return self.compose(other)
        raise TypeError(f'Unsupported @ operand {type(other)}')


# ======================================================================
#                           Transform2D
# ======================================================================

class Transform2D:
    """
    2D version of SE(2).
    angle: rotation in radians
    t: 2D translation
    """

    def __init__(self, angle: float, t: Point2D):
        self.angle = float(angle)
        self.t = np.asarray(t, dtype=np.float32)

    @classmethod
    def from_radians(cls, angle, t):
        return cls(angle, t)

    @classmethod
    def from_R(cls, R: RotationMatrix2D, t: Point2D):
        angle = np.arctan2(R[1, 0], R[0, 0])
        return cls(angle, t)

    @classmethod
    def from_Transform3D(cls, transform: Transform3D):
        R2 = transform.R[:2, :2]
        t2 = transform.t[:2]
        return cls.from_R(R2, t2)

    # --------------------------------------------------------------

    @property
    def R(self) -> RotationMatrix2D:
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    # --------------------------------------------------------------

    def to_3x3matrix(self):
        M = np.eye(3, dtype=np.float32)
        M[:2, :2] = self.R
        M[:2, 2] = self.t
        return M

    # --------------------------------------------------------------

    @property
    def inv(self):
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform2D(-self.angle, t_inv)

    # --------------------------------------------------------------

    def magnitude(self) -> Tuple[float, float]:
        dr = abs(np.degrees(self.angle)) % 360
        dr = min(dr, 360 - dr)
        dt = np.linalg.norm(self.t)
        return dr, dt

    # --------------------------------------------------------------

    def transform(self, points: Points2D) -> Points2D:
        return (self.R @ points.T).T + self.t

    # --------------------------------------------------------------

    def compose(self, other: "Transform2D"):
        R = self.R @ other.R
        angle = np.arctan2(R[1, 0], R[0, 0])
        t = self.R @ other.t + self.t
        return Transform2D(angle, t)

    # --------------------------------------------------------------

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return self.transform(other)
        elif isinstance(other, Transform2D):
            return self.compose(other)
        raise TypeError(f'Unsupported @ operand {type(other)}')


# ======================================================================
#                               Camera
# ======================================================================

class Camera:
    """
    Simple pinhole camera:
      wh = (width, height)
      f  = focal lengths (fx, fy)
      c  = principal point (cx, cy)
    """

    def __init__(self, wh, f, c):
        self.wh = np.asarray(wh, dtype=np.float32)
        self.f = np.asarray(f, dtype=np.float32)
        self.c = np.asarray(c, dtype=np.float32)
        self.eps = 1e-3

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """
        Expected dict (your loader produces this format now):
            intrinsics = {
                'K': 3x3 numpy array,
                'width': w,
                'height': h
            }
        """
        K = np.asarray(d["K"], dtype=np.float32)
        w, h = d["width"], d["height"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        return cls(wh=np.array([w, h]), f=np.array([fx, fy]), c=np.array([cx, cy]))

    # --------------------------------------------------------------

    def K(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.f[0]
        K[1, 1] = self.f[1]
        K[0, 2] = self.c[0]
        K[1, 2] = self.c[1]
        return K

    # --------------------------------------------------------------

    def project(self, p3d: Points3D) -> Tuple[Points2D, np.ndarray]:
        z = p3d[:, 2]
        valid = z >= self.eps
        z = np.maximum(z, self.eps)
        p2d = p3d[:, :2] / z[:, None]
        return p2d, valid

    # --------------------------------------------------------------

    def denormalize(self, p2d: Points2D) -> Points2D:
        return p2d * self.f[None, :] + self.c[None, :]

    # --------------------------------------------------------------

    def normalize(self, p2d: Points2D) -> Points2D:
        return (p2d - self.c[None, :]) / self.f[None, :]

    # --------------------------------------------------------------

    def in_image(self, p2d: Points2D) -> np.ndarray:
        return (
            (p2d[:, 0] >= 0)
            & (p2d[:, 1] >= 0)
            & (p2d[:, 0] < self.wh[0])
            & (p2d[:, 1] < self.wh[1])
        )

    # --------------------------------------------------------------

    def world2image(self, p3d: Points3D):
        p2d, valid = self.project(p3d)
        p2d = self.denormalize(p2d)
        valid = valid & self.in_image(p2d)
        return p2d, valid


# ======================================================================
#                           FisheyeCamera
# ======================================================================

class FisheyeCamera(Camera):
    """
    Simplified fisheye camera matching SNAP interface.

    NOTE:
    iGibson has standard pinhole intrinsics; radial distortion defaults to 0.
    """

    def __init__(self, wh, f, c, k_radial=None, max_fov=np.radians(115.0)):
        super().__init__(wh, f, c)
        if k_radial is None:
            k_radial = np.zeros(3, dtype=np.float32)
        self.k_radial = np.asarray(k_radial, dtype=np.float32)
        self.max_fov = float(max_fov)

    @classmethod
    def from_dict(cls, intrinsics: Dict[str, Any]):
        K = np.asarray(intrinsics["K"], dtype=np.float32)
        w = intrinsics["width"]
        h = intrinsics["height"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Optional distortion
        distortion = intrinsics.get("distortion", {})
        k = distortion.get("radial", np.zeros(3, dtype=np.float32))

        max_fov = intrinsics.get("maxfov", np.radians(115.0))

        return cls(
            wh=np.array([w, h]),
            f=np.array([fx, fy]),
            c=np.array([cx, cy]),
            k_radial=k,
            max_fov=max_fov,
        )

    # --------------------------------------------------------------

    def distort_points(self, p2d: Points2D):
        """
        Fisheye radial distortion, same formula as original JAX version.
        """
        radius2 = np.sum(p2d**2, axis=1)
        radius = np.sqrt(np.maximum(radius2, self.eps**2))

        theta = np.arctan(radius)
        theta2 = theta ** 2

        offset = (
            self.k_radial[0] * theta2
            + self.k_radial[1] * theta2**2
            + self.k_radial[2] * theta2**3
        )

        dist = (offset + 1) * theta / radius
        p2d_dist = p2d * dist[:, None]

        valid = radius < np.tan(0.5 * self.max_fov)
        return p2d_dist, valid

    # --------------------------------------------------------------

    def world2image(self, p3d: Points3D):
        p2d, valid_p = self.project(p3d)
        p2d, valid_d = self.distort_points(p2d)
        p2d = self.denormalize(p2d)
        valid = valid_p & valid_d & self.in_image(p2d)
        return p2d, valid
