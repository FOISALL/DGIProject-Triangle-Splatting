"""
COLMAP Data Loader
==================
Loads cameras and images from the south-building sparse reconstruction.

Files read:
  sparse/cameras.txt  -- intrinsics (focal length, principal point)
  sparse/images.txt   -- extrinsics (rotation quaternion + translation)
  images/             -- the actual JPEGs used as training targets

The COLMAP convention for a camera is:
    x_cam = R @ x_world + t

where R is the rotation matrix built from (QW, QX, QY, QZ) and
t = (TX, TY, TZ) is the translation vector.

Your renderer currently uses:
    x_cam = (x_world - cam_pos) @ R_renderer

These are the same transformation written differently.
The conversion is:
    R_renderer  = R_colmap.T          (transpose, because numpy matmul is row-major)
    cam_pos     = -R_colmap.T @ t     (camera centre in world space)

This file provides:
  - load_cameras()   -> dict[int, CameraIntrinsics]
  - load_images()    -> list[TrainingView]
  - load_target_image(view, W, H) -> np.ndarray (H, W, 3) float32 in [0,255]
"""

import os
import struct
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    camera_id: int
    width:     int
    height:    int
    fx:        float   # focal length in pixels
    cx:        float   # principal point x
    cy:        float   # principal point y
    k1:        float   # radial distortion (we ignore this for now)


@dataclass
class TrainingView:
    image_id:   int
    filename:   str          # e.g. "P1180189.JPG"
    R:          np.ndarray   # (3, 3) float32 -- renderer convention (R_colmap.T)
    cam_pos:    np.ndarray   # (3,)   float32 -- camera centre in world space
    camera_id:  int


# ---------------------------------------------------------------------------
# Quaternion -> rotation matrix
# ---------------------------------------------------------------------------

def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert a unit quaternion to a 3x3 rotation matrix.
    This is the COLMAP convention rotation: x_cam = R_colmap @ x_world + t
    """
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)
    return R


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_cameras(cameras_path: str) -> dict[int, CameraIntrinsics]:
    """
    Parse sparse/cameras.txt.
    Handles SIMPLE_RADIAL model (the one your scene uses).
    """
    cameras = {}
    with open(cameras_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model  = parts[1]
            width  = int(parts[2])
            height = int(parts[3])

            if model == "SIMPLE_RADIAL":
                # PARAMS: f, cx, cy, k1
                fx = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])
                k1 = float(parts[7])
            elif model == "PINHOLE":
                # PARAMS: fx, fy, cx, cy  (no distortion)
                fx = float(parts[4])
                # fy = float(parts[5])  # ignore fy, use fx
                cx = float(parts[6])
                cy = float(parts[7])
                k1 = 0.0
            else:
                # Fallback: just grab the first param as focal length
                fx = float(parts[4])
                cx = width  / 2.0
                cy = height / 2.0
                k1 = 0.0

            cameras[cam_id] = CameraIntrinsics(cam_id, width, height, fx, cx, cy, k1)

    print(f"Loaded {len(cameras)} camera(s).")
    return cameras


def load_images(images_path: str) -> list[TrainingView]:
    """
    Parse sparse/images.txt.
    Returns one TrainingView per image, with R and cam_pos in renderer convention.
    """
    views = []
    with open(images_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has two lines per image; step by 2
    i = 0
    while i < len(lines):
        parts = lines[i].split()

        image_id  = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz      = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        filename  = parts[9]

        # COLMAP rotation matrix: x_cam = R_colmap @ x_world + t
        R_colmap = quat_to_rotmat(qw, qx, qy, qz)   # (3, 3)
        t        = np.array([tx, ty, tz], dtype=np.float32)

        # Convert to renderer convention:
        #   x_cam = R_colmap @ x_world + t
        #         = R_colmap @ (x_world - cam_pos)   where cam_pos = -R_colmap.T @ t
        #         = (x_world - cam_pos) @ R_colmap.T  (transpose for row-vector convention)
        # So:
        R_renderer = R_colmap.T                          # (3, 3)
        cam_pos    = -R_colmap.T @ t                     # (3,)

        views.append(TrainingView(
            image_id  = image_id,
            filename  = filename,
            R         = R_renderer,
            cam_pos   = cam_pos,
            camera_id = camera_id,
        ))

        i += 2   # skip the keypoint line

    print(f"Loaded {len(views)} training view(s).")
    return views


def load_target_image(
    view:       TrainingView,
    images_dir: str,
    out_W:      int,
    out_H:      int,
) -> np.ndarray:
    """
    Load the JPEG for a TrainingView, resize to (out_H, out_W),
    and return as float32 array in [0, 255] with shape (out_H, out_W, 3).

    Requires PIL (pip install pillow).
    """
    from PIL import Image

    path = os.path.join(images_dir, view.filename)
    img  = Image.open(path).convert("RGB")
    img  = img.resize((out_W, out_H), Image.LANCZOS)
    arr  = np.array(img, dtype=np.float32)   # (H, W, 3) in [0, 255]
    return arr


# ---------------------------------------------------------------------------
# Convenience: build a training dataset
# ---------------------------------------------------------------------------

def load_dataset(
    sparse_dir:  str,
    images_dir:  str,
    screen_W:    int,
    screen_H:    int,
    every_nth:   int = 8,   # hold out every 8th image for test (same as 3DGS paper)
) -> tuple[list[TrainingView], list[TrainingView], CameraIntrinsics]:
    """
    Load cameras and image metadata, split into train/test.

    Returns
    -------
    train_views : list[TrainingView]
    test_views  : list[TrainingView]
    camera      : CameraIntrinsics  (the single camera in this scene)
    """
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path  = os.path.join(sparse_dir, "images.txt")

    cameras = load_cameras(cameras_path)
    views   = load_images(images_path)

    # Sort by image_id for reproducible train/test split
    views.sort(key=lambda v: v.image_id)

    train_views = [v for i, v in enumerate(views) if i % every_nth != 0]
    test_views  = [v for i, v in enumerate(views) if i % every_nth == 0]

    print(f"Train: {len(train_views)} views  |  Test: {len(test_views)} views")

    # Your scene has one camera
    camera = list(cameras.values())[0]
    return train_views, test_views, camera


# ---------------------------------------------------------------------------
# Focal length adapter
# ---------------------------------------------------------------------------

def colmap_focal_to_renderer(camera: CameraIntrinsics, out_W: int, out_H: int) -> float:
    """
    The COLMAP focal length is given in pixels at the original image resolution.
    If you render at a different resolution (out_W x out_H), scale it.
    The south-building camera is 3072x2304 with fx=2559.68.
    """
    scale = out_W / camera.width
    return camera.fx * scale


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sparse_dir = "south-building/sparse"
    images_dir = "south-building/images"

    W, H = 1200, 800

    train_views, test_views, camera = load_dataset(sparse_dir, images_dir, W, H)
    focal = colmap_focal_to_renderer(camera, W, H)

    print(f"\nCamera intrinsics:")
    print(f"  Original resolution : {camera.width} x {camera.height}")
    print(f"  Focal length (orig) : {camera.fx:.2f} px")
    print(f"  Focal length (scaled to {W}x{H}): {focal:.2f} px")
    print(f"  Principal point     : ({camera.cx:.1f}, {camera.cy:.1f})")

    print(f"\nFirst training view:")
    v = train_views[0]
    print(f"  File     : {v.filename}")
    print(f"  cam_pos  : {v.cam_pos}")
    print(f"  R[0]     : {v.R[0]}")

    # Try loading the actual image
    try:
        img = load_target_image(v, images_dir, W, H)
        print(f"  Image shape: {img.shape}  dtype: {img.dtype}  range: [{img.min():.0f}, {img.max():.0f}]")
    except FileNotFoundError:
        print(f"  (Image file not found at {images_dir}/{v.filename} -- metadata loaded fine)")