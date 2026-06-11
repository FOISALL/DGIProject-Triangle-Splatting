"""
Differentiable Triangle Splatting Renderer  (complete final version)
=====================================================================
All features in one file:
  - CONFIG block at top — change numbers here, nothing else needs touching
  - Differentiable forward pass (vertices, colors, opacity, sigma)
  - L1 + D-SSIM + opacity loss Lo + size loss Ls   (paper Eq. 3)
  - Midpoint-subdivision densification              (paper Sec. 3.2)
  - Visibility-based pruning                        (paper Sec. 3.2)
  - Exponential LR decay for vertices               (paper training schedule)
  - Batched GPU depth sort                          (fast, no per-triangle CPU transfer)
  - Stable interactive preview camera               (no jumping)
  - Step timer                                      (steps/s, eta per 10 steps)
  - Checkpoint save/load                            (resume training across runs)
  - PNG export every N steps                        (review progress without pygame)
  - Multi-chunk training                            (cycles through spatial chunks)
  - Ctrl+S manual checkpoint save
  - Auto-save checkpoint and final PNG on exit
"""

import math
import os
import time

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from scipy.spatial import KDTree

from render_utils import Pixel, Point3D, Triangle, load_points3D
from colmap_loader import load_dataset, colmap_focal_to_renderer, load_target_image

# Safe GPU speed settings. These do not change model structure.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ===========================================================================
# CONFIG  — only change things here
# ===========================================================================

CFG = {
    # --- scene paths ---
    "sparse_dir":  "south-building/sparse",
    "images_dir":  "south-building/images",
    "points_file": "south-building/sparse/points3D.txt",

    # --- reproducibility ---
    # Fixes the random triangle directions at initialization. This does not
    # change a loaded checkpoint, but it makes fresh runs easier to compare.
    "seed": 1234,

    # --- point count ---
    # Controls startup time and scene coverage. One point = one triangle.
    #   100   → instant start, tiny test
    #   1000  → a few seconds, check gradients
    #   5000  → ~30s startup, recognisable patch
    #   61000 → full scene, slow startup
    "max_points": 5000,

    # --- chunk size ---
    # How many of the loaded points form one active training chunk.
    # Larger = more coverage per step but slower steps.
    "chunk_k": 2800,

    # --- speed knobs ---
    # Training at 1200x800 is extremely expensive in a Python renderer.
    # Start low; increase after the renderer is working.
    "screen_w":         640,
    "screen_h":         426,

    # Skip the SSIM term for the first steps. L1 is much cheaper and good
    # enough for early geometry/color discovery. Set 0 to always use SSIM.
    "ssim_start_step":  1600,

    # Load/resize each training image once instead of doing disk I/O every step.
    # Values: "cpu", "gpu", or "off". GPU is fastest but uses VRAM.
    "target_cache":     "cpu",

    # Pygame preview is another full render. During training, keep it off or rare.
    "preview_during_training": False,

    # --- chunk-focused crop training ---
    # This is the key Option-C change: the active chunk is trained only against
    # the image region where that chunk projects, instead of being punished for
    # not reconstructing the whole training photo.
    "use_chunk_crop": True,

    # Before entering the Python render loop, remove triangles whose projected
    # bbox does not touch the active crop. This keeps the math the same but
    # avoids looping over triangles that cannot affect the cropped loss.
    "use_crop_culling": True,

    # Avoid infinite loops if many views stop producing a valid crop.
    "max_consecutive_crop_skips": 80,
    "fallback_to_central_crop_after_skips": False,

    # Compute the crop from the original chunk geometry instead of the current
    # optimized model. This keeps the target region stable if vertices drift.
    "use_static_crop_reference": True,

    # Extra pixels around the projected chunk bounding box. Increase if the crop
    # is too tight; decrease for more speed.
    "crop_margin": 25,

    # Reject crops smaller than this. If a chunk barely appears in a view, the
    # step is skipped rather than using a noisy tiny target.
    "crop_min_size": 32,

    # Hard cap so a bad/wide projection does not accidentally become full-frame.
    # If the crop is larger than this fraction of the full image, it is shrunk
    # around its center. 0.50 means at most half the full image area.
    "crop_max_area_frac": 0.15,

    # --- crop debugging ---
    # Saves images that show where the active chunk crop lands in each target.
    # Use this for a short test run before committing to a long run.
    "crop_debug_enabled": False,
    "crop_debug_every": 50,
    "crop_debug_dir": "crop_debug",
    "crop_debug_save_rectangle": True,
    "crop_debug_save_target_crop": True,

    # --- optional training-view whitelist ---
    # Put COPIES of the good training images in this folder. The code uses
    # only the filenames as a whitelist, while still loading targets from
    # CFG["images_dir"]. This lets you select views where the chunk is visible.
    "use_goodimages_whitelist": True,
    "goodimages_dir": "south-building/goodimages",

    # Optional: stop training automatically. Use None for infinite interactive mode.
    "max_train_steps": None,

    # --- multi-chunk ---
    # 1  = train one fixed chunk (original behaviour)
    # >1 = cycle through this many spatially distinct chunks,
    #      adding each chunk's triangles to the model as training progresses
    "n_chunks": 1,

    # --- densification / pruning ---
    # Debug switches: disable these independently when isolating crashes.
    # Keeping the interval nonzero avoids modulo-by-zero mistakes.
    "enable_densification": True,
    "enable_pruning": True,
    "densify_every":  2300,
    "prune_every":    400,
    "max_triangles":  3000,   # hard cap; raise for fuller reconstruction
    "tau_prune":      0.004,

    # --- learning rates ---
    "lr_color_opacity_sigma": 6e-4,
    "lr_vertices_init":       3e-4,   # vertex LR at step 0
    "lr_vertices_final":      7e-5,   # vertex LR after decay_steps
    "lr_decay_steps":         8000,

    # Keep optimized triangles near their starting chunk geometry. Without this
    # the crop can remain valid while the trainable triangles drift out of it.
    "vertex_anchor_weight":   0.05,
    "vertex_anchor_max_offset": 0.4,
    "max_abs_logit": 10.0,
    "min_log_sigma": 0.0,
    "max_log_sigma": 4.0,
    "reset_optimizer_on_load": True,

    # If too many views in a row cannot render any triangles, stop the runaway
    # skip loop and print state diagnostics.
    "max_consecutive_render_skips": 80,

    # Numerical guards for the screen-space triangle SDF. Very small/edge-on
    # projected triangles can produce enormous or non-finite gradients.
    "min_projected_bbox_px": 2,
    "sdf_phi_eps": 1e-3,
    "projection_min_depth": 0.05,
    "sdf_normal_eps": 1e-2,
    "render_ratio_eps": 1e-4,
    "max_consecutive_rejected_updates": 5,
    "rejected_update_lr_decay": 0.5,
    "min_recovery_vertex_lr": 5e-5,

    # --- display ---
    "display_every":   200,    # pygame refresh every N training steps

    # --- saving ---
    "checkpoint_path":  "checkpoint_renderer3_stable_debug.pt",   # auto-saved every save_every steps
    "save_every":        100,              # checkpoint interval (steps)
    "save_image_every":  100,              # PNG export interval (steps)
    "image_dir":         "renders_renderer3_debug",        # folder for PNG exports

    # --- training-camera render exports ---
    # These are more useful than the interactive preview render because they
    # use the same COLMAP camera angle as the current training loss.
    "save_training_view_every": 100,
    "training_render_dir": "renders_renderer3_training_views",
    "save_training_full_view": False,
    "save_training_crop_compare": True,
}

# ===========================================================================


# Apply deterministic seeds for reproducible fresh initializations.
# Checkpoint loading still takes precedence over these initial random values.
SEED = int(CFG.get("seed", 1234))
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Globals  (camera state, not trained)
# ---------------------------------------------------------------------------

class Globals:
    SCREEN_WIDTH  = CFG["screen_w"]
    SCREEN_HEIGHT = CFG["screen_h"]
    focalLength   = 500.0
    pointcloudData: list[Point3D] = []
    cameraPosition = np.array([-3.40018177, -0.90000010, -4.63711214], dtype=np.float32)
    R = np.array([
        [0.79608387, 0.00000000, 0.60518640],
        [0.00000000, 1.00000000, 0.00000000],
        [-0.60518640, 0.00000000, 0.79608381],
    ], dtype=np.float32)
    delta          = 0.1
    yaw_speed      = 0.05
    pitch_speed    = 0.05
    show_debug     = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Triangle initialisation
# ---------------------------------------------------------------------------

def initialize_triangle(point: Point3D, neighbours) -> Triangle:
    q = np.array([point.x, point.y, point.z], dtype=np.float32)
    U = []
    for _ in range(3):
        u = np.random.uniform(-1.0, 1.0, 3)
        U.append(u / (np.linalg.norm(u) + 1e-8))
    d = (neighbours[0][1] + neighbours[1][1] + neighbours[2][1]) / 3.0
    verts = [q + 2.2 * d * u for u in U]
    return Triangle(
        vertices=verts,
        color=np.array([point.r, point.g, point.b]),
        opacity=0.9,
        sigma=1.16,
    )


def initialize_triangles(points: list[Point3D]) -> list[Triangle]:
    print("Initializing triangles...")
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    kdtree = KDTree(coords)
    triangles = []
    for point in points:
        distances, indices = kdtree.query([point.x, point.y, point.z], k=4)
        neighbours = [(points[idx], dist)
                      for idx, dist in zip(indices[1:], distances[1:])]
        triangles.append(initialize_triangle(point, neighbours))
    print(f"  {len(triangles)} triangles ready")
    return triangles


def get_chunk_indices(points: list[Point3D], k: int, seed_idx: int = 0) -> list[int]:
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    _, indices = KDTree(coords).query(coords[seed_idx], k=min(k, len(points)))
    return list(indices)


def get_all_chunk_indices(points: list[Point3D],
                          chunk_k: int,
                          n_chunks: int) -> list[list[int]]:
    """
    Spread n_chunks seed points evenly through the point list and return
    the chunk_k nearest neighbours of each seed as separate chunks.
    """
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    tree   = KDTree(coords)
    seeds  = np.linspace(0, len(points) - 1, n_chunks, dtype=int)
    chunks = []
    for seed_idx in seeds:
        _, indices = tree.query(coords[seed_idx], k=min(chunk_k, len(points)))
        chunks.append(list(indices))
    return chunks


# ---------------------------------------------------------------------------
# Learnable model
# ---------------------------------------------------------------------------

class TriangleSplatModel(nn.Module):
    def __init__(self, vertices_init, colors_init, opacities_init, sigmas_init):
        super().__init__()
        self.vertices      = nn.Parameter(vertices_init.clone())
        self.color_logit   = nn.Parameter(
            torch.logit(colors_init.clamp(1e-4, 1 - 1e-4)))
        self.opacity_logit = nn.Parameter(
            torch.logit(opacities_init.clamp(1e-4, 1 - 1e-4)))
        self.log_sigma     = nn.Parameter(torch.log(sigmas_init.clamp(min=1e-4)))

    @property
    def colors(self)    -> torch.Tensor: return torch.sigmoid(self.color_logit)
    @property
    def opacities(self) -> torch.Tensor: return torch.sigmoid(self.opacity_logit)
    @property
    def sigmas(self)    -> torch.Tensor: return torch.exp(self.log_sigma)
    def num_triangles(self) -> int:      return self.vertices.shape[0]


def build_model(triangles: list[Triangle],
                indices:   list[int],
                device) -> TriangleSplatModel:
    verts  = np.array([triangles[i].vertices for i in indices], dtype=np.float32)
    colors = np.array([triangles[i].color    for i in indices], dtype=np.float32)
    opacs  = np.array([triangles[i].opacity  for i in indices], dtype=np.float32)
    sigs   = np.array([triangles[i].sigma    for i in indices], dtype=np.float32)
    return TriangleSplatModel(
        torch.from_numpy(verts),
        torch.from_numpy(colors),
        torch.from_numpy(opacs),
        torch.from_numpy(sigs),
    ).to(device)


# ---------------------------------------------------------------------------
# Optimizer + LR scheduler
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(model: TriangleSplatModel,
                                   train_steps_so_far: int = 0):
    """
    Two parameter groups:
      - Vertices: exponential LR decay from lr_vertices_init to lr_vertices_final
      - Color / opacity / sigma: flat LR
    Pass train_steps_so_far when rebuilding after densify/prune so the decay
    continues from the right point rather than restarting.
    """
    optimizer = optim.Adam([
        {"params": model.vertices,      "lr": CFG["lr_vertices_init"]},
        {"params": model.color_logit,   "lr": CFG["lr_color_opacity_sigma"]},
        {"params": model.opacity_logit, "lr": CFG["lr_color_opacity_sigma"]},
        {"params": model.log_sigma,     "lr": CFG["lr_color_opacity_sigma"]},
    ])

    decay_steps = max(CFG["lr_decay_steps"], 1)
    gamma = (CFG["lr_vertices_final"] / CFG["lr_vertices_init"]) ** (1.0 / decay_steps)
    ratio = CFG["lr_vertices_final"] / CFG["lr_vertices_init"]

    def lr_lambda_vertices(step):
        return max(gamma ** (step + train_steps_so_far), ratio)

    def lr_lambda_flat(_step):
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
        lr_lambda_vertices,
        lr_lambda_flat,
        lr_lambda_flat,
        lr_lambda_flat,
    ])
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler,
                    train_steps, total_loss, chunk_indices, path):
    torch.save({
        "train_steps":   train_steps,
        "total_loss":    total_loss,
        "chunk_indices": chunk_indices,
        "vertices":      model.vertices.detach().cpu(),
        "color_logit":   model.color_logit.detach().cpu(),
        "opacity_logit": model.opacity_logit.detach().cpu(),
        "log_sigma":     model.log_sigma.detach().cpu(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
    }, path)
    print(f"[ckpt] Saved {path}  (step {train_steps}, "
          f"N={model.num_triangles()})")


def load_checkpoint(path, device):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    print(f"[ckpt] Loaded {path}  (step {ckpt['train_steps']})")
    return ckpt


def model_from_checkpoint(ckpt, device) -> TriangleSplatModel:
    model = TriangleSplatModel(
        ckpt["vertices"].to(device),
        torch.sigmoid(ckpt["color_logit"].to(device)),
        torch.sigmoid(ckpt["opacity_logit"].to(device)),
        torch.exp(ckpt["log_sigma"].to(device)),
    ).to(device)
    with torch.no_grad():
        model.color_logit.copy_(ckpt["color_logit"].to(device))
        model.opacity_logit.copy_(ckpt["opacity_logit"].to(device))
        model.log_sigma.copy_(ckpt["log_sigma"].to(device))
    clamp_model_parameter_ranges(model)
    return model


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def update_rotation(yaw: float, pitch: float) -> None:
    cy, sy = math.cos(yaw), math.sin(yaw)
    Globals.R = Globals.R @ np.array(
        [[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float32)
    if pitch != 0:
        cp, sp = math.cos(pitch), math.sin(pitch)
        Globals.R = Globals.R @ np.array(
            [[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Batched GPU depth sort
# ---------------------------------------------------------------------------

def sort_triangles_by_depth(vertices: torch.Tensor,
                             cam_pos:  torch.Tensor,
                             R:        torch.Tensor,
                             min_depth: float = 0.1) -> list[int]:
    """
    One matmul on GPU, one argsort, one .tolist() — no per-triangle CPU transfers.
    Returns front-to-back sorted list of triangle indices (ascending z).
    """
    with torch.no_grad():
        centroids = vertices.mean(dim=1)                     # (N, 3)
        p_cam     = (centroids - cam_pos.unsqueeze(0)) @ R  # (N, 3)
        depths    = p_cam[:, 2]                              # (N,)
        mask      = depths > min_depth
        order     = torch.argsort(depths)
        return [int(i) for i in order.tolist() if mask[i]]



# ---------------------------------------------------------------------------
# Crop culling helper
# ---------------------------------------------------------------------------

def filter_sort_order_to_crop(model, cam_pos, R, focal, W, H, sort_order, crop):
    """
    Keep only triangles whose projected screen-space bounding box intersects
    the active crop. This reduces the Python render loop without changing the
    rendering math: triangles outside the crop cannot affect the cropped loss.
    """
    if crop is None or not sort_order:
        return sort_order

    x0, x1, y0, y1 = [int(v) for v in crop]

    with torch.no_grad():
        screen_xy, valid = project_vertices(model.vertices, cam_pos, R, focal, W, H)

        tri_min_x = screen_xy[:, :, 0].min(dim=1).values
        tri_max_x = screen_xy[:, :, 0].max(dim=1).values
        tri_min_y = screen_xy[:, :, 1].min(dim=1).values
        tri_max_y = screen_xy[:, :, 1].max(dim=1).values

        overlaps = (
            valid
            & (tri_max_x >= x0)
            & (tri_min_x < x1)
            & (tri_max_y >= y0)
            & (tri_min_y < y1)
        )

        filtered = [idx for idx in sort_order if bool(overlaps[idx])]

    return filtered


# ---------------------------------------------------------------------------
# Differentiable projection
# ---------------------------------------------------------------------------

def project_vertices(vertices, cam_pos, R, focal, W, H):
    p_cam  = (vertices - cam_pos.unsqueeze(0).unsqueeze(0)) @ R
    z      = p_cam[..., 2]
    min_depth = float(CFG.get("projection_min_depth", 0.05))
    valid  = (z > min_depth).all(dim=1)
    z_safe = z.clamp(min=min_depth)
    sx     = focal * p_cam[..., 0] / z_safe + W / 2.0
    sy     = focal * p_cam[..., 1] / z_safe + H / 2.0
    return torch.stack([sx, sy], dim=-1), valid


# ---------------------------------------------------------------------------
# Differentiable SDF
# ---------------------------------------------------------------------------

def compute_sdf_params(screen_xy):
    v = screen_xy
    normals, offsets = [], []
    for i in range(3):
        a    = v[:, i,       :]
        b    = v[:, (i-1)%3, :]
        edge = b - a
        n    = torch.stack([-edge[:, 1], edge[:, 0]], dim=-1)
        opp  = v[:, (i-2)%3, :]
        dot  = (n * (opp - a)).sum(-1)
        sign = torch.where(dot > 0,
                           -torch.ones_like(dot), torch.ones_like(dot))
        n    = n * sign.unsqueeze(-1)
        n_norm = n.norm(dim=-1, keepdim=True).detach().clamp(
            min=float(CFG.get("sdf_normal_eps", 1e-2))
        )
        ni   = n / n_norm
        di   = -(ni * a).sum(-1)
        normals.append(ni)
        offsets.append(di)
    ns    = torch.stack(normals, dim=1)
    ds    = torch.stack(offsets, dim=1)
    a_len = (v[:, 2] - v[:, 1]).norm(dim=-1)
    b_len = (v[:, 2] - v[:, 0]).norm(dim=-1)
    c_len = (v[:, 1] - v[:, 0]).norm(dim=-1)
    perim = (a_len + b_len + c_len).detach().clamp(
        min=float(CFG.get("sdf_normal_eps", 1e-2))
    )
    inc   = (a_len.unsqueeze(-1) * v[:, 0]
           + b_len.unsqueeze(-1) * v[:, 1]
           + c_len.unsqueeze(-1) * v[:, 2]) / perim.unsqueeze(-1)
    phi_s = ((inc.unsqueeze(1) * ns).sum(-1) + ds).max(dim=1).values
    return ns, ds, phi_s


# ---------------------------------------------------------------------------
# Differentiable render
# ---------------------------------------------------------------------------

def render_differentiable(model, cam_pos, R, focal, W, H,
                          sort_order, return_max_weights=False, crop=None):
    """
    Faster Python/PyTorch renderer with optional crop rendering.

    crop is either None or (x0, x1, y0, y1) in full-image coordinates,
    where x1/y1 are exclusive. When crop is provided, the renderer still
    projects triangles in the full image coordinate system, but it only builds
    the canvas/transmittance buffers for that crop. This is a large speed win
    when training one local chunk.
    """
    screen_xy, valid = project_vertices(model.vertices, cam_pos, R, focal, W, H)
    ns_all, ds_all, phi_s_all = compute_sdf_params(screen_xy)

    device = model.vertices.device

    if crop is None:
        x0, x1, y0, y1 = 0, W, 0, H
    else:
        x0, x1, y0, y1 = crop
        x0 = max(0, min(W - 1, int(x0)))
        x1 = max(x0 + 1, min(W, int(x1)))
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(y0 + 1, min(H, int(y1)))

    out_W = x1 - x0
    out_H = y1 - y0

    flat_canvas = torch.zeros(out_H * out_W, 3, device=device)
    flat_trans  = torch.ones(out_H * out_W, device=device)

    colors    = model.colors
    opacities = model.opacities
    sigmas    = model.sigmas
    max_weights = torch.zeros(model.num_triangles(), device=device)
    min_bbox_px = int(CFG.get("min_projected_bbox_px", 2))
    phi_eps = float(CFG.get("sdf_phi_eps", 1e-3))
    ratio_eps = float(CFG.get("render_ratio_eps", 1e-4))

    for idx in sort_order:
        if not bool(valid[idx]):
            continue
        phi_s = phi_s_all[idx]
        if (not bool(torch.isfinite(phi_s).detach())
                or float(phi_s.detach()) >= -phi_eps):
            continue

        vxy = screen_xy[idx]
        if not bool(torch.isfinite(vxy).all().detach()):
            continue
        if (not bool(torch.isfinite(ns_all[idx]).all().detach())
                or not bool(torch.isfinite(ds_all[idx]).all().detach())):
            continue

        # Triangle bounding box in full-image coordinates, intersected with crop.
        bx0 = max(x0, int(torch.floor(vxy[:, 0].min()).detach().item()))
        bx1 = min(x1 - 1, int(torch.ceil(vxy[:, 0].max()).detach().item()))
        by0 = max(y0, int(torch.floor(vxy[:, 1].min()).detach().item()))
        by1 = min(y1 - 1, int(torch.ceil(vxy[:, 1].max()).detach().item()))
        if bx0 > bx1 or by0 > by1:
            continue
        if (bx1 - bx0 + 1) < min_bbox_px or (by1 - by0 + 1) < min_bbox_px:
            continue

        xs = torch.arange(bx0, bx1 + 1, device=device, dtype=torch.float32)
        ys = torch.arange(by0, by1 + 1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        px = grid_x.reshape(-1)
        py = grid_y.reshape(-1)
        if px.numel() == 0:
            continue

        pts = torch.stack([px, py], dim=-1).to(dtype=ns_all.dtype)
        phi_p = ((pts.unsqueeze(1) * ns_all[idx].unsqueeze(0)).sum(-1)
                 + ds_all[idx].unsqueeze(0)).max(dim=1).values

        # Detach the inradius-like scale term. Backpropagating through this
        # denominator is a common source of exploding gradients for tiny or
        # edge-on projected triangles, while phi_p still gives useful vertex
        # motion gradients.
        phi_s_denom = phi_s.detach().clamp(max=-phi_eps)
        ratio = (phi_p / phi_s_denom).clamp(0.0, 1.0)
        active = ratio > ratio_eps
        if not bool(active.any()):
            continue
        px = px[active]
        py = py[active]
        ratio = ratio[active].clamp(min=ratio_eps)
        sigma = sigmas[idx].clamp(min=1.0)
        alpha = opacities[idx] * ratio.pow(sigma)
        if not bool(torch.isfinite(alpha).all().detach()):
            continue

        # Convert full-image pixel coordinates into crop-local flat indices.
        local_x = px.long() - x0
        local_y = py.long() - y0
        flat_idx = local_y * out_W + local_x
        T = flat_trans[flat_idx]

        if return_max_weights:
            max_weights[idx] = (T * opacities[idx]).max().detach()

        contrib = colors[idx].unsqueeze(0) * (alpha * T).unsqueeze(-1) * 255.0
        flat_canvas.index_add_(0, flat_idx, contrib)

        # Detached transmittance prevents a huge graph through all previous tris.
        flat_trans[flat_idx] = T * (1.0 - alpha).detach()

    canvas = flat_canvas.reshape(out_H, out_W, 3)
    return (canvas, max_weights) if return_max_weights else canvas


# ---------------------------------------------------------------------------
# Chunk crop helpers
# ---------------------------------------------------------------------------

def compute_model_crop(model_or_vertices, cam_pos, R, focal, W, H,
                       margin=25, min_size=48, max_area_frac=0.15):
    """
    Compute a robust full-image crop around the current model/chunk projection.

    Important change from the first crop version:
      - Uses projected triangle CENTERS rather than all triangle vertices.
      - Ignores extreme projected outliers using quantiles.

    This avoids one stretched triangle or one bad projected vertex forcing the
    crop to cover half the image. Returns (x0, x1, y0, y1), where x1/y1 are
    exclusive, or None if the chunk is not usefully visible in this camera.
    """
    with torch.no_grad():
        vertices = (model_or_vertices.vertices
                    if hasattr(model_or_vertices, "vertices")
                    else model_or_vertices)
        screen_xy, valid = project_vertices(vertices, cam_pos, R, focal, W, H)
        if not bool(valid.any()):
            return None

        # Use projected triangle centers, not every vertex. This makes the crop
        # follow the spatial chunk rather than being dominated by stretched or
        # outlier triangle corners.
        pts = screen_xy[valid].mean(dim=1)  # (N_visible_triangles, 2)
        sx, sy = pts[:, 0], pts[:, 1]

        # Keep centers that are not wildly outside the image. Some margin is
        # allowed because triangles near the border can still contribute.
        loose = ((sx >= -0.5 * W) & (sx <= 1.5 * W) &
                 (sy >= -0.5 * H) & (sy <= 1.5 * H))
        if int(loose.sum().item()) < 3:
            return None
        sx = sx[loose]
        sy = sy[loose]

        # Robust crop: ignore the most extreme projected centers. This prevents
        # a few unstable/outlier triangles from forcing a giant crop.
        if sx.numel() >= 20:
            qlo, qhi = 0.10, 0.90
            sx0 = torch.quantile(sx, qlo)
            sx1 = torch.quantile(sx, qhi)
            sy0 = torch.quantile(sy, qlo)
            sy1 = torch.quantile(sy, qhi)
            keep = (sx >= sx0) & (sx <= sx1) & (sy >= sy0) & (sy <= sy1)
            if int(keep.sum().item()) >= 3:
                sx = sx[keep]
                sy = sy[keep]

        x0 = int(torch.floor(sx.min()).item()) - margin
        x1 = int(torch.ceil(sx.max()).item()) + margin + 1
        y0 = int(torch.floor(sy.min()).item()) - margin
        y1 = int(torch.ceil(sy.max()).item()) + margin + 1

        x0 = max(0, min(W - 1, x0))
        x1 = max(x0 + 1, min(W, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(y0 + 1, min(H, y1))

        if (x1 - x0) < min_size or (y1 - y0) < min_size:
            return None

        # If the chunk still projects too broadly, shrink to a bounded area
        # around the crop center. With max_area_frac=0.15, the crop should be
        # much closer to a local patch than a half-frame target.
        if max_area_frac is not None and max_area_frac > 0:
            max_area = max_area_frac * W * H
            area = (x1 - x0) * (y1 - y0)
            if area > max_area:
                aspect = (x1 - x0) / max(1, (y1 - y0))
                new_h = int(math.sqrt(max_area / max(aspect, 1e-6)))
                new_w = int(new_h * aspect)
                new_w = max(min_size, min(W, new_w))
                new_h = max(min_size, min(H, new_h))
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                x0 = max(0, min(W - new_w, cx - new_w // 2))
                y0 = max(0, min(H - new_h, cy - new_h // 2))
                x1 = x0 + new_w
                y1 = y0 + new_h

        return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# Crop debug helpers
# ---------------------------------------------------------------------------

def _target_tensor_to_uint8_image(target_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a target tensor with values in [0,255] or [0,1] to uint8 HxWx3.
    This keeps the debug functions robust even if target normalization changes.
    """
    arr = target_tensor.detach().cpu().float()
    if arr.numel() == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if float(arr.max().item()) <= 2.0:
        arr = arr * 255.0
    return arr.clamp(0, 255).byte().numpy()


def save_crop_debug_images(target_tensor: torch.Tensor,
                           crop,
                           step: int,
                           view_name: str,
                           W: int,
                           H: int,
                           out_dir: str = "crop_debug") -> None:
    """
    Save debug images for the active crop:
      1. Full target image with a red crop rectangle.
      2. Standalone target crop.

    These images are for sanity checking: the crop should cover a meaningful
    scene patch, not the whole image, empty sky, or a tiny meaningless area.
    """
    if crop is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    x0, x1, y0, y1 = [int(v) for v in crop]
    x0 = max(0, min(W - 1, x0))
    x1 = max(x0 + 1, min(W, x1))
    y0 = max(0, min(H - 1, y0))
    y1 = max(y0 + 1, min(H, y1))

    safe_name = view_name.replace("/", "_").replace("\\", "_")
    crop_w = x1 - x0
    crop_h = y1 - y0
    area_frac = (crop_w * crop_h) / max(1, W * H)

    img = _target_tensor_to_uint8_image(target_tensor)

    if CFG.get("crop_debug_save_rectangle", True):
        from PIL import ImageDraw
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Red rectangle around the crop. x1/y1 are exclusive for tensors, but
        # ImageDraw expects inclusive-looking coordinates; subtract 1 for display.
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=3)
        rect_path = os.path.join(
            out_dir, f"crop_rect_step_{step:05d}_{safe_name}_{crop_w}x{crop_h}_{area_frac:.0%}.png"
        )
        pil.save(rect_path)

    if CFG.get("crop_debug_save_target_crop", True):
        crop_img = img[y0:y1, x0:x1, :]
        crop_path = os.path.join(
            out_dir, f"target_crop_step_{step:05d}_{safe_name}_{crop_w}x{crop_h}_{area_frac:.0%}.png"
        )
        Image.fromarray(crop_img).save(crop_path)

    print(
        f"[crop-debug] step={step} view={view_name} "
        f"x={x0}:{x1} y={y0}:{y1} size={crop_w}x{crop_h} area={area_frac:.1%} "
        f"saved to {out_dir}/"
    )



# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def blit_canvas_to_pygame(canvas: torch.Tensor, screen: pygame.Surface) -> None:
    with torch.no_grad():
        frame = canvas.clamp(0, 255).byte().cpu().numpy()   # (H, W, 3)
    pygame.surfarray.blit_array(screen, frame.transpose(1, 0, 2))


def render_preview_canvas(model, device, W, H) -> torch.Tensor:
    """Render from the INTERACTIVE camera (stable, not the training camera)."""
    cam_pos_t = torch.from_numpy(Globals.cameraPosition).to(device)
    R_t       = torch.from_numpy(Globals.R).to(device)
    order     = sort_triangles_by_depth(model.vertices, cam_pos_t, R_t)
    with torch.no_grad():
        return render_differentiable(
            model, cam_pos_t, R_t, Globals.focalLength, W, H, order)


def save_render_png(model, device, W, H, path: str) -> None:
    canvas = render_preview_canvas(model, device, W, H)
    frame  = canvas.clamp(0, 255).byte().cpu().numpy()   # (H, W, 3)
    Image.fromarray(frame).save(path)
    print(f"[image] {path}")


def save_training_view_render_png(
    model,
    device,
    view,
    target,
    W,
    H,
    focal,
    step: int,
    out_dir: str,
    crop=None,
):
    """
    Save render images from the CURRENT TRAINING CAMERA, not the interactive camera.

    Files saved when enabled:
      1. Full training-view render.
      2. Rendered crop.
      3. Target crop.
      4. Side-by-side crop comparison: rendered crop | target crop.

    The side-by-side comparison is usually the most useful progress image.
    """
    os.makedirs(out_dir, exist_ok=True)

    safe_name = view.filename.replace("/", "_").replace("\\", "_")
    cam_t = torch.from_numpy(view.cam_pos).to(device)
    R_t = torch.from_numpy(view.R).to(device)
    order = sort_triangles_by_depth(model.vertices, cam_t, R_t)

    with torch.no_grad():
        # ------------------------------------------------------------
        # 1. Full training-camera render
        # ------------------------------------------------------------
        if CFG.get("save_training_full_view", True):
            full_canvas = render_differentiable(
                model, cam_t, R_t, focal, W, H, order, crop=None
            )
            full_img = full_canvas.clamp(0, 255).byte().cpu().numpy()
            full_path = os.path.join(
                out_dir,
                f"traincam_full_step_{step:05d}_{safe_name}.png",
            )
            Image.fromarray(full_img).save(full_path)
            print(f"[traincam-image] {full_path}")

        # ------------------------------------------------------------
        # 2. Crop render + crop target comparison
        # ------------------------------------------------------------
        if crop is not None and CFG.get("save_training_crop_compare", True):
            x0, x1, y0, y1 = [int(v) for v in crop]
            x0 = max(0, min(W - 1, x0))
            x1 = max(x0 + 1, min(W, x1))
            y0 = max(0, min(H - 1, y0))
            y1 = max(y0 + 1, min(H, y1))

            crop_canvas = render_differentiable(
                model, cam_t, R_t, focal, W, H, order, crop=(x0, x1, y0, y1)
            )
            render_crop = crop_canvas.clamp(0, 255).byte().cpu().numpy()

            target_crop_t = target[y0:y1, x0:x1, :]
            target_crop = target_crop_t.detach().cpu().float().clamp(0, 255).byte().numpy()

            render_path = os.path.join(
                out_dir,
                f"traincam_render_crop_step_{step:05d}_{safe_name}.png",
            )
            target_path = os.path.join(
                out_dir,
                f"traincam_target_crop_step_{step:05d}_{safe_name}.png",
            )
            compare_path = os.path.join(
                out_dir,
                f"traincam_compare_step_{step:05d}_{safe_name}.png",
            )

            Image.fromarray(render_crop).save(render_path)
            Image.fromarray(target_crop).save(target_path)

            # Side-by-side comparison: rendered crop | target crop.
            if render_crop.shape == target_crop.shape:
                spacer = np.full((render_crop.shape[0], 8, 3), 255, dtype=np.uint8)
                comparison = np.concatenate([render_crop, spacer, target_crop], axis=1)
                Image.fromarray(comparison).save(compare_path)
                print(f"[traincam-compare] {compare_path}")
            else:
                print(f"[traincam-crop] {render_path} and {target_path}")


# ---------------------------------------------------------------------------
# Loss functions  (paper Eq. 3)
# ---------------------------------------------------------------------------

def _ssim_loss(pred, target, window_size=11):
    p   = (pred   / 255.0).permute(2, 0, 1).unsqueeze(0)
    t   = (target / 255.0).permute(2, 0, 1).unsqueeze(0)
    pad = window_size // 2
    mu_p  = F.avg_pool2d(p,   window_size, stride=1, padding=pad)
    mu_t  = F.avg_pool2d(t,   window_size, stride=1, padding=pad)
    mu_pp = F.avg_pool2d(p*p, window_size, stride=1, padding=pad)
    mu_tt = F.avg_pool2d(t*t, window_size, stride=1, padding=pad)
    mu_pt = F.avg_pool2d(p*t, window_size, stride=1, padding=pad)
    sp    = mu_pp - mu_p*mu_p
    st    = mu_tt - mu_t*mu_t
    spt   = mu_pt - mu_p*mu_t
    C1, C2 = 0.01**2, 0.03**2
    ssim   = ((2*mu_p*mu_t + C1)*(2*spt + C2)) / \
             ((mu_p**2 + mu_t**2 + C1)*(sp + st + C2))
    return (1.0 - ssim).mean() / 2.0


def compute_loss(canvas, target, model, lam=0.2, beta1=0.01, beta_area=1e-5,
                 use_ssim=True, anchor_vertices=None, anchor_weight=0.0):
    """
    Photometric loss + opacity regularization + POSITIVE area penalty.

    Important: older versions used ls = -area and added beta * ls.
    That rewarded triangles for growing larger, because minimizing the loss
    made the negative area term more negative. This version penalizes area
    instead, so the printed ls value should be positive.
    """
    p, t = canvas / 255.0, target / 255.0
    l1 = (p - t).abs().mean()

    if use_ssim:
        dssim = _ssim_loss(canvas, target)
        photo = (1.0 - lam) * l1 + lam * dssim
    else:
        dssim = torch.zeros((), device=canvas.device, dtype=canvas.dtype)
        photo = l1

    o = model.opacities
    lo = -(o * torch.log(o + 1e-7) + (1 - o) * torch.log(1 - o + 1e-7)).mean()

    v = model.vertices
    area = 0.5 * torch.cross(
        v[:, 1] - v[:, 0],
        v[:, 2] - v[:, 0],
        dim=-1,
    ).norm(dim=-1).mean()

    if (anchor_vertices is not None
            and anchor_weight > 0
            and anchor_vertices.shape == model.vertices.shape):
        anchor = (model.vertices - anchor_vertices).pow(2).mean()
    else:
        anchor = torch.zeros((), device=canvas.device, dtype=canvas.dtype)

    total = photo + beta1 * lo + beta_area * area + anchor_weight * anchor

    return total, {
        "photo": photo.item(),
        "l1": l1.item(),
        "dssim": dssim.item(),
        "lo": lo.item(),
        "ls": area.item(),
        "anchor": anchor.item(),
    }


# ---------------------------------------------------------------------------
# Densification  (paper Sec. 3.2)
# ---------------------------------------------------------------------------

def densify_model(model, max_triangles=2000, tau_small=1e-3):
    N = model.num_triangles()
    if N >= max_triangles:
        return model

    remaining = max_triangles - N
    if remaining <= 0:
        return model

    # Each selected parent can produce up to 4 new triangles. Choose parent
    # count based on remaining capacity so densification does not blow past
    # max_triangles. This is intentionally conservative for the Python renderer.
    max_parents = max(1, remaining // 4)
    n_to_add = min(max(N // 8, 1), max_parents)

    with torch.no_grad():
        opacs = model.opacities.detach()
        sigs  = model.sigmas.detach()
        prob  = (0.5 * opacs / opacs.sum().clamp(1e-7)
               + 0.5 * (1/sigs.clamp(1e-4)) / (1/sigs.clamp(1e-4)).sum().clamp(1e-7))
        prob  = prob / prob.sum()
        chosen = torch.multinomial(prob, num_samples=min(n_to_add, N),
                                   replacement=False)
        verts_old  = model.vertices.detach()
        colors_old = model.color_logit.detach()
        opacs_old  = model.opacity_logit.detach()
        sigs_old   = model.log_sigma.detach()
        nv, nc, no, ns = [], [], [], []

        for idx in chosen:
            v         = verts_old[idx]
            v0,v1,v2  = v[0], v[1], v[2]
            m01,m12,m02 = (v0+v1)/2, (v1+v2)/2, (v0+v2)/2
            area = 0.5 * torch.cross(v1-v0, v2-v0, dim=-1).norm()
            if area < tau_small:
                e1  = v1 - v0
                nrm = torch.cross(e1, v2-v0, dim=-1)
                nrm = nrm / nrm.norm().clamp(1e-7)
                t1  = e1 / e1.norm().clamp(1e-7)
                t2  = torch.cross(nrm, t1, dim=-1)
                noise = (torch.randn(3, device=v.device).unsqueeze(-1)
                         * torch.stack([t1, t2, t1], dim=0)) * area.sqrt() * 0.1
                new_v = (v + noise).unsqueeze(0)
            else:
                new_v = torch.stack([
                    torch.stack([v0,  m01, m02]),
                    torch.stack([m01, v1,  m12]),
                    torch.stack([m02, m12, v2 ]),
                    torch.stack([m01, m12, m02]),
                ], dim=0)
            n_new = new_v.shape[0]
            nv.append(new_v)
            nc.append(colors_old[idx].unsqueeze(0).expand(n_new, -1))
            no.append(opacs_old[idx].unsqueeze(0).expand(n_new))
            ns.append(sigs_old[idx].unsqueeze(0).expand(n_new))

        if not nv:
            return model
        cat_v = torch.cat([verts_old,  torch.cat(nv, 0)], 0)
        cat_c = torch.cat([colors_old, torch.cat(nc, 0)], 0)
        cat_o = torch.cat([opacs_old,  torch.cat(no, 0)], 0)
        cat_s = torch.cat([sigs_old,   torch.cat(ns, 0)], 0)

    new_m = TriangleSplatModel(
        cat_v, torch.sigmoid(cat_c), torch.sigmoid(cat_o), torch.exp(cat_s)
    ).to(model.vertices.device)
    with torch.no_grad():
        new_m.color_logit.copy_(cat_c)
        new_m.opacity_logit.copy_(cat_o)
        new_m.log_sigma.copy_(cat_s)
    print(f"[densify] {N} -> {new_m.num_triangles()} (+{new_m.num_triangles()-N})")
    return new_m


# ---------------------------------------------------------------------------
# Pruning  (paper Sec. 3.2)
# ---------------------------------------------------------------------------

def prune_model(model, max_weights, tau_prune=0.005):
    with torch.no_grad():
        keep     = max_weights >= tau_prune
        n_keep   = int(keep.sum().item())
        n_pruned = model.num_triangles() - n_keep
        if n_pruned == 0:
            return model
        new_m = TriangleSplatModel(
            model.vertices.detach()[keep],
            torch.sigmoid(model.color_logit.detach()[keep]),
            torch.sigmoid(model.opacity_logit.detach()[keep]),
            torch.exp(model.log_sigma.detach()[keep]),
        ).to(model.vertices.device)
        new_m.color_logit.copy_(model.color_logit.detach()[keep])
        new_m.opacity_logit.copy_(model.opacity_logit.detach()[keep])
        new_m.log_sigma.copy_(model.log_sigma.detach()[keep])
    print(f"[prune]   {model.num_triangles()} -> {n_keep} (-{n_pruned})")
    return new_m


def clamp_vertices_to_anchor(model, anchor_vertices, max_offset):
    if (anchor_vertices is None
            or max_offset is None
            or max_offset <= 0
            or anchor_vertices.shape != model.vertices.shape):
        return

    with torch.no_grad():
        delta = model.vertices - anchor_vertices
        dist = delta.norm(dim=-1, keepdim=True)
        scale = (float(max_offset) / dist.clamp(min=1e-7)).clamp(max=1.0)
        model.vertices.copy_(anchor_vertices + delta * scale)


def model_parameters_are_finite(model):
    return (
        torch.isfinite(model.vertices).all()
        and torch.isfinite(model.color_logit).all()
        and torch.isfinite(model.opacity_logit).all()
        and torch.isfinite(model.log_sigma).all()
    )


def model_gradients_are_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def clamp_model_parameter_ranges(model):
    with torch.no_grad():
        max_abs_logit = float(CFG.get("max_abs_logit", 10.0))
        min_log_sigma = float(CFG.get("min_log_sigma", -4.0))
        max_log_sigma = float(CFG.get("max_log_sigma", 4.0))
        model.color_logit.clamp_(-max_abs_logit, max_abs_logit)
        model.opacity_logit.clamp_(-max_abs_logit, max_abs_logit)
        model.log_sigma.clamp_(min_log_sigma, max_log_sigma)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(model, optimizer, scheduler, target,
                  cam_pos, R, focal, W, H, sort_order, acc_max_weights,
                  use_ssim=True, crop=None, anchor_vertices=None):
    optimizer.zero_grad()
    zero = {"vertices":0.0, "sigma":0.0, "opacity":0.0, "color":0.0}
    if not sort_order:
        return 0.0, zero, {"skipped": True}

    canvas, max_w = render_differentiable(
        model, cam_pos, R, focal, W, H, sort_order,
        return_max_weights=True, crop=crop)

    if crop is None:
        target_for_loss = target
    else:
        x0, x1, y0, y1 = crop
        target_for_loss = target[y0:y1, x0:x1, :]

    if target_for_loss.numel() == 0 or canvas.numel() == 0:
        return 0.0, zero, {"skipped": True}

    if acc_max_weights.shape[0] == model.num_triangles():
        acc_max_weights[:] = torch.max(acc_max_weights, max_w)

    loss, components = compute_loss(
        canvas, target_for_loss, model,
        use_ssim=use_ssim,
        anchor_vertices=anchor_vertices,
        anchor_weight=CFG.get("vertex_anchor_weight", 0.0),
    )
    components["crop_w"] = int(canvas.shape[1])
    components["crop_h"] = int(canvas.shape[0])
    if crop is None:
        components["crop_x0"] = 0
        components["crop_x1"] = W
        components["crop_y0"] = 0
        components["crop_y1"] = H
        components["crop_area_frac"] = 1.0
    else:
        x0, x1, y0, y1 = crop
        components["crop_x0"] = int(x0)
        components["crop_x1"] = int(x1)
        components["crop_y0"] = int(y0)
        components["crop_y1"] = int(y1)
        components["crop_area_frac"] = float(((x1 - x0) * (y1 - y0)) / max(1, W * H))

    if not loss.requires_grad:
        return float(loss.item()), zero, components

    components["rolled_back"] = 0.0
    components["rebuild_optimizer"] = 0.0

    if not torch.isfinite(loss):
        components["rolled_back"] = 1.0
        components["rebuild_optimizer"] = 1.0
        print("[rollback] Non-finite loss before backward; skipped optimizer step.", flush=True)
        optimizer.zero_grad(set_to_none=True)
        return 0.0, zero, components

    prev_state = {
        "vertices": model.vertices.detach().clone(),
        "color_logit": model.color_logit.detach().clone(),
        "opacity_logit": model.opacity_logit.detach().clone(),
        "log_sigma": model.log_sigma.detach().clone(),
    }

    loss.backward()
    total_grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
        error_if_nonfinite=False,
    )

    if (not torch.isfinite(total_grad_norm)) or (not model_gradients_are_finite(model)):
        with torch.no_grad():
            model.vertices.copy_(prev_state["vertices"])
            model.color_logit.copy_(prev_state["color_logit"])
            model.opacity_logit.copy_(prev_state["opacity_logit"])
            model.log_sigma.copy_(prev_state["log_sigma"])
        optimizer.zero_grad(set_to_none=True)
        components["rolled_back"] = 1.0
        components["rebuild_optimizer"] = 1.0
        print("[rollback] Non-finite gradients before optimizer step; skipped step.", flush=True)
        return float(loss.detach().item()), zero, components

    grad_norms = {
        "vertices": float(model.vertices.grad.norm().item())
                    if model.vertices.grad is not None else 0.0,
        "sigma":    float(model.log_sigma.grad.norm().item())
                    if model.log_sigma.grad is not None else 0.0,
        "opacity":  float(model.opacity_logit.grad.norm().item())
                    if model.opacity_logit.grad is not None else 0.0,
        "color":    float(model.color_logit.grad.norm().item())
                    if model.color_logit.grad is not None else 0.0,
    }
    optimizer.step()
    clamp_model_parameter_ranges(model)
    clamp_vertices_to_anchor(
        model,
        anchor_vertices,
        CFG.get("vertex_anchor_max_offset", None),
    )
    if not model_parameters_are_finite(model):
        with torch.no_grad():
            model.vertices.copy_(prev_state["vertices"])
            model.color_logit.copy_(prev_state["color_logit"])
            model.opacity_logit.copy_(prev_state["opacity_logit"])
            model.log_sigma.copy_(prev_state["log_sigma"])
        components["rolled_back"] = 1.0
        components["rebuild_optimizer"] = 1.0
        optimizer.zero_grad(set_to_none=True)
        print("[rollback] Non-finite model parameters after optimizer step; restored previous step.", flush=True)
    else:
        components["rolled_back"] = 0.0
        components["rebuild_optimizer"] = 0.0
        scheduler.step()
    return float(loss.item()), grad_norms, components



# ---------------------------------------------------------------------------
# Good-images whitelist helper
# ---------------------------------------------------------------------------

def load_goodimage_whitelist(goodimages_dir: str) -> set[str]:
    """
    Use the image filenames inside goodimages_dir as a whitelist.

    Example folder:
        goodimages/P1180155.JPG
        goodimages/P1180188.JPG

    The images can be copied into goodimages/. They do not need to be loaded
    from there during training; their filenames are used to filter the COLMAP
    TrainingView list. Target images are still loaded from CFG["images_dir"].
    """
    if not goodimages_dir or not os.path.isdir(goodimages_dir):
        print(f"[goodimages] Folder not found: {goodimages_dir}")
        return set()

    allowed_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    names: set[str] = set()

    for fname in os.listdir(goodimages_dir):
        path = os.path.join(goodimages_dir, fname)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(fname)
        if ext in allowed_exts:
            # Store basename only. COLMAP view.filename usually contains only
            # the image name, but this also works if paths are accidentally used.
            names.add(os.path.basename(fname))

    print(f"[goodimages] Loaded {len(names)} whitelisted filename(s) from {goodimages_dir}")
    for name in sorted(list(names))[:20]:
        print(f"  {name}")
    if len(names) > 20:
        print(f"  ... and {len(names) - 20} more")

    return names


def apply_goodimage_whitelist(train_views: list, good_names: set[str]) -> list:
    """
    Filter TrainingView objects by filename using exact basename matching first,
    then case-insensitive matching as a convenience for .jpg/.JPG differences.
    """
    if not good_names:
        return train_views

    exact_names = {os.path.basename(n) for n in good_names}
    lower_names = {n.lower() for n in exact_names}

    filtered = []
    matched = set()
    for view in train_views:
        base = os.path.basename(view.filename)
        if base in exact_names or base.lower() in lower_names:
            filtered.append(view)
            matched.add(base)

    print(f"[goodimages] Filtered train views: {len(train_views)} -> {len(filtered)}")

    # Helpful warning if copied filenames do not match COLMAP filenames.
    view_bases_lower = {os.path.basename(v.filename).lower() for v in train_views}
    missing = sorted([n for n in exact_names if n.lower() not in view_bases_lower])
    if missing:
        print("[goodimages] Warning: these whitelist files did not match any COLMAP training view:")
        for name in missing[:20]:
            print(f"  {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if not filtered:
        raise RuntimeError(
            "goodimages whitelist removed all training views. Check that files in "
            "goodimages/ have names that exactly match COLMAP image filenames, "
            "for example P1180188.JPG."
        )

    return filtered


# ---------------------------------------------------------------------------
# Target image cache
# ---------------------------------------------------------------------------

def make_target_cache(views, images_dir, W, H, device):
    mode = CFG.get("target_cache", "cpu")
    cache = {}
    if mode == "off":
        return cache
    print(f"[cache] Preloading {len(views)} target images to {mode}...")
    for view in views:
        arr = load_target_image(view, images_dir, W, H)
        ten = torch.from_numpy(arr)
        if mode == "gpu" and device.type == "cuda":
            ten = ten.to(device, non_blocking=True)
        elif mode == "cpu" and device.type == "cuda":
            try:
                ten = ten.pin_memory()
            except RuntimeError:
                pass
        cache[view.filename] = ten
    return cache

def get_target_tensor(view, cache, images_dir, W, H, device):
    ten = cache.get(view.filename)
    if ten is None:
        ten = torch.from_numpy(load_target_image(view, images_dir, W, H))
    if ten.device != device:
        ten = ten.to(device, non_blocking=True)
    return ten


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    W, H = CFG["screen_w"], CFG["screen_h"]
    os.makedirs(CFG["image_dir"], exist_ok=True)
    os.makedirs(CFG.get("training_render_dir", "renders_training_views"), exist_ok=True)

    # ---- data ----
    Globals.pointcloudData = load_points3D(CFG["points_file"])
    train_views, test_views, colmap_camera = load_dataset(
        CFG["sparse_dir"], CFG["images_dir"], W, H)
    Globals.focalLength = colmap_focal_to_renderer(colmap_camera, W, H)

    # Optional manual view selection: put copies of views where your chunk is
    # actually visible into goodimages/. Only matching COLMAP views are trained.
    if CFG.get("use_goodimages_whitelist", False):
        good_names = load_goodimage_whitelist(CFG.get("goodimages_dir", "goodimages"))
        if good_names:
            train_views = apply_goodimage_whitelist(train_views, good_names)
            print(f"[goodimages] Using {len(train_views)} selected training view(s).")
        else:
            print("[goodimages] No whitelist images found; using all training views.")

    max_pts = CFG["max_points"]
    points  = (Globals.pointcloudData[:max_pts]
               if max_pts else Globals.pointcloudData)
    print(f"Using {len(points)} points  |  focal {Globals.focalLength:.1f} px")

    all_triangles = initialize_triangles(points)

    N_CHUNKS  = CFG["n_chunks"]
    all_chunks = get_all_chunk_indices(points, CFG["chunk_k"], N_CHUNKS)
    current_chunk_idx = 0
    chunk_indices     = all_chunks[current_chunk_idx]

    device = Globals.device
    print(f"Device: {device}  |  {N_CHUNKS} chunk(s) of up to "
          f"{CFG['chunk_k']} triangles")

    # ---- view filtering ----
    def chunk_visible_from(view, c_idx):
        for i in c_idx:
            v_cam = (all_triangles[i].vertices.mean(0) - view.cam_pos) @ view.R
            if v_cam[2] > 0.1:
                return True
        return False

    def get_vis_views(c_idx):
        vis = [v for v in train_views if chunk_visible_from(v, c_idx)]
        return vis if vis else train_views

    def build_crop_reference_vertices(c_idx):
        verts = np.array([all_triangles[i].vertices for i in c_idx],
                         dtype=np.float32)
        return torch.from_numpy(verts).to(device)

    vis_views = get_vis_views(chunk_indices)
    print(f"Chunk 0: {len(chunk_indices)} tris | "
          f"{len(vis_views)} visible views")

    # Cache all training targets once. The original code loaded and resized a
    # JPEG from disk every single training step.
    target_cache = make_target_cache(train_views, CFG["images_dir"], W, H, device)

    # ---- model: try checkpoint first ----
    ckpt = load_checkpoint(CFG["checkpoint_path"], device)
    if ckpt is not None:
        model         = model_from_checkpoint(ckpt, device)
        train_steps   = ckpt["train_steps"]
        total_loss    = ckpt["total_loss"]
        chunk_indices = ckpt.get("chunk_indices", chunk_indices)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, train_steps_so_far=train_steps)
        if CFG.get("reset_optimizer_on_load", True):
            print("[ckpt] Reset optimizer/scheduler state on load.")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resuming from step {train_steps}, "
              f"N={model.num_triangles()} triangles")
    else:
        model       = build_model(all_triangles, chunk_indices, device)
        train_steps = 0
        total_loss  = 0.0
        optimizer, scheduler = build_optimizer_and_scheduler(model, 0)

    acc_max_weights = torch.zeros(model.num_triangles(), device=device)
    crop_reference_vertices = build_crop_reference_vertices(chunk_indices)
    anchor_vertices = model.vertices.detach().clone()

    DISPLAY_EVERY = CFG["display_every"]
    DENSIFY_EVERY = CFG["densify_every"]
    PRUNE_EVERY   = CFG["prune_every"]
    MAX_TRI       = CFG["max_triangles"]
    TAU_PRUNE     = CFG["tau_prune"]
    SAVE_EVERY    = CFG["save_every"]
    SAVE_IMG      = CFG["save_image_every"]
    IMG_DIR       = CFG["image_dir"]
    CKPT_PATH     = CFG["checkpoint_path"]

    # ---- pygame ----
    pygame.init()
    clock  = pygame.time.Clock()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Triangle Splatting")
    font   = pygame.font.SysFont("Arial", 18)

    current_view_idx = 0
    train_mode       = False
    last_grad_norms  = {"vertices":0.0,"sigma":0.0,"opacity":0.0,"color":0.0}
    last_components  = {}
    step_times: list[float] = []
    consecutive_crop_skips = 0
    consecutive_render_skips = 0
    consecutive_rejected_updates = 0

    running = True
    while running:

        # ---- events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    Globals.show_debug = not Globals.show_debug
                if event.key == pygame.K_t:
                    train_mode = True
                    print("Training ON.")
                if (event.key == pygame.K_s
                        and pygame.key.get_mods() & pygame.KMOD_CTRL):
                    save_checkpoint(model, optimizer, scheduler,
                                    train_steps, total_loss,
                                    chunk_indices, CKPT_PATH)

        # ---- camera ----
        keys  = pygame.key.get_pressed()
        fwd   = Globals.R @ np.array([0, 0, 1], dtype=np.float32)
        right = Globals.R @ np.array([1, 0, 0], dtype=np.float32)
        up    = np.array([0, 1, 0], dtype=np.float32)
        if keys[pygame.K_w]:      Globals.cameraPosition += fwd   * Globals.delta
        if keys[pygame.K_s]:      Globals.cameraPosition -= fwd   * Globals.delta
        if keys[pygame.K_a]:      Globals.cameraPosition -= right * Globals.delta
        if keys[pygame.K_d]:      Globals.cameraPosition += right * Globals.delta
        if keys[pygame.K_LSHIFT]: Globals.cameraPosition += up    * Globals.delta
        if keys[pygame.K_SPACE]:  Globals.cameraPosition -= up    * Globals.delta
        dyaw = dpitch = 0.0
        if keys[pygame.K_LEFT]:  dyaw   =  Globals.yaw_speed
        if keys[pygame.K_RIGHT]: dyaw   = -Globals.yaw_speed
        if keys[pygame.K_UP]:    dpitch =  Globals.pitch_speed
        if keys[pygame.K_DOWN]:  dpitch = -Globals.pitch_speed
        update_rotation(dyaw, dpitch)

        # ---- training ----
        if train_mode and vis_views:
            t0 = time.perf_counter()

            view      = vis_views[current_view_idx % len(vis_views)]
            target    = get_target_tensor(view, target_cache, CFG["images_dir"], W, H, device)
            cam_t     = torch.from_numpy(view.cam_pos).to(device)
            R_t       = torch.from_numpy(view.R).to(device)

            crop = None
            if CFG.get("use_chunk_crop", True):
                crop_source = (crop_reference_vertices
                               if CFG.get("use_static_crop_reference", True)
                               else model)
                crop = compute_model_crop(
                    crop_source, cam_t, R_t, Globals.focalLength, W, H,
                    margin=CFG.get("crop_margin", 60),
                    min_size=CFG.get("crop_min_size", 32),
                    max_area_frac=CFG.get("crop_max_area_frac", 0.50),
                )
                if crop is None:
                    consecutive_crop_skips += 1
                    max_skips = int(CFG.get("max_consecutive_crop_skips", len(vis_views) + 5))

                    if (consecutive_crop_skips >= max_skips
                            and CFG.get("fallback_to_central_crop_after_skips", True)):
                        crop_w = int(W * 0.35)
                        crop_h = int(H * 0.35)
                        cx = W // 2
                        cy = H // 2
                        x0 = max(0, cx - crop_w // 2)
                        y0 = max(0, cy - crop_h // 2)
                        x1 = min(W, x0 + crop_w)
                        y1 = min(H, y0 + crop_h)
                        crop = (x0, x1, y0, y1)
                        print(
                            f"[crop-warning] step={train_steps + 1}: "
                            f"{consecutive_crop_skips} consecutive crop=None views. "
                            f"Using fallback crop={crop} on {view.filename}",
                            flush=True,
                        )
                        consecutive_crop_skips = 0
                    else:
                        current_view_idx += 1
                        if consecutive_crop_skips % 25 == 0:
                            print(
                                f"[crop-skip] step={train_steps + 1}: "
                                f"{consecutive_crop_skips} consecutive crop=None views; "
                                f"latest={view.filename}",
                                flush=True,
                            )
                        continue
                else:
                    consecutive_crop_skips = 0

                if (CFG.get("crop_debug_enabled", False)
                        and CFG.get("crop_debug_every", 0)
                        and train_steps % int(CFG.get("crop_debug_every", 50)) == 0):
                    save_crop_debug_images(
                        target, crop, train_steps, view.filename, W, H,
                        out_dir=CFG.get("crop_debug_dir", "crop_debug"),
                    )

            train_sorted = sort_triangles_by_depth(model.vertices, cam_t, R_t)

            cull_before = len(train_sorted)
            cull_after = cull_before
            if crop is not None and CFG.get("use_crop_culling", True):
                sorted_before_cull = train_sorted
                train_sorted = filter_sort_order_to_crop(
                    model, cam_t, R_t, Globals.focalLength, W, H, train_sorted, crop
                )
                cull_after = len(train_sorted)
                if cull_before > 0 and cull_after == 0:
                    train_sorted = sorted_before_cull
                    cull_after = cull_before
                    if train_steps % 25 == 0:
                        print(
                            f"[crop-cull-warning] step={train_steps + 1}: "
                            f"crop culling removed all triangles on {view.filename}; "
                            "using unculled sorted list for this step.",
                            flush=True,
                        )
                if train_steps % 50 == 0:
                    print(f"[crop-cull] triangles {cull_before} -> {cull_after}", flush=True)

            if not train_sorted:
                current_view_idx += 1
                consecutive_render_skips += 1
                if consecutive_render_skips % 25 == 0:
                    print(
                        f"[render-skip] step={train_steps + 1}: "
                        f"{consecutive_render_skips} consecutive views with no visible triangles; "
                        f"latest={view.filename}",
                        flush=True,
                    )
                max_render_skips = int(CFG.get("max_consecutive_render_skips", len(vis_views) + 5))
                if consecutive_render_skips >= max_render_skips:
                    if anchor_vertices.shape == model.vertices.shape:
                        with torch.no_grad():
                            model.vertices.copy_(anchor_vertices)
                        optimizer, scheduler = build_optimizer_and_scheduler(
                            model, train_steps_so_far=train_steps)
                        print(
                            f"[recovery] step={train_steps + 1}: "
                            f"{consecutive_render_skips} render skips. "
                            "Reset vertices to anchor and rebuilt optimizer.",
                            flush=True,
                        )
                    else:
                        print(
                            f"[recovery-warning] step={train_steps + 1}: "
                            f"{consecutive_render_skips} render skips, but anchor shape "
                            "does not match model shape.",
                            flush=True,
                        )
                    consecutive_render_skips = 0
                continue

            N_cur = model.num_triangles()
            if acc_max_weights.shape[0] != N_cur:
                new_acc = torch.zeros(N_cur, device=device)
                old_n   = min(acc_max_weights.shape[0], N_cur)
                new_acc[:old_n] = acc_max_weights[:old_n]
                acc_max_weights = new_acc

            loss_val, grad_norms, components = training_step(
                model, optimizer, scheduler, target,
                cam_t, R_t, Globals.focalLength, W, H,
                train_sorted, acc_max_weights,
                use_ssim=(train_steps >= CFG["ssim_start_step"]),
                crop=crop,
                anchor_vertices=anchor_vertices)

            # CUDA launches work asynchronously. Synchronize here so timing logs
            # include the real GPU work for this step instead of accidentally
            # charging it to the next innocent-looking CUDA call.
            if device.type == "cuda":
                torch.cuda.synchronize()

            if components.get("rebuild_optimizer", 0.0):
                consecutive_rejected_updates += 1
                if consecutive_rejected_updates >= int(CFG.get("max_consecutive_rejected_updates", 5)):
                    old_lr = float(CFG["lr_vertices_init"])
                    new_lr = max(
                        float(CFG.get("min_recovery_vertex_lr", 5e-5)),
                        old_lr * float(CFG.get("rejected_update_lr_decay", 0.5)),
                    )
                    CFG["lr_vertices_init"] = new_lr
                    with torch.no_grad():
                        anchor_vertices = model.vertices.detach().clone()
                    print(
                        f"[recovery] step={train_steps + 1}: "
                        f"{consecutive_rejected_updates} rejected updates. "
                        f"Reducing vertex LR {old_lr:.2e} -> {new_lr:.2e} "
                        "and resetting anchor.",
                        flush=True,
                    )
                    consecutive_rejected_updates = 0
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model, train_steps_so_far=train_steps)
                current_view_idx += 1
                consecutive_render_skips = 0
                print(
                    f"[recovery] step={train_steps + 1}: "
                    "rebuilt optimizer after rejected non-finite update.",
                    flush=True,
                )
                continue

            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            if len(step_times) > 10:
                step_times.pop(0)

            components["tris_cull_before"] = int(cull_before)
            components["tris_cull_after"] = int(cull_after)

            last_grad_norms  = grad_norms
            last_components  = components
            total_loss      += loss_val
            train_steps     += 1
            current_view_idx += 1
            consecutive_render_skips = 0
            consecutive_rejected_updates = 0

            if train_steps % 10 == 0:
                avg    = total_loss / train_steps
                sps    = 1.0/(sum(step_times)/len(step_times)) if step_times else 0
                eta_10 = 10.0/sps if sps > 0 else 0
                c      = last_components
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {train_steps:4d} | {view.filename} | avg {avg:.4f} | "
                    f"l1={c.get('l1',0):.4f} ssim={c.get('dssim',0):.4f} "
                    f"lo={c.get('lo',0):.4f} ls={c.get('ls',0):.4f} "
                    f"anchor={c.get('anchor',0):.4f} rb={int(c.get('rolled_back',0))} | "
                    f"N={model.num_triangles()} | "
                    f"tris_used={c.get('tris_cull_after', model.num_triangles())}/{c.get('tris_cull_before', model.num_triangles())} | "
                    f"crop={c.get('crop_w', W)}x{c.get('crop_h', H)} "
                    f"({100*c.get('crop_area_frac',1.0):.1f}%) | "
                    f"chunk {current_chunk_idx+1}/{N_CHUNKS} | "
                    f"lr_v={cur_lr:.2e} | "
                    f"{sps:.1f} sps (~{eta_10:.0f}s/10 steps)"
                )

            # ---- densification ----
            if (CFG.get("enable_densification", True)
                    and DENSIFY_EVERY
                    and train_steps % DENSIFY_EVERY == 0):
                model = densify_model(model, max_triangles=MAX_TRI)
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model, train_steps_so_far=train_steps)
                anchor_vertices = model.vertices.detach().clone()

            # ---- pruning ----
            if (CFG.get("enable_pruning", True)
                    and PRUNE_EVERY
                    and train_steps % PRUNE_EVERY == 0
                    and model.num_triangles() > 10):
                if acc_max_weights.shape[0] == model.num_triangles():
                    model = prune_model(model, acc_max_weights,
                                        tau_prune=TAU_PRUNE)
                    optimizer, scheduler = build_optimizer_and_scheduler(
                        model, train_steps_so_far=train_steps)
                    acc_max_weights = torch.zeros(model.num_triangles(),
                                                  device=device)
                    anchor_vertices = model.vertices.detach().clone()

            # ---- chunk cycling (multi-chunk mode) ----
            if N_CHUNKS > 1 and current_view_idx % len(vis_views) == 0:
                current_chunk_idx = (current_chunk_idx + 1) % N_CHUNKS
                chunk_indices     = all_chunks[current_chunk_idx]
                vis_views         = get_vis_views(chunk_indices)
                crop_reference_vertices = build_crop_reference_vertices(chunk_indices)

                # Append new chunk's triangles to the model
                new_idx  = chunk_indices
                nv = np.array([all_triangles[i].vertices for i in new_idx],
                              dtype=np.float32)
                nc = np.array([all_triangles[i].color    for i in new_idx],
                              dtype=np.float32)
                no = np.array([all_triangles[i].opacity  for i in new_idx],
                              dtype=np.float32)
                ns = np.array([all_triangles[i].sigma    for i in new_idx],
                              dtype=np.float32)
                with torch.no_grad():
                    cat_v   = torch.cat([model.vertices.detach(),
                                         torch.from_numpy(nv).to(device)], 0)
                    nc_t    = torch.from_numpy(nc).to(device).clamp(1e-4, 1-1e-4)
                    no_t    = torch.from_numpy(no).to(device).clamp(1e-4, 1-1e-4)
                    ns_t    = torch.from_numpy(ns).to(device).clamp(1e-4)
                    cat_c   = torch.cat([model.color_logit.detach(),
                                         torch.logit(nc_t)], 0)
                    cat_o   = torch.cat([model.opacity_logit.detach(),
                                         torch.logit(no_t)], 0)
                    cat_s   = torch.cat([model.log_sigma.detach(),
                                         torch.log(ns_t)], 0)
                model = TriangleSplatModel(
                    cat_v, torch.sigmoid(cat_c),
                    torch.sigmoid(cat_o), torch.exp(cat_s)).to(device)
                with torch.no_grad():
                    model.color_logit.copy_(cat_c)
                    model.opacity_logit.copy_(cat_o)
                    model.log_sigma.copy_(cat_s)
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model, train_steps_so_far=train_steps)
                acc_max_weights = torch.zeros(model.num_triangles(), device=device)
                anchor_vertices = model.vertices.detach().clone()
                print(f"[chunk] -> {current_chunk_idx+1}/{N_CHUNKS}  "
                      f"model has {model.num_triangles()} triangles")

            # ---- checkpoint ----
            if train_steps % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler,
                                train_steps, total_loss,
                                chunk_indices, CKPT_PATH)

            # ---- PNG export ----
            if train_steps % SAVE_IMG == 0:
                # Old interactive-camera preview render. This is useful only if
                # you manually moved the pygame camera to a good viewpoint.
                img_path = os.path.join(
                    IMG_DIR, f"render_step_{train_steps:05d}.png")
                save_render_png(model, device, W, H, img_path)

            # ---- Training-camera export ----
            # This is the better progress export: it saves from the actual
            # COLMAP camera angle currently used for the loss.
            SAVE_TRAIN_VIEW = CFG.get("save_training_view_every", 100)
            if SAVE_TRAIN_VIEW and train_steps % SAVE_TRAIN_VIEW == 0:
                save_training_view_render_png(
                    model=model,
                    device=device,
                    view=view,
                    target=target,
                    W=W,
                    H=H,
                    focal=Globals.focalLength,
                    step=train_steps,
                    out_dir=CFG.get("training_render_dir", "renders_training_views"),
                    crop=crop,
                )

            if CFG.get("max_train_steps") is not None and train_steps >= CFG["max_train_steps"]:
                print(f"Reached max_train_steps={CFG['max_train_steps']}; exiting.")
                running = False

        # ---- display (stable interactive camera, not training camera) ----
        do_display = ((not train_mode) or CFG.get("preview_during_training", False)) and (train_steps % DISPLAY_EVERY == 0)
        if do_display:
            screen.fill((0, 0, 0))
            preview = render_preview_canvas(model, device, W, H)
            blit_canvas_to_pygame(preview, screen)

            if Globals.show_debug:
                sps    = (1.0/(sum(step_times)/len(step_times))
                          if step_times else 0.0)
                eta_10 = 10.0/sps if sps > 0 else 0.0
                c      = last_components
                cur_lr = optimizer.param_groups[0]["lr"]
                vn     = (vis_views[current_view_idx % len(vis_views)].filename
                          if train_mode and vis_views else "—")
                lines = [
                    f"Points: {len(points)}  Triangles: {model.num_triangles()}",
                    f"Chunks: {current_chunk_idx+1}/{N_CHUNKS}  "
                    f"chunk_k={CFG['chunk_k']}",
                    f"Focal: {Globals.focalLength:.1f} px  |  {device}",
                    f"Train: {'ON' if train_mode else 'OFF — press T'}  "
                    f"step {train_steps}",
                    f"Avg loss: {total_loss/max(1,train_steps):.4f}",
                    f"l1={c.get('l1',0):.4f}  ssim={c.get('dssim',0):.4f}  "
                    f"lo={c.get('lo',0):.4f}",
                    f"lr_vertices: {cur_lr:.2e}",
                    f"Speed: {sps:.1f} sps  (~{eta_10:.0f}s / 10 steps)",
                    f"Training view: {vn}",
                    f"Crop train: {CFG.get('use_chunk_crop', True)}  "
                    f"last={c.get('crop_w', W)}x{c.get('crop_h', H)}",
                    f"Ckpt: {CKPT_PATH}  (Ctrl+S to save now)",
                    f"Images: {IMG_DIR}/render_step_XXXXX.png",
                    "F1 debug | WASD/arrows | T train | Ctrl+S save",
                ]
                for i, line in enumerate(lines):
                    surf = font.render(line, True, (255, 255, 0))
                    screen.blit(surf, (10, 10 + i * 22))

            pygame.display.flip()

        clock.tick(0)   # uncapped — training runs as fast as possible

    # ---- auto-save on exit ----
    save_checkpoint(model, optimizer, scheduler,
                    train_steps, total_loss, chunk_indices, CKPT_PATH)
    save_render_png(model, device, W, H,
                    os.path.join(IMG_DIR,
                                 f"render_final_step_{train_steps}.png"))
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
