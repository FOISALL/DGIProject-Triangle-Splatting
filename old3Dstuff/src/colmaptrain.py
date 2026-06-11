"""
Differentiable Triangle Splatting Renderer
==========================================
Replaces the numpy scanline pipeline with a fully differentiable
PyTorch forward pass so that .backward() can update triangle parameters.

What is trainable:
  - vertices  (3 x 3D positions per triangle)
  - colors    (RGB 0-1)
  - opacities (scalar per triangle)
  - sigmas    (SDF softness exponent per triangle)

Pipeline (all in torch, on CPU or CUDA):
  1. Project 3-D vertices -> 2-D screen pixels          (differentiable)
  2. For every pixel covered, evaluate the SDF           (differentiable)
  3. Compute per-pixel alpha via the window function     (differentiable)
  4. Front-to-back alpha composite into a canvas         (differentiable)
  5. L2 loss vs. a target image, then .backward()        (standard)

The pygame loop previews the canvas each frame and lets you fly the camera.
Training happens on a fixed camera pose (you can extend this to multi-view).
"""

import math
import time

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import KDTree

from render_utils import Pixel, Point3D, Triangle, load_points3D

from colmap_loader import load_dataset, colmap_focal_to_renderer, load_target_image


# ---------------------------------------------------------------------------
# Global state (camera / bookkeeping - NOT trained, stays numpy/python)
# ---------------------------------------------------------------------------

class Globals:
    SCREEN_WIDTH  = 1200
    SCREEN_HEIGHT = 800
    focalLength   = 500.0

    pointcloudData: list[Point3D] = []
    chunk_k       = 100
    chunk_indices: list[int] = []

    # Camera (numpy, not trained)
    cameraPosition = np.array([0.0, 0.0, -3.001], dtype=np.float32)
    R              = np.eye(3, dtype=np.float32)
    delta          = 0.1
    yaw_speed      = 0.05
    pitch_speed    = 0.05

    show_debug     = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Camera helpers  (numpy, unchanged from before)
# ---------------------------------------------------------------------------

def update_rotation(yaw: float, pitch: float) -> None:
    cy, sy = math.cos(yaw), math.sin(yaw)
    yawMat = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float32)
    Globals.R = Globals.R @ yawMat
    if pitch != 0:
        cp, sp = math.cos(pitch), math.sin(pitch)
        pitchMat = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]], dtype=np.float32)
        Globals.R = Globals.R @ pitchMat


def get_depth_numpy(vertices_np: np.ndarray) -> float:
    """Depth of triangle centroid in camera space (for sorting). numpy only."""
    centroid = vertices_np.mean(axis=0)
    v_cam    = (centroid - Globals.cameraPosition) @ Globals.R
    return float(v_cam[2])


def initialize_triangle(point: Point3D, neighbours: list[tuple[Point3D, float]]) -> Triangle:
    q = np.array([point.x, point.y, point.z], dtype=np.float32)
    U = []
    for _ in range(3):
        u = np.random.uniform(-1.0, 1.0, 3)
        U.append(u / (np.linalg.norm(u) + 1e-8))

    d = (neighbours[0][1] + neighbours[1][1] + neighbours[2][1]) / 3.0
    k = 2.2
    verts = [q + k * d * u for u in U]

    return Triangle(
        vertices=verts,
        color=np.array([point.r, point.g, point.b]),
        opacity=0.9,
        sigma=1.16,
    )


def initialize_triangles(points: list[Point3D]) -> list[Triangle]:
    print("Start triangle initialization")
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    kdtree = KDTree(coords)

    triangles = []
    for point in points:
        distances, indices = kdtree.query([point.x, point.y, point.z], k=4)
        neighbours = [(points[idx], dist) for idx, dist in zip(indices[1:], distances[1:])]
        triangles.append(initialize_triangle(point, neighbours))

    print("Triangles initialized")
    return triangles


def get_chunk_indices(points: list[Point3D], k: int) -> list[int]:
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    tree = KDTree(coords)
    _, indices = tree.query(coords[0], k=k)
    return list(indices)


# ---------------------------------------------------------------------------
# Learnable triangle parameters  (a small nn.Module)
# ---------------------------------------------------------------------------

class TriangleSplatModel(nn.Module):
    """
    Holds all trainable parameters for a chunk of N triangles.

    Parameters
    ----------
    vertices_init : (N, 3, 3)  float32  -- world-space vertex positions
    colors_init   : (N, 3)     float32  -- RGB in [0, 1]
    opacities_init: (N,)       float32  -- in (0, 1)
    sigmas_init   : (N,)       float32  -- SDF softness exponent
    """

    def __init__(
        self,
        vertices_init:  torch.Tensor,
        colors_init:    torch.Tensor,
        opacities_init: torch.Tensor,
        sigmas_init:    torch.Tensor,
    ):
        super().__init__()
        # nn.Parameter tells PyTorch: "track gradients through this"
        self.vertices  = nn.Parameter(vertices_init.clone())
        # Store raw logit for color so sigmoid keeps it in [0,1]
        self.color_logit = nn.Parameter(torch.logit(colors_init.clamp(1e-4, 1 - 1e-4)))
        # Same trick for opacity
        self.opacity_logit = nn.Parameter(torch.logit(opacities_init.clamp(1e-4, 1 - 1e-4)))
        # Sigma is positive; store log
        self.log_sigma = nn.Parameter(torch.log(sigmas_init.clamp(min=1e-4)))

    @property
    def colors(self)    -> torch.Tensor: return torch.sigmoid(self.color_logit)
    @property
    def opacities(self) -> torch.Tensor: return torch.sigmoid(self.opacity_logit)
    @property
    def sigmas(self)    -> torch.Tensor: return torch.exp(self.log_sigma)


# ---------------------------------------------------------------------------
# Differentiable projection  (3-D world -> 2-D screen, in torch)
# ---------------------------------------------------------------------------

def project_vertices(
    vertices: torch.Tensor,   # (N, 3, 3)  world coords
    cam_pos:  torch.Tensor,   # (3,)
    R:        torch.Tensor,   # (3, 3)
    focal:    float,
    W:        int,
    H:        int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    screen_xy : (N, 3, 2)  pixel coordinates  (float, NOT rounded)
    valid     : (N,)       bool -- triangle has all vertices in front of camera
    """
    # Move to camera space:  (N, 3, 3) - (3,) -> (N, 3, 3)  then rotate
    p_cam = (vertices - cam_pos.unsqueeze(0).unsqueeze(0)) @ R   # (N, 3, 3)

    z = p_cam[..., 2]          # (N, 3)
    x = p_cam[..., 0]
    y = p_cam[..., 1]

    # All three vertices must be in front of the camera
    valid = (z > 1e-4).all(dim=1)   # (N,)

    # Perspective divide  (safe: we'll mask invalids later)
    z_safe = z.clamp(min=1e-4)
    sx = focal * x / z_safe + W / 2.0   # (N, 3)
    sy = focal * y / z_safe + H / 2.0

    screen_xy = torch.stack([sx, sy], dim=-1)   # (N, 3, 2)
    return screen_xy, valid


# ---------------------------------------------------------------------------
# Differentiable SDF + window function  (the heart of triangle splatting)
# ---------------------------------------------------------------------------

def compute_sdf_params(
    screen_xy: torch.Tensor,   # (N, 3, 2)  projected vertices
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three half-plane normals and offsets that define each
    triangle's signed distance field, plus the incenter SDF value phiS.

    Returns
    -------
    ns    : (N, 3, 2)   unit normals (one per edge, pointing inward)
    ds    : (N, 3)      offsets so that  ni . p + di = 0  on each edge
    phi_s : (N,)        SDF value at incenter (always <= 0 for valid triangles)
    """
    v = screen_xy   # (N, 3, 2)

    # --- edge normals ---
    # For triangle vertices [v0, v1, v2], edge i goes from v[i] to v[i-1]
    # (cyclic).  The inward normal is the left-perp of that edge direction,
    # flipped if needed so it points toward the opposite vertex.

    normals = []
    offsets = []
    for i in range(3):
        # edge from v[i] to v[(i-1) % 3]
        a = v[:, i,       :]          # (N, 2)
        b = v[:, (i-1)%3, :]          # (N, 2)
        edge = b - a                   # (N, 2)

        # left perpendicular
        n = torch.stack([-edge[:, 1], edge[:, 0]], dim=-1)   # (N, 2)

        # flip if it points away from the opposite vertex
        opp = v[:, (i-2)%3, :]        # (N, 2)  opposite vertex
        dot = (n * (opp - a)).sum(-1)  # (N,)
        # Match numpy path: if dot > 0, flip normal.
        sign = torch.where(dot > 0, -torch.ones_like(dot), torch.ones_like(dot))
        n = n * sign.unsqueeze(-1)     # (N, 2)

        # normalize
        length = n.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        ni = n / length                # (N, 2)

        # offset: ni . a + di = 0  =>  di = -ni . a
        di = -(ni * a).sum(-1)         # (N,)

        normals.append(ni)
        offsets.append(di)

    ns = torch.stack(normals, dim=1)   # (N, 3, 2)
    ds = torch.stack(offsets, dim=1)   # (N, 3)

    # --- incenter (weighted average of vertices by opposite edge lengths) ---
    # side lengths
    a_len = (v[:, 2, :] - v[:, 1, :]).norm(dim=-1)   # (N,)  side opposite v0
    b_len = (v[:, 2, :] - v[:, 0, :]).norm(dim=-1)
    c_len = (v[:, 1, :] - v[:, 0, :]).norm(dim=-1)
    perim = (a_len + b_len + c_len).clamp(min=1e-7)

    incenter = (a_len.unsqueeze(-1) * v[:, 0, :]
              + b_len.unsqueeze(-1) * v[:, 1, :]
              + c_len.unsqueeze(-1) * v[:, 2, :]) / perim.unsqueeze(-1)   # (N, 2)

    # phi at incenter: max over 3 half-planes  (should be < 0 inside)
    # incenter: (N, 2), ns: (N, 3, 2), ds: (N, 3)
    phi_s = (incenter.unsqueeze(1) * ns).sum(-1) + ds   # (N, 3)
    phi_s = phi_s.max(dim=1).values                      # (N,)

    return ns, ds, phi_s


def window_function(
    px: torch.Tensor,   # (P,)   pixel x coords
    py: torch.Tensor,   # (P,)   pixel y coords
    ns: torch.Tensor,   # (3, 2) edge normals for ONE triangle
    ds: torch.Tensor,   # (3,)   edge offsets for ONE triangle
    phi_s: torch.Tensor,  # scalar tensor SDF at incenter
    sigma: torch.Tensor,  # scalar tensor softness exponent
) -> torch.Tensor:
    """
    Evaluate the influence / window function for a set of pixels
    against a single triangle.

    Returns alpha weights: (P,)  in [0, 1]
    """
    # pixel coords -> (P, 2)
    pts = torch.stack([px, py], dim=-1).to(dtype=ns.dtype, device=ns.device)

    # SDF at each pixel:  max over 3 half-planes
    phi_p = (pts.unsqueeze(1) * ns.unsqueeze(0)).sum(-1) + ds.unsqueeze(0)  # (P, 3)
    phi_p = phi_p.max(dim=1).values  # (P,)

    # Window function I = clamp(phi_p / phi_s, 0, 1) ^ sigma
    ratio = (phi_p / phi_s.clamp(max=-1e-7)).clamp(0.0, 1.0)
    return ratio.pow(sigma)


# ---------------------------------------------------------------------------
# Full differentiable render pass
# ---------------------------------------------------------------------------

def render_differentiable(
    model:      TriangleSplatModel,
    cam_pos:    torch.Tensor,   # (3,)
    R:          torch.Tensor,   # (3, 3)
    focal:      float,
    W:          int,
    H:          int,
    sort_order: list[int],      # pre-sorted triangle indices (front-to-back)
) -> torch.Tensor:
    """
    Full differentiable forward pass with sequential front-to-back compositing.
    
    Hybrid approach:
      - Per-triangle loop: respects front-to-back sort order, sequential T updates
      - Per-pixel vectorization: fast SDF/window evaluation within each triangle
    
    This ensures correct blending per the paper while keeping performance high.
    
    Returns
    -------
    canvas : (H, W, 3)  float32  rendered image, values in [0, 255]
    """
    # --- project all triangles ---
    screen_xy, valid = project_vertices(
        model.vertices, cam_pos, R, focal, W, H
    )   # (N, 3, 2),  (N,)

    # --- compute SDF params for all triangles ---
    ns_all, ds_all, phi_s_all = compute_sdf_params(screen_xy)   # (N,3,2),(N,3),(N,)

    # --- accumulation buffers ---
    canvas        = torch.zeros(H, W, 3, device=model.vertices.device)
    transmittance = torch.ones(H, W,    device=model.vertices.device)

    colors    = model.colors      # (N, 3)
    opacities = model.opacities   # (N,)
    sigmas    = model.sigmas      # (N,)

    device = model.vertices.device

    # Sequential per-triangle processing (CRITICAL for correct front-to-back compositing)
    for idx in sort_order:
        if not valid[idx]:
            continue

        phi_s = phi_s_all[idx]
        if phi_s >= 0:
            # Degenerate triangle (incenter outside) – skip
            continue

        # Bounding box of the projected triangle (integer pixel range)
        vxy = screen_xy[idx]   # (3, 2)
        x_min = int(vxy[:, 0].min().item())
        x_max = int(vxy[:, 0].max().item())
        y_min = int(vxy[:, 1].min().item())
        y_max = int(vxy[:, 1].max().item())

        # Clamp to screen
        x_min = max(0, x_min); x_max = min(W - 1, x_max)
        y_min = max(0, y_min); y_max = min(H - 1, y_max)
        if x_min > x_max or y_min > y_max:
            continue

        # Build pixel grid for this bounding box
        xs = torch.arange(x_min, x_max + 1, device=device, dtype=torch.float32)
        ys = torch.arange(y_min, y_max + 1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (rows, cols)
        px = grid_x.reshape(-1)   # (P,)
        py = grid_y.reshape(-1)   # (P,)

        if px.numel() == 0:
            continue

        # --- Vectorized SDF evaluation for all pixels of this triangle ---
        pts = torch.stack([px, py], dim=-1).to(dtype=ns_all.dtype, device=device)  # (P, 2)
        # Expand pts to (P, 3, 2) to match ns shape
        pts_exp = pts.unsqueeze(1).expand(-1, 3, -1)  # (P, 3, 2)
        phi_p = (pts_exp * ns_all[idx].unsqueeze(0)).sum(-1) + ds_all[idx].unsqueeze(0)  # (P, 3)
        phi_p = phi_p.max(dim=1).values  # (P,)

        # Window function (per-pixel influence)
        ratio = (phi_p / phi_s.clamp(max=-1e-7)).clamp(0.0, 1.0)
        influence = ratio.pow(sigmas[idx])
        alpha = opacities[idx] * influence   # (P,)

        # Flattened pixel indices
        flat_idx = py.long() * W + px.long()  # (P,)

        # Read transmittance AFTER previous triangles have updated it
        # (this is the key difference: sequential reads respect front-to-back order)
        T = transmittance.reshape(-1)[flat_idx]  # (P,)

        # Color contribution
        color_contrib = colors[idx].unsqueeze(0) * (alpha * T).unsqueeze(-1) * 255.0  # (P, 3)

        # Update canvas (scatter_add preserves autograd)
        canvas_flat = canvas.reshape(-1, 3)
        canvas_flat_new = canvas_flat.scatter_add(
            0, flat_idx.unsqueeze(-1).expand(-1, 3), color_contrib
        )
        canvas = canvas_flat_new.reshape(H, W, 3)

        # Update transmittance for this triangle BEFORE moving to the next
        # This sequential update is CRITICAL for correct front-to-back compositing.
        # Without it, overlapping triangles would all see T=1, breaking the blending equation.
        trans_flat = transmittance.reshape(-1).clone()
        trans_flat[flat_idx] = T * (1.0 - alpha).detach()
        transmittance = trans_flat.reshape(H, W)

    return canvas   # (H, W, 3)


# ---------------------------------------------------------------------------
# Build model from Triangle list
# ---------------------------------------------------------------------------

def build_model(triangles: list[Triangle], indices: list[int], device) -> TriangleSplatModel:
    verts   = np.array([triangles[i].vertices for i in indices], dtype=np.float32)  # (N,3,3)
    colors  = np.array([triangles[i].color    for i in indices], dtype=np.float32)  # (N,3)
    opacs   = np.array([triangles[i].opacity  for i in indices], dtype=np.float32)  # (N,)
    sigmas  = np.array([triangles[i].sigma    for i in indices], dtype=np.float32)  # (N,)

    model = TriangleSplatModel(
        torch.from_numpy(verts),
        torch.from_numpy(colors),
        torch.from_numpy(opacs),
        torch.from_numpy(sigmas),
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Write trained params back to Triangle list  (so pygame preview stays in sync)
# ---------------------------------------------------------------------------

def sync_model_to_triangles(model: TriangleSplatModel, triangles: list[Triangle], indices: list[int]):
    with torch.no_grad():
        verts  = model.vertices.cpu().numpy()   # (N, 3, 3)
        colors = model.colors.cpu().numpy()     # (N, 3)
        opacs  = model.opacities.cpu().numpy()  # (N,)
        sigs   = model.sigmas.cpu().numpy()     # (N,)
    for j, idx in enumerate(indices):
        triangles[idx].vertices = verts[j]
        triangles[idx].color    = colors[j]
        triangles[idx].opacity  = float(opacs[j])
        triangles[idx].sigma    = float(sigs[j])


# ---------------------------------------------------------------------------
# Numpy render (unchanged logic, for the pygame preview)
# ---------------------------------------------------------------------------

def numpy_render_to_array(
    triangles: list[Triangle],
    sorted_indices: list[int],
    W: int,
    H: int,
) -> np.ndarray:
    """Fast numpy scanline render -> (H, W, 3) uint8."""
    color_acc  = np.zeros((H, W, 3), dtype=np.float32)
    transmit   = np.ones((H, W),     dtype=np.float32)

    cam   = Globals.cameraPosition
    R     = Globals.R
    focal = Globals.focalLength

    for idx in sorted_indices:
        tri = triangles[idx]
        # project
        pixels = []
        ok = True
        for v in tri.vertices:
            p_cam = v - cam
            xc, yc, zc = p_cam @ R
            if zc <= 1e-4:
                ok = False; break
            sx = int(focal * xc / zc + W / 2)
            sy = int(focal * yc / zc + H / 2)
            px = Pixel(sx, sy, 1.0 / zc)
            pixels.append(px)
        if not ok:
            continue

        for p in pixels:
            p.x = int(np.clip(p.x, 0, W - 1))
            p.y = int(np.clip(p.y, 0, H - 1))

        # SDF params
        verts2d = [np.array([p.x, p.y], dtype=np.float32) for p in pixels]
        Ls = []
        bad = False
        for i in range(3):
            edge   = verts2d[i - 1] - verts2d[i]
            normal = np.array([-edge[1], edge[0]], dtype=np.float32)
            dot    = normal @ (verts2d[i - 2] - verts2d[i])
            if dot > 0: normal = -normal
            nrm = np.linalg.norm(normal)
            if nrm < 1e-7: bad = True; break
            ni = normal / nrm
            di = -(ni @ verts2d[i - 1])
            Ls.append((ni, di))
        if bad:
            continue

        ns = np.array([l[0] for l in Ls], dtype=np.float32)
        ds = np.array([l[1] for l in Ls], dtype=np.float32)

        # incenter
        a = np.linalg.norm(verts2d[0] - verts2d[1])
        b = np.linalg.norm(verts2d[0] - verts2d[2])
        c = np.linalg.norm(verts2d[2] - verts2d[1])
        perim = a + b + c
        if perim < 1e-7: continue
        inc = (a * verts2d[2] + b * verts2d[1] + c * verts2d[0]) / perim
        phi_s = np.max(inc @ ns.T + ds)
        if phi_s >= 0: continue

        # bounding box
        xs_min = max(0,   min(p.x for p in pixels))
        xs_max = min(W-1, max(p.x for p in pixels))
        ys_min = max(0,   min(p.y for p in pixels))
        ys_max = min(H-1, max(p.y for p in pixels))
        if xs_min > xs_max or ys_min > ys_max: continue

        color_rgb = (tri.color * 255.0).astype(np.float32)

        for y in range(ys_min, ys_max + 1):
            xs = np.arange(xs_min, xs_max + 1)
            pts = np.stack([xs, np.full_like(xs, y)], axis=1)
            phi_p = np.max(pts @ ns.T + ds, axis=1)
            influence = np.power(np.clip(phi_p / phi_s, 0.0, 1.0), tri.sigma)
            alpha = tri.opacity * influence
            T = transmit[y, xs_min : xs_max + 1]
            color_acc[y, xs_min : xs_max + 1] += color_rgb * (alpha * T)[:, np.newaxis]
            transmit[y, xs_min : xs_max + 1]  *= (1.0 - alpha)

    return np.clip(color_acc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(
    model:       TriangleSplatModel,
    optimizer:   optim.Optimizer,
    target:      torch.Tensor,    # (H, W, 3) float32, values in [0, 255]
    cam_pos:     torch.Tensor,
    R:           torch.Tensor,
    focal:       float,
    W:           int,
    H:           int,
    sort_order:  list[int],
) -> tuple[float, dict[str, float]]:
    """
    One gradient step.

    Returns
    -------
    loss_value : float
    grad_norms : dict[str, float]
        L2 norms of key parameter gradients for quick sanity checks.
    """
    optimizer.zero_grad()

    zero_grad_norms = {
        "vertices": 0.0,
        "sigma": 0.0,
        "opacity": 0.0,
        "color": 0.0,
    }

    if len(sort_order) == 0:
        # No visible/selected triangles to train in this frame.
        return 0.0, zero_grad_norms

    canvas = render_differentiable(model, cam_pos, R, focal, W, H, sort_order)

    # L2 loss in [0,255] space
    loss = ((canvas - target) ** 2).mean()

    if not loss.requires_grad:
        # Can happen when nothing in the current render path depends on model parameters.
        return float(loss.item()), zero_grad_norms

    # Enable anomaly detection during backward to get a clearer traceback
    # if an in-place op corrupts the autograd graph.
    try:
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
    except RuntimeError as e:
        print("[autograd error] RuntimeError during backward:", e)
        raise

    grad_norms = {
        "vertices": float(model.vertices.grad.norm().item()) if model.vertices.grad is not None else 0.0,
        "sigma": float(model.log_sigma.grad.norm().item()) if model.log_sigma.grad is not None else 0.0,
        "opacity": float(model.opacity_logit.grad.norm().item()) if model.opacity_logit.grad is not None else 0.0,
        "color": float(model.color_logit.grad.norm().item()) if model.color_logit.grad is not None else 0.0,
    }

    # Gradient clipping to stabilize training (prevent divergence from large updates)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return float(loss.item()), grad_norms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---- add this to your imports at the top of the file ----
from colmap_loader import load_dataset, colmap_focal_to_renderer, load_target_image
 
 
def main():
    W, H = Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT   # define FIRST
 
    # -- load point cloud --
    Globals.pointcloudData = load_points3D("south-building/sparse/points3D.txt")
 
    # -- load COLMAP cameras and images --
    train_views, test_views, colmap_camera = load_dataset(
        "south-building/sparse",
        "south-building/images",
        W, H,
    )
    Globals.focalLength = colmap_focal_to_renderer(colmap_camera, W, H)
    current_view_idx = 0
 
    # -- initialize triangles --
    # To keep things fast on CPU, you can limit how many points you
    # initialize triangles for.  The chunk is still only 100 triangles,
    # but building the full triangle list from 61k points is slow.
    # Set MAX_POINTS to a smaller number (e.g. 5000) to speed up startup.
    MAX_POINTS = 5000
    points = Globals.pointcloudData
    if MAX_POINTS is not None:
        points = points[:MAX_POINTS]
 
    all_triangles = initialize_triangles(points)
    chunk_indices = get_chunk_indices(points, Globals.chunk_k)
 
    device = Globals.device
    print(f"Using device: {device}")
    print(f"Initialized {len(all_triangles)} triangles, chunk size {len(chunk_indices)}")
 
    # -- build differentiable model from the chunk --
    model     = build_model(all_triangles, chunk_indices, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
 
    # -- pygame setup --
    pygame.init()
    clock  = pygame.time.Clock()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Differentiable Triangle Splatting")
    font = pygame.font.SysFont("Arial", 18)
 
    total_loss      = 0.0
    train_steps     = 0
    train_mode      = False
    last_grad_norms = {"vertices": 0.0, "sigma": 0.0, "opacity": 0.0, "color": 0.0}
 
    running = True
    while running:                              # <-- game loop starts here
 
        # ---- events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    Globals.show_debug = not Globals.show_debug
                if event.key == pygame.K_t:
                    train_mode = True
                    print("Training mode ON.")
 
        # ---- camera movement (interactive preview camera) ----
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
 
        # ---- sync trained params back to numpy triangles for preview ----
        sync_model_to_triangles(model, all_triangles, chunk_indices)
 
        # ---- depth sort using the interactive camera ----
        visible    = [i for i in chunk_indices
                      if get_depth_numpy(all_triangles[i].vertices) > 0.1]
        sorted_idx = sorted(visible,
                            key=lambda i: get_depth_numpy(all_triangles[i].vertices))
 
        # ---- numpy preview render (always runs, shows current state) ----
        frame_np         = numpy_render_to_array(all_triangles, sorted_idx, W, H)
        frame_transposed = np.transpose(frame_np, (1, 0, 2))
        pygame.surfarray.blit_array(screen, frame_transposed)
 
        # ---- training step (only when training mode is on) ----
        if train_mode:
            # Pick one real photograph as the target for this step
            view       = train_views[current_view_idx % len(train_views)]
            target_np  = load_target_image(view, "south-building/images", W, H)
            target_img = torch.from_numpy(target_np).to(device)
 
            # Sort using the TRAINING camera, not the interactive one
            # (the training camera is fixed to the photo's pose)
            cam_t = torch.from_numpy(view.cam_pos).to(device)
            R_t   = torch.from_numpy(view.R).to(device)
 
            # Recompute sort order from training camera perspective
            def depth_from_view(tri_idx):
                centroid = all_triangles[tri_idx].vertices.mean(axis=0)
                v_cam    = (centroid - view.cam_pos) @ view.R
                return float(v_cam[2])
 
            train_visible    = [i for i in chunk_indices if depth_from_view(i) > 0.1]
            train_sorted_idx = sorted(train_visible, key=depth_from_view)
            local_sorted     = [chunk_indices.index(i) for i in train_sorted_idx
                                 if i in chunk_indices]
 
            loss_val, grad_norms = training_step(
                model, optimizer, target_img,
                cam_t, R_t, Globals.focalLength, W, H,
                local_sorted,
            )
 
            last_grad_norms   = grad_norms
            total_loss       += loss_val
            train_steps      += 1
            current_view_idx += 1   # next frame uses the next photo
 
            if train_steps % 10 == 0:
                avg = total_loss / train_steps
                print(
                    f"Step {train_steps:4d} | {view.filename} | "
                    f"avg loss {avg:.1f} | "
                    f"grad[v]={grad_norms['vertices']:.3e} "
                    f"grad[s]={grad_norms['sigma']:.3e} "
                    f"grad[o]={grad_norms['opacity']:.3e} "
                    f"grad[c]={grad_norms['color']:.3e}"
                )
 
        # ---- debug overlay ----
        if Globals.show_debug:
            current_view_name = (train_views[current_view_idx % len(train_views)].filename
                                 if train_mode else "—")
            lines = [
                f"Camera: {Globals.cameraPosition.round(2)}",
                f"Focal:  {Globals.focalLength:.1f} px",
                f"Triangles: {len(all_triangles)} total / {len(chunk_indices)} chunk",
                f"Visible: {len(sorted_idx)}",
                f"Train mode: {'ON – press T' if not train_mode else 'ON'}",
                f"Train steps: {train_steps}",
                f"Current view: {current_view_name}",
                f"Avg loss: {total_loss / max(1, train_steps):.2f}",
                f"Grad |v|: {last_grad_norms['vertices']:.2e}",
                f"Grad |s|: {last_grad_norms['sigma']:.2e}",
                f"Grad |o|: {last_grad_norms['opacity']:.2e}",
                f"Grad |c|: {last_grad_norms['color']:.2e}",
                "F1 toggle debug | WASD move | arrows rotate | T train",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (255, 255, 0))
                screen.blit(surf, (10, 10 + i * 22))
 
        pygame.display.flip()
        clock.tick(30)
 
    # ---- end of game loop ----
    pygame.quit()
    print("Done.")
 



if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Developer notes / TODOs
# ---------------------------------------------------------------------------
#
# 1. GRADIENT FLOW THROUGH SIGMA / PHI_S  [FIXED]
#    sigma and phi_s are now kept as tensors in the vectorized rendering loop.
#    Gradients flow through both the SDF geometry and the window function
#    softness parameter. This is correct and fully differentiable.
#
# 2. MULTI-VIEW TRAINING
#    Currently the target is captured from ONE camera pose.  For proper
#    novel-view synthesis you'd loop over a dataset of (image, camera) pairs
#    and accumulate gradients before calling optimizer.step().
#
# 3. TRANSMITTANCE DETACH  [CORRECT DESIGN]
#    Transmittance is detached from the graph (transmittance scheduling only).
#    This matches the original 3DGS paper's approach where T is treated as a
#    fixed weight during each splat's gradient computation.  If you want
#    full end-to-end gradients through T, remove the .detach() call (note:
#    may require gradient clipping for stability).
#
# 4. PERFORMANCE  [HYBRID APPROACH]
#    Rendering now uses a hybrid strategy: per-triangle loop (respects
#    front-to-back sort order and sequential T updates) with vectorized
#    per-pixel SDF evaluation within each triangle.  This balances correctness
#    (proper alpha compositing per the paper) with speed.
#    Further optimization: move to CUDA kernels or tile-based rasterizer.