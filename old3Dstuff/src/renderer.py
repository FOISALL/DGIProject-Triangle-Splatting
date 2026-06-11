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


# ===========================================================================
# CONFIG  — only change things here
# ===========================================================================

CFG = {
    # --- scene paths ---
    "sparse_dir":  "south-building/sparse",
    "images_dir":  "south-building/images",
    "points_file": "south-building/sparse/points3D.txt",

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
    "chunk_k": 500,

    # --- multi-chunk ---
    # 1  = train one fixed chunk (original behaviour)
    # >1 = cycle through this many spatially distinct chunks,
    #      adding each chunk's triangles to the model as training progresses
    "n_chunks": 1,

    # --- densification / pruning ---
    "densify_every":  100,
    "prune_every":    200,
    "max_triangles":  2000,   # hard cap; raise for fuller reconstruction
    "tau_prune":      0.005,

    # --- learning rates ---
    "lr_color_opacity_sigma": 1e-4,
    "lr_vertices_init":       1e-4,   # vertex LR at step 0
    "lr_vertices_final":      1e-6,   # vertex LR after decay_steps
    "lr_decay_steps":         3000,

    # --- display ---
    "display_every":    20,    # pygame refresh every N training steps
    "screen_w":        1200,
    "screen_h":         800,

    # --- saving ---
    "checkpoint_path":  "checkpoint.pt",   # auto-saved every save_every steps
    "save_every":        200,              # checkpoint interval (steps)
    "save_image_every":  100,              # PNG export interval (steps)
    "image_dir":         "renders",        # folder for PNG exports
}

# ===========================================================================


# ---------------------------------------------------------------------------
# Globals  (camera state, not trained)
# ---------------------------------------------------------------------------

class Globals:
    SCREEN_WIDTH  = CFG["screen_w"]
    SCREEN_HEIGHT = CFG["screen_h"]
    focalLength   = 500.0
    pointcloudData: list[Point3D] = []
    cameraPosition = np.array([0.0, 0.0, -3.001], dtype=np.float32)
    R              = np.eye(3, dtype=np.float32)
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
# Differentiable projection
# ---------------------------------------------------------------------------

def project_vertices(vertices, cam_pos, R, focal, W, H):
    p_cam  = (vertices - cam_pos.unsqueeze(0).unsqueeze(0)) @ R
    z      = p_cam[..., 2]
    valid  = (z > 1e-4).all(dim=1)
    z_safe = z.clamp(min=1e-4)
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
        ni   = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        di   = -(ni * a).sum(-1)
        normals.append(ni)
        offsets.append(di)
    ns    = torch.stack(normals, dim=1)
    ds    = torch.stack(offsets, dim=1)
    a_len = (v[:, 2] - v[:, 1]).norm(dim=-1)
    b_len = (v[:, 2] - v[:, 0]).norm(dim=-1)
    c_len = (v[:, 1] - v[:, 0]).norm(dim=-1)
    perim = (a_len + b_len + c_len).clamp(min=1e-7)
    inc   = (a_len.unsqueeze(-1) * v[:, 0]
           + b_len.unsqueeze(-1) * v[:, 1]
           + c_len.unsqueeze(-1) * v[:, 2]) / perim.unsqueeze(-1)
    phi_s = ((inc.unsqueeze(1) * ns).sum(-1) + ds).max(dim=1).values
    return ns, ds, phi_s


# ---------------------------------------------------------------------------
# Differentiable render
# ---------------------------------------------------------------------------

def render_differentiable(model, cam_pos, R, focal, W, H,
                          sort_order, return_max_weights=False):
    screen_xy, valid = project_vertices(model.vertices, cam_pos, R, focal, W, H)
    ns_all, ds_all, phi_s_all = compute_sdf_params(screen_xy)

    canvas        = torch.zeros(H, W, 3, device=model.vertices.device)
    transmittance = torch.ones(H, W,    device=model.vertices.device)
    colors    = model.colors
    opacities = model.opacities
    sigmas    = model.sigmas
    device    = model.vertices.device
    max_weights = torch.zeros(model.num_triangles(), device=device)

    for idx in sort_order:
        if not valid[idx]:
            continue
        phi_s = phi_s_all[idx]
        if phi_s >= 0:
            continue

        vxy   = screen_xy[idx]
        x_min = max(0,   int(vxy[:, 0].min().item()))
        x_max = min(W-1, int(vxy[:, 0].max().item()))
        y_min = max(0,   int(vxy[:, 1].min().item()))
        y_max = min(H-1, int(vxy[:, 1].max().item()))
        if x_min > x_max or y_min > y_max:
            continue

        xs = torch.arange(x_min, x_max+1, device=device, dtype=torch.float32)
        ys = torch.arange(y_min, y_max+1, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        px = grid_x.reshape(-1)
        py = grid_y.reshape(-1)
        if px.numel() == 0:
            continue

        pts     = torch.stack([px, py], dim=-1).to(dtype=ns_all.dtype)
        pts_exp = pts.unsqueeze(1).expand(-1, 3, -1)
        phi_p   = ((pts_exp * ns_all[idx].unsqueeze(0)).sum(-1)
                   + ds_all[idx].unsqueeze(0)).max(dim=1).values

        ratio     = (phi_p / phi_s.clamp(max=-1e-7)).clamp(0.0, 1.0)
        influence = ratio.pow(sigmas[idx])
        alpha     = opacities[idx] * influence

        flat_idx = py.long() * W + px.long()
        T        = transmittance.reshape(-1)[flat_idx]

        if return_max_weights:
            max_weights[idx] = (T * opacities[idx]).max().detach()

        color_contrib = (colors[idx].unsqueeze(0)
                         * (alpha * T).unsqueeze(-1) * 255.0)
        canvas = canvas.reshape(-1, 3).scatter_add(
            0, flat_idx.unsqueeze(-1).expand(-1, 3),
            color_contrib).reshape(H, W, 3)

        trans_flat           = transmittance.reshape(-1).clone()
        trans_flat[flat_idx] = T * (1.0 - alpha).detach()
        transmittance        = trans_flat.reshape(H, W)

    return (canvas, max_weights) if return_max_weights else canvas


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


def compute_loss(canvas, target, model, lam=0.2, beta1=0.01, beta4=0.001):
    p, t  = canvas / 255.0, target / 255.0
    l1    = (p - t).abs().mean()
    dssim = _ssim_loss(canvas, target)
    photo = (1.0 - lam) * l1 + lam * dssim
    o     = model.opacities
    lo    = -(o*torch.log(o+1e-7) + (1-o)*torch.log(1-o+1e-7)).mean()
    v     = model.vertices
    ls    = -0.5 * torch.cross(
        v[:,1]-v[:,0], v[:,2]-v[:,0], dim=-1).norm(dim=-1).mean()
    total = photo + beta1*lo + beta4*ls
    return total, {"photo": photo.item(), "l1": l1.item(),
                   "dssim": dssim.item(), "lo": lo.item(), "ls": ls.item()}


# ---------------------------------------------------------------------------
# Densification  (paper Sec. 3.2)
# ---------------------------------------------------------------------------

def densify_model(model, max_triangles=2000, tau_small=1e-3):
    N = model.num_triangles()
    if N >= max_triangles:
        return model
    n_to_add = min(max(N // 4, 1), max_triangles - N)
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


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(model, optimizer, scheduler, target,
                  cam_pos, R, focal, W, H, sort_order, acc_max_weights):
    optimizer.zero_grad()
    zero = {"vertices":0.0, "sigma":0.0, "opacity":0.0, "color":0.0}
    if not sort_order:
        return 0.0, zero, {}

    canvas, max_w = render_differentiable(
        model, cam_pos, R, focal, W, H, sort_order, return_max_weights=True)

    if acc_max_weights.shape[0] == model.num_triangles():
        acc_max_weights[:] = torch.max(acc_max_weights, max_w)

    loss, components = compute_loss(canvas, target, model)
    if not loss.requires_grad:
        return float(loss.item()), zero, components

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
    scheduler.step()
    return float(loss.item()), grad_norms, components


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    W, H = CFG["screen_w"], CFG["screen_h"]
    os.makedirs(CFG["image_dir"], exist_ok=True)

    # ---- data ----
    Globals.pointcloudData = load_points3D(CFG["points_file"])
    train_views, test_views, colmap_camera = load_dataset(
        CFG["sparse_dir"], CFG["images_dir"], W, H)
    Globals.focalLength = colmap_focal_to_renderer(colmap_camera, W, H)

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

    vis_views = get_vis_views(chunk_indices)
    print(f"Chunk 0: {len(chunk_indices)} tris | "
          f"{len(vis_views)} visible views")

    # ---- model: try checkpoint first ----
    ckpt = load_checkpoint(CFG["checkpoint_path"], device)
    if ckpt is not None:
        model         = model_from_checkpoint(ckpt, device)
        train_steps   = ckpt["train_steps"]
        total_loss    = ckpt["total_loss"]
        chunk_indices = ckpt.get("chunk_indices", chunk_indices)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, train_steps_so_far=train_steps)
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
            target_np = load_target_image(view, CFG["images_dir"], W, H)
            target    = torch.from_numpy(target_np).to(device)
            cam_t     = torch.from_numpy(view.cam_pos).to(device)
            R_t       = torch.from_numpy(view.R).to(device)

            train_sorted = sort_triangles_by_depth(model.vertices, cam_t, R_t)

            N_cur = model.num_triangles()
            if acc_max_weights.shape[0] != N_cur:
                new_acc = torch.zeros(N_cur, device=device)
                old_n   = min(acc_max_weights.shape[0], N_cur)
                new_acc[:old_n] = acc_max_weights[:old_n]
                acc_max_weights = new_acc

            loss_val, grad_norms, components = training_step(
                model, optimizer, scheduler, target,
                cam_t, R_t, Globals.focalLength, W, H,
                train_sorted, acc_max_weights)

            elapsed = time.perf_counter() - t0
            step_times.append(elapsed)
            if len(step_times) > 10:
                step_times.pop(0)

            last_grad_norms  = grad_norms
            last_components  = components
            total_loss      += loss_val
            train_steps     += 1
            current_view_idx += 1

            if train_steps % 10 == 0:
                avg    = total_loss / train_steps
                sps    = 1.0/(sum(step_times)/len(step_times)) if step_times else 0
                eta_10 = 10.0/sps if sps > 0 else 0
                c      = last_components
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {train_steps:4d} | {view.filename} | avg {avg:.4f} | "
                    f"l1={c.get('l1',0):.4f} ssim={c.get('dssim',0):.4f} "
                    f"lo={c.get('lo',0):.4f} ls={c.get('ls',0):.4f} | "
                    f"N={model.num_triangles()} | "
                    f"chunk {current_chunk_idx+1}/{N_CHUNKS} | "
                    f"lr_v={cur_lr:.2e} | "
                    f"{sps:.1f} sps (~{eta_10:.0f}s/10 steps)"
                )

            # ---- densification ----
            if train_steps % DENSIFY_EVERY == 0:
                model = densify_model(model, max_triangles=MAX_TRI)
                optimizer, scheduler = build_optimizer_and_scheduler(
                    model, train_steps_so_far=train_steps)

            # ---- pruning ----
            if train_steps % PRUNE_EVERY == 0 and model.num_triangles() > 10:
                if acc_max_weights.shape[0] == model.num_triangles():
                    model = prune_model(model, acc_max_weights,
                                        tau_prune=TAU_PRUNE)
                    optimizer, scheduler = build_optimizer_and_scheduler(
                        model, train_steps_so_far=train_steps)
                    acc_max_weights = torch.zeros(model.num_triangles(),
                                                  device=device)

            # ---- chunk cycling (multi-chunk mode) ----
            if N_CHUNKS > 1 and current_view_idx % len(vis_views) == 0:
                current_chunk_idx = (current_chunk_idx + 1) % N_CHUNKS
                chunk_indices     = all_chunks[current_chunk_idx]
                vis_views         = get_vis_views(chunk_indices)

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
                print(f"[chunk] -> {current_chunk_idx+1}/{N_CHUNKS}  "
                      f"model has {model.num_triangles()} triangles")

            # ---- checkpoint ----
            if train_steps % SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scheduler,
                                train_steps, total_loss,
                                chunk_indices, CKPT_PATH)

            # ---- PNG export ----
            if train_steps % SAVE_IMG == 0:
                img_path = os.path.join(
                    IMG_DIR, f"render_step_{train_steps:05d}.png")
                save_render_png(model, device, W, H, img_path)

        # ---- display (stable interactive camera, not training camera) ----
        do_display = (not train_mode) or (train_steps % DISPLAY_EVERY == 0)
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