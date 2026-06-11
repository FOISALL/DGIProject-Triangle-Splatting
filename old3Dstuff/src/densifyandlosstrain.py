"""
Differentiable Triangle Splatting Renderer
==========================================
Full implementation including:
  - Differentiable forward pass (vertices, colors, opacity, sigma)
  - L1 + D-SSIM photometric loss  (paper Eq. 3, main terms)
  - Opacity regularisation loss   (Lo, from paper Eq. 3)
  - Size regularisation loss      (Ls = -0.5 * ||(v1-v0) x (v2-v0)||, paper Eq. 3)
  - Midpoint-subdivision densification (paper Sec. 3.2)
  - Visibility-based pruning       (paper Sec. 3.2)
  - COLMAP camera loading
  - Pygame interactive preview
"""

import math
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import KDTree

from render_utils import Pixel, Point3D, Triangle, load_points3D
from colmap_loader import load_dataset, colmap_focal_to_renderer, load_target_image


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

class Globals:
    SCREEN_WIDTH  = 1200
    SCREEN_HEIGHT = 800
    focalLength   = 500.0

    pointcloudData: list[Point3D] = []
    chunk_k = 500

    cameraPosition = np.array([0.0, 0.0, -3.001], dtype=np.float32)
    R              = np.eye(3, dtype=np.float32)
    delta          = 0.1
    yaw_speed      = 0.05
    pitch_speed    = 0.05
    show_debug     = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Triangle initialisation  (unchanged from before)
# ---------------------------------------------------------------------------

def initialize_triangle(point: Point3D, neighbours) -> Triangle:
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
    print(f"Initialized {len(triangles)} triangles")
    return triangles


def get_chunk_indices(points: list[Point3D], k: int) -> list[int]:
    coords = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    tree = KDTree(coords)
    _, indices = tree.query(coords[0], k=k)
    return list(indices)


# ---------------------------------------------------------------------------
# Learnable model
# ---------------------------------------------------------------------------

class TriangleSplatModel(nn.Module):
    """
    Holds all trainable parameters for N triangles.
    Colors and opacities stored as logits (sigmoid keeps them in [0,1]).
    Sigma stored as log (exp keeps it positive).
    Vertices stored directly (unbounded world-space coords).
    """

    def __init__(self, vertices_init, colors_init, opacities_init, sigmas_init):
        super().__init__()
        self.vertices      = nn.Parameter(vertices_init.clone())
        self.color_logit   = nn.Parameter(torch.logit(colors_init.clamp(1e-4, 1 - 1e-4)))
        self.opacity_logit = nn.Parameter(torch.logit(opacities_init.clamp(1e-4, 1 - 1e-4)))
        self.log_sigma     = nn.Parameter(torch.log(sigmas_init.clamp(min=1e-4)))

    @property
    def colors(self)    -> torch.Tensor: return torch.sigmoid(self.color_logit)
    @property
    def opacities(self) -> torch.Tensor: return torch.sigmoid(self.opacity_logit)
    @property
    def sigmas(self)    -> torch.Tensor: return torch.exp(self.log_sigma)

    def num_triangles(self) -> int:
        return self.vertices.shape[0]


def build_model(triangles: list[Triangle], indices: list[int], device) -> TriangleSplatModel:
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


def sync_model_to_triangles(model: TriangleSplatModel,
                             triangles: list[Triangle],
                             indices: list[int]):
    with torch.no_grad():
        verts  = model.vertices.cpu().numpy()
        colors = model.colors.cpu().numpy()
        opacs  = model.opacities.cpu().numpy()
        sigs   = model.sigmas.cpu().numpy()
    for j, idx in enumerate(indices):
        triangles[idx].vertices = verts[j]
        triangles[idx].color    = colors[j]
        triangles[idx].opacity  = float(opacs[j])
        triangles[idx].sigma    = float(sigs[j])


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def update_rotation(yaw: float, pitch: float) -> None:
    cy, sy = math.cos(yaw), math.sin(yaw)
    yawMat = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float32)
    Globals.R = Globals.R @ yawMat
    if pitch != 0:
        cp, sp = math.cos(pitch), math.sin(pitch)
        pitchMat = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        Globals.R = Globals.R @ pitchMat


def get_depth_numpy(vertices_np: np.ndarray, cam_pos=None, R=None) -> float:
    if cam_pos is None: cam_pos = Globals.cameraPosition
    if R is None:       R       = Globals.R
    centroid = vertices_np.mean(axis=0)
    v_cam    = (centroid - cam_pos) @ R
    return float(v_cam[2])


# ---------------------------------------------------------------------------
# Differentiable projection
# ---------------------------------------------------------------------------

def project_vertices(vertices, cam_pos, R, focal, W, H):
    """
    vertices : (N, 3, 3)
    Returns screen_xy (N, 3, 2) and valid (N,) bool mask.
    """
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
    """
    screen_xy : (N, 3, 2)
    Returns ns (N,3,2), ds (N,3), phi_s (N,)
    """
    v = screen_xy
    normals, offsets = [], []
    for i in range(3):
        a    = v[:, i,        :]
        b    = v[:, (i-1)%3,  :]
        edge = b - a
        n    = torch.stack([-edge[:, 1], edge[:, 0]], dim=-1)
        opp  = v[:, (i-2)%3,  :]
        dot  = (n * (opp - a)).sum(-1)
        sign = torch.where(dot > 0, -torch.ones_like(dot), torch.ones_like(dot))
        n    = n * sign.unsqueeze(-1)
        ni   = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        di   = -(ni * a).sum(-1)
        normals.append(ni)
        offsets.append(di)
    ns = torch.stack(normals, dim=1)
    ds = torch.stack(offsets, dim=1)

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

def render_differentiable(model, cam_pos, R, focal, W, H, sort_order,
                          return_max_weights=False):
    """
    Forward pass. Returns canvas (H,W,3) in [0,255].
    If return_max_weights=True also returns max_T_alpha (N,) used for pruning.
    """
    screen_xy, valid = project_vertices(model.vertices, cam_pos, R, focal, W, H)
    ns_all, ds_all, phi_s_all = compute_sdf_params(screen_xy)

    canvas        = torch.zeros(H, W, 3, device=model.vertices.device)
    transmittance = torch.ones(H, W,    device=model.vertices.device)
    colors    = model.colors
    opacities = model.opacities
    sigmas    = model.sigmas
    device    = model.vertices.device

    # For pruning: track max(T * o) per triangle
    N = model.num_triangles()
    max_weights = torch.zeros(N, device=device)

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
        phi_p   = (pts_exp * ns_all[idx].unsqueeze(0)).sum(-1) + ds_all[idx].unsqueeze(0)
        phi_p   = phi_p.max(dim=1).values

        ratio     = (phi_p / phi_s.clamp(max=-1e-7)).clamp(0.0, 1.0)
        influence = ratio.pow(sigmas[idx])
        alpha     = opacities[idx] * influence

        flat_idx = py.long() * W + px.long()
        T        = transmittance.reshape(-1)[flat_idx]

        # track max blending weight for pruning
        if return_max_weights:
            max_weights[idx] = (T * opacities[idx]).max().detach()

        color_contrib = colors[idx].unsqueeze(0) * (alpha * T).unsqueeze(-1) * 255.0

        canvas_flat     = canvas.reshape(-1, 3)
        canvas_flat_new = canvas_flat.scatter_add(
            0, flat_idx.unsqueeze(-1).expand(-1, 3), color_contrib)
        canvas = canvas_flat_new.reshape(H, W, 3)

        trans_flat          = transmittance.reshape(-1).clone()
        trans_flat[flat_idx] = T * (1.0 - alpha).detach()
        transmittance        = trans_flat.reshape(H, W)

    if return_max_weights:
        return canvas, max_weights
    return canvas


# ---------------------------------------------------------------------------
# Loss functions  (paper Eq. 3)
# ---------------------------------------------------------------------------

def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    D-SSIM loss = (1 - SSIM) / 2.
    pred, target: (H, W, 3) in [0, 255].
    Uses a simple average-pooling approximation for speed on CPU.
    """
    # Normalise to [0,1]
    p = pred   / 255.0
    t = target / 255.0

    # Move to (1, 3, H, W) for F.avg_pool2d
    p = p.permute(2, 0, 1).unsqueeze(0)
    t = t.permute(2, 0, 1).unsqueeze(0)

    import torch.nn.functional as F
    pad = window_size // 2

    mu_p  = F.avg_pool2d(p, window_size, stride=1, padding=pad)
    mu_t  = F.avg_pool2d(t, window_size, stride=1, padding=pad)
    mu_pp = F.avg_pool2d(p * p, window_size, stride=1, padding=pad)
    mu_tt = F.avg_pool2d(t * t, window_size, stride=1, padding=pad)
    mu_pt = F.avg_pool2d(p * t, window_size, stride=1, padding=pad)

    sigma_p  = mu_pp - mu_p * mu_p
    sigma_t  = mu_tt - mu_t * mu_t
    sigma_pt = mu_pt - mu_p * mu_t

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2))
    return (1.0 - ssim_map).mean() / 2.0


def compute_loss(canvas: torch.Tensor,
                 target: torch.Tensor,
                 model:  TriangleSplatModel,
                 lam:    float = 0.2,
                 beta1:  float = 0.01,
                 beta4:  float = 0.001) -> tuple[torch.Tensor, dict]:
    """
    Full loss from paper Eq. 3 (simplified: L1 + D-SSIM + Lo + Ls).
    We omit Ld (distortion) and Ln (normal) as they require depth maps.

    lam   : weight for D-SSIM vs L1 (paper uses 0.2)
    beta1 : weight for opacity loss Lo
    beta4 : weight for size loss Ls
    """
    # --- Photometric ---
    p   = canvas / 255.0
    t   = target / 255.0
    l1  = (p - t).abs().mean()
    dssim = _ssim_loss(canvas, target)
    photo = (1.0 - lam) * l1 + lam * dssim

    # --- Opacity loss Lo  (paper cites 3DGS-MCMC [22]) ---
    # Encourages opacities to be near 0 or 1, not stuck at 0.5.
    # Lo = -sum( o * log(o) + (1-o) * log(1-o) ) / N  (binary entropy)
    o  = model.opacities
    lo = -(o * torch.log(o + 1e-7) + (1 - o) * torch.log(1 - o + 1e-7)).mean()

    # --- Size loss Ls  (paper: Ls = -0.5 * ||(v1-v0) x (v2-v0)||) ---
    # Negative because we MAXIMISE triangle size (add to loss with minus sign
    # already baked in by the paper's definition).
    v  = model.vertices                         # (N, 3, 3)
    e1 = v[:, 1] - v[:, 0]                     # (N, 3)
    e2 = v[:, 2] - v[:, 0]
    cross = torch.cross(e1, e2, dim=-1)         # (N, 3)
    ls = -0.5 * cross.norm(dim=-1).mean()       # negative → minimising loss maximises area

    total = photo + beta1 * lo + beta4 * ls

    return total, {"photo": photo.item(), "l1": l1.item(),
                   "dssim": dssim.item(), "lo": lo.item(), "ls": ls.item()}


# ---------------------------------------------------------------------------
# Densification: midpoint subdivision  (paper Sec. 3.2)
# ---------------------------------------------------------------------------

def densify_model(model: TriangleSplatModel,
                  optimizer: optim.Optimizer,
                  max_triangles: int = 500,
                  tau_small:     float = 1e-3) -> TriangleSplatModel:
    """
    Midpoint subdivision densification (paper Sec. 3.2).

    At each call we pick which triangles to split using Bernoulli sampling
    proportional alternately to opacity and 1/sigma (paper: MCMC-style).
    Each selected triangle is split into 4 by connecting edge midpoints.
    If a triangle is smaller than tau_small it is cloned with noise instead.

    Returns a NEW model with more triangles (optimizer is rebuilt).
    """
    N = model.num_triangles()
    if N >= max_triangles:
        return model

    n_to_add = min(max(N // 4, 1), max_triangles - N)

    with torch.no_grad():
        opacs  = model.opacities.detach()   # (N,)
        sigs   = model.sigmas.detach()      # (N,)

        # Alternate between opacity-based and 1/sigma-based sampling
        prob_o = opacs / opacs.sum().clamp(min=1e-7)
        prob_s = (1.0 / sigs.clamp(min=1e-4))
        prob_s = prob_s / prob_s.sum().clamp(min=1e-7)
        prob   = 0.5 * prob_o + 0.5 * prob_s
        prob   = prob / prob.sum()

        chosen = torch.multinomial(prob, num_samples=min(n_to_add, N),
                                   replacement=False)

        verts_old  = model.vertices.detach()       # (N, 3, 3)
        colors_old = model.color_logit.detach()    # (N, 3) — logit space
        opacs_old  = model.opacity_logit.detach()  # (N,)
        sigs_old   = model.log_sigma.detach()      # (N,)

        new_verts  = []
        new_colors = []
        new_opacs  = []
        new_sigs   = []

        for idx in chosen:
            v = verts_old[idx]      # (3, 3) — three vertices
            v0, v1, v2 = v[0], v[1], v[2]

            # Edge midpoints
            m01 = (v0 + v1) / 2
            m12 = (v1 + v2) / 2
            m02 = (v0 + v2) / 2

            # Compute area to check if "small"
            e1    = v1 - v0
            e2    = v2 - v0
            area  = 0.5 * torch.cross(e1, e2, dim=-1).norm()

            if area < tau_small:
                # Clone with small noise in the triangle's plane
                normal = torch.cross(e1, e2, dim=-1)
                normal = normal / normal.norm().clamp(min=1e-7)
                # Two in-plane tangent vectors
                t1 = e1 / e1.norm().clamp(min=1e-7)
                t2 = torch.cross(normal, t1, dim=-1)
                noise_scale = area.sqrt() * 0.1
                noise = (torch.randn(3, device=v.device).unsqueeze(-1) *
                         torch.stack([t1, t2, t1], dim=0)) * noise_scale
                new_v = v + noise
                new_verts.append(new_v.unsqueeze(0))
            else:
                # Split into 4 sub-triangles
                sub = torch.stack([
                    torch.stack([v0,  m01, m02]),  # (3, 3)
                    torch.stack([m01, v1,  m12]),
                    torch.stack([m02, m12, v2 ]),
                    torch.stack([m01, m12, m02]),
                ], dim=0)  # (4, 3, 3)
                new_verts.append(sub)

            n_new = new_verts[-1].shape[0]
            # Inherit colour, opacity, sigma from parent
            new_colors.append(colors_old[idx].unsqueeze(0).expand(n_new, -1))
            new_opacs.append(opacs_old[idx].unsqueeze(0).expand(n_new))
            new_sigs.append(sigs_old[idx].unsqueeze(0).expand(n_new))

        if not new_verts:
            return model

        all_new_verts  = torch.cat(new_verts,  dim=0)  # (M, 3, 3)
        all_new_colors = torch.cat(new_colors, dim=0)  # (M, 3)
        all_new_opacs  = torch.cat(new_opacs,  dim=0)  # (M,)
        all_new_sigs   = torch.cat(new_sigs,   dim=0)  # (M,)

        # Concatenate with existing parameters
        cat_verts  = torch.cat([verts_old,  all_new_verts],  dim=0)
        cat_colors = torch.cat([colors_old, all_new_colors], dim=0)
        cat_opacs  = torch.cat([opacs_old,  all_new_opacs],  dim=0)
        cat_sigs   = torch.cat([sigs_old,   all_new_sigs],   dim=0)

    # Build a new model — we pass logit/log values directly
    new_model = TriangleSplatModel(
        cat_verts,
        torch.sigmoid(cat_colors),      # convert back from logit for the constructor
        torch.sigmoid(cat_opacs),
        torch.exp(cat_sigs),
    ).to(model.vertices.device)

    # Override the stored logits/logs so we don't lose precision
    with torch.no_grad():
        new_model.color_logit.copy_(cat_colors)
        new_model.opacity_logit.copy_(cat_opacs)
        new_model.log_sigma.copy_(cat_sigs)

    print(f"[densify] {N} -> {new_model.num_triangles()} triangles "
          f"(+{new_model.num_triangles() - N})")
    return new_model


# ---------------------------------------------------------------------------
# Pruning  (paper Sec. 3.2)
# ---------------------------------------------------------------------------

def prune_model(model:       TriangleSplatModel,
                max_weights: torch.Tensor,
                tau_prune:   float = 0.005) -> TriangleSplatModel:
    """
    Remove triangles whose maximum blending weight T*o across all views
    seen so far is below tau_prune.

    max_weights : (N,) tensor, updated each training step via return_max_weights=True.
    """
    with torch.no_grad():
        keep = max_weights >= tau_prune
        n_keep = keep.sum().item()
        n_pruned = model.num_triangles() - n_keep
        if n_pruned == 0:
            return model

        new_verts  = model.vertices.detach()[keep]
        new_colors = torch.sigmoid(model.color_logit.detach()[keep])
        new_opacs  = torch.sigmoid(model.opacity_logit.detach()[keep])
        new_sigs   = torch.exp(model.log_sigma.detach()[keep])

    new_model = TriangleSplatModel(new_verts, new_colors, new_opacs, new_sigs
                                   ).to(model.vertices.device)
    with torch.no_grad():
        new_model.color_logit.copy_(model.color_logit.detach()[keep])
        new_model.opacity_logit.copy_(model.opacity_logit.detach()[keep])
        new_model.log_sigma.copy_(model.log_sigma.detach()[keep])

    print(f"[prune]   {model.num_triangles()} -> {n_keep} triangles "
          f"(-{n_pruned})")
    return new_model


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def training_step(model, optimizer, target, cam_pos, R, focal, W, H, sort_order,
                  accumulated_max_weights):
    """
    One gradient step using the full paper loss.
    accumulated_max_weights : (N,) tensor updated in-place for pruning.
    Returns (loss_value, grad_norms_dict, loss_components_dict).
    """
    optimizer.zero_grad()

    zero = {"vertices": 0.0, "sigma": 0.0, "opacity": 0.0, "color": 0.0}
    if len(sort_order) == 0:
        return 0.0, zero, {}

    canvas, max_w = render_differentiable(
        model, cam_pos, R, focal, W, H, sort_order, return_max_weights=True)

    # Accumulate max weights across views (running max)
    N_model = model.num_triangles()
    N_acc   = accumulated_max_weights.shape[0]
    if N_model == N_acc:
        # update only the triangles that were in sort_order
        accumulated_max_weights[:] = torch.max(accumulated_max_weights, max_w)

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
    return float(loss.item()), grad_norms, components


# ---------------------------------------------------------------------------
# Numpy preview render  (unchanged, fast)
# ---------------------------------------------------------------------------

def numpy_render_to_array(triangles, sorted_indices, W, H):
    color_acc = np.zeros((H, W, 3), dtype=np.float32)
    transmit  = np.ones((H, W),     dtype=np.float32)
    cam   = Globals.cameraPosition
    R     = Globals.R
    focal = Globals.focalLength

    for idx in sorted_indices:
        tri = triangles[idx]
        pixels, ok = [], True
        for v in tri.vertices:
            p_cam = v - cam
            xc, yc, zc = p_cam @ R
            if zc <= 1e-4: ok = False; break
            sx = int(focal * xc / zc + W / 2)
            sy = int(focal * yc / zc + H / 2)
            pixels.append(Pixel(sx, sy, 1.0 / zc))
        if not ok: continue

        for p in pixels:
            p.x = int(np.clip(p.x, 0, W-1))
            p.y = int(np.clip(p.y, 0, H-1))

        verts2d = [np.array([p.x, p.y], dtype=np.float32) for p in pixels]
        Ls, bad = [], False
        for i in range(3):
            edge   = verts2d[i-1] - verts2d[i]
            normal = np.array([-edge[1], edge[0]], dtype=np.float32)
            dot    = normal @ (verts2d[i-2] - verts2d[i])
            if dot > 0: normal = -normal
            nrm = np.linalg.norm(normal)
            if nrm < 1e-7: bad = True; break
            ni = normal / nrm
            Ls.append((ni, -(ni @ verts2d[i-1])))
        if bad: continue

        ns = np.array([l[0] for l in Ls], dtype=np.float32)
        ds = np.array([l[1] for l in Ls], dtype=np.float32)
        a = np.linalg.norm(verts2d[0]-verts2d[1])
        b = np.linalg.norm(verts2d[0]-verts2d[2])
        c = np.linalg.norm(verts2d[2]-verts2d[1])
        perim = a+b+c
        if perim < 1e-7: continue
        inc   = (a*verts2d[2]+b*verts2d[1]+c*verts2d[0])/perim
        phi_s = np.max(inc @ ns.T + ds)
        if phi_s >= 0: continue

        xs_min = max(0,   min(p.x for p in pixels))
        xs_max = min(W-1, max(p.x for p in pixels))
        ys_min = max(0,   min(p.y for p in pixels))
        ys_max = min(H-1, max(p.y for p in pixels))
        if xs_min > xs_max or ys_min > ys_max: continue

        color_rgb = (tri.color * 255.0).astype(np.float32)
        for y in range(ys_min, ys_max+1):
            xs   = np.arange(xs_min, xs_max+1)
            pts  = np.stack([xs, np.full_like(xs, y)], axis=1)
            phi_p = np.max(pts @ ns.T + ds, axis=1)
            infl  = np.power(np.clip(phi_p/phi_s, 0.0, 1.0), tri.sigma)
            alpha = tri.opacity * infl
            T     = transmit[y, xs_min:xs_max+1]
            color_acc[y, xs_min:xs_max+1] += color_rgb*(alpha*T)[:,np.newaxis]
            transmit[y, xs_min:xs_max+1]  *= (1.0-alpha)

    return np.clip(color_acc, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    W, H = Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT

    # -- data --
    Globals.pointcloudData = load_points3D("south-building/sparse/points3D.txt")

    train_views, test_views, colmap_camera = load_dataset(
        "south-building/sparse", "south-building/images", W, H)
    Globals.focalLength = colmap_focal_to_renderer(colmap_camera, W, H)

    # Limit points for faster startup. Set to None to use all 61k.
    MAX_POINTS = 15000
    points = Globals.pointcloudData
    if MAX_POINTS is not None:
        points = points[:MAX_POINTS]

    all_triangles = initialize_triangles(points)
    chunk_indices = get_chunk_indices(points, Globals.chunk_k)

    device = Globals.device
    print(f"Device: {device}  |  focal: {Globals.focalLength:.1f} px")

    # -- filter training views to those that see the chunk --
    def chunk_visible_from(view):
        for i in chunk_indices:
            v_cam = (all_triangles[i].vertices.mean(axis=0) - view.cam_pos) @ view.R
            if v_cam[2] > 0.1:
                return True
        return False

    visible_train_views = [v for v in train_views if chunk_visible_from(v)]
    print(f"Views that see chunk: {len(visible_train_views)} / {len(train_views)}")
    if not visible_train_views:
        print("WARNING: no training views see the chunk. "
              "Try a different chunk or all points.")
        visible_train_views = train_views   # fallback

    # -- model and optimizer --
    model     = build_model(all_triangles, chunk_indices, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Accumulated max blending weights for pruning (one value per triangle)
    acc_max_weights = torch.zeros(model.num_triangles(), device=device)

    # Training schedule
    DENSIFY_EVERY  = 100   # densify every N training steps
    PRUNE_EVERY    = 200   # prune every N training steps
    MAX_TRIANGLES  = 1000  # cap on total triangles
    TAU_PRUNE      = 0.005 # pruning threshold (max T*o)

    # -- pygame --
    pygame.init()
    clock  = pygame.time.Clock()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Triangle Splatting")
    font = pygame.font.SysFont("Arial", 18)

    total_loss      = 0.0
    train_steps     = 0
    current_view_idx = 0
    train_mode      = False
    last_grad_norms = {"vertices": 0.0, "sigma": 0.0, "opacity": 0.0, "color": 0.0}
    last_components = {}

    running = True
    while running:

        # -- events --
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    Globals.show_debug = not Globals.show_debug
                if event.key == pygame.K_t:
                    train_mode = True
                    print("Training ON.")

        # -- camera movement --
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

        # -- sync & sort for preview --
        N = model.num_triangles()
        # chunk_indices may now be 0..N-1 after densification
        preview_indices = list(range(min(N, len(chunk_indices))))
        sync_model_to_triangles(model, all_triangles,
                                 chunk_indices[:len(preview_indices)])
        visible    = [i for i in chunk_indices[:len(preview_indices)]
                      if get_depth_numpy(all_triangles[i].vertices) > 0.1]
        sorted_idx = sorted(visible,
                            key=lambda i: get_depth_numpy(all_triangles[i].vertices))

        # -- numpy preview --
        frame_np = numpy_render_to_array(all_triangles, sorted_idx, W, H)
        pygame.surfarray.blit_array(screen, np.transpose(frame_np, (1, 0, 2)))

        # -- training step --
        if train_mode and visible_train_views:
            view      = visible_train_views[current_view_idx % len(visible_train_views)]
            target_np = load_target_image(view, "south-building/images", W, H)
            target    = torch.from_numpy(target_np).to(device)

            cam_t = torch.from_numpy(view.cam_pos).to(device)
            R_t   = torch.from_numpy(view.R).to(device)

            def depth_from_view(tri_idx):
                c = model.vertices[tri_idx].detach().cpu().numpy().mean(axis=0)
                return float(((c - view.cam_pos) @ view.R)[2])

            N_cur = model.num_triangles()
            train_sorted = sorted(
                [i for i in range(N_cur) if depth_from_view(i) > 0.1],
                key=depth_from_view)

            # Resize acc_max_weights if model grew
            if acc_max_weights.shape[0] != N_cur:
                new_acc = torch.zeros(N_cur, device=device)
                old_n   = min(acc_max_weights.shape[0], N_cur)
                new_acc[:old_n] = acc_max_weights[:old_n]
                acc_max_weights = new_acc

            loss_val, grad_norms, components = training_step(
                model, optimizer, target,
                cam_t, R_t, Globals.focalLength, W, H,
                train_sorted, acc_max_weights)

            last_grad_norms  = grad_norms
            last_components  = components
            total_loss      += loss_val
            train_steps     += 1
            current_view_idx += 1

            if train_steps % 10 == 0:
                avg = total_loss / train_steps
                c   = last_components
                print(
                    f"Step {train_steps:4d} | {view.filename} | "
                    f"avg {avg:.0f} | "
                    f"l1={c.get('l1',0):.4f} ssim={c.get('dssim',0):.4f} "
                    f"lo={c.get('lo',0):.3f} ls={c.get('ls',0):.4f} | "
                    f"N={model.num_triangles()} | "
                    f"grad[v]={grad_norms['vertices']:.2e}"
                )

            # -- densification --
            if train_steps % DENSIFY_EVERY == 0:
                model = densify_model(model, optimizer,
                                      max_triangles=MAX_TRIANGLES)
                # Rebuild optimizer for new parameter set
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

            # -- pruning --
            if train_steps % PRUNE_EVERY == 0 and model.num_triangles() > 10:
                if acc_max_weights.shape[0] == model.num_triangles():
                    model = prune_model(model, acc_max_weights,
                                        tau_prune=TAU_PRUNE)
                    optimizer = optim.Adam(model.parameters(), lr=1e-4)
                    acc_max_weights = torch.zeros(model.num_triangles(),
                                                  device=device)

        # -- debug overlay --
        if Globals.show_debug:
            view_name = (visible_train_views[current_view_idx
                         % len(visible_train_views)].filename
                         if train_mode and visible_train_views else "—")
            lines = [
                f"Triangles: {model.num_triangles()}  chunk: {len(chunk_indices)}",
                f"Focal: {Globals.focalLength:.1f} px",
                f"Train: {'ON' if train_mode else 'OFF (press T)'}  "
                f"step {train_steps}",
                f"Avg loss: {total_loss/max(1,train_steps):.0f}",
                f"View: {view_name}",
                f"l1={last_components.get('l1',0):.0f}  "
                f"ssim={last_components.get('dssim',0):.3f}  "
                f"lo={last_components.get('lo',0):.3f}",
                f"Grad |v|: {last_grad_norms['vertices']:.2e}  "
                f"|c|: {last_grad_norms['color']:.2e}",
                "F1 debug | WASD move | arrows rotate | T train",
            ]
            for i, line in enumerate(lines):
                surf = font.render(line, True, (255, 255, 0))
                screen.blit(surf, (10, 10 + i * 22))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()