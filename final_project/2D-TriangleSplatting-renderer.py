import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F



# Utils


def make_logit(x, eps=1e-4):
    x = x.clamp(eps, 1.0 - eps) # x in (0,1)
    return torch.log(x / (1.0 - x))

# used for initializing sigma later
def inv_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float32).clamp_min(1e-8)
    return torch.log(torch.expm1(y).clamp_min(1e-8))


def save_image_tensor(img_chw, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img_chw.detach().clamp(0.0, 1.0).cpu()
    arr = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_target_image(image_path, max_size):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    scale = min(1.0, float(max_size) / float(max(w, h)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    target = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return target

# we need the coordinates of all the pixels
def make_xy_grid(h, w, device):
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)  # [H,W,2]


def triangle_areas(vertices):
    v0 = vertices[:, 0]
    v1 = vertices[:, 1]
    v2 = vertices[:, 2]
    # this gets the are of all the triangles at the same time
    cross = (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )
    return 0.5 * cross.abs()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Logging
# ============================================================

class TrainingLogger:
    def __init__(self, outdir):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.text_path = self.outdir / "training_log.txt"
        self.csv_path = self.outdir / "training_log.csv"

        # Reset files at the start of a run.
        with open(self.text_path, "w", encoding="utf-8") as f:
            f.write("2D triangle renderer training log\n")
            f.write("================================\n")

        self.csv_fields = [
            "elapsed_sec",
            "step",
            "loss",
            "l1",
            "dssim",
            "ssim",
            "ssim_weight",
            "opacity_reg",
            "size_reg",
            "sigma_reg",
            "sigma_mean",
            "sigma_min",
            "sigma_max",
            "depth_min",
            "depth_mean",
            "depth_max",
            "num_triangles",
            "step_time_sec",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

    def elapsed(self):
        return time.time() - self.start_time

    def write(self, message):
        elapsed = self.elapsed()
        line = f"[{elapsed:9.2f}s | {elapsed / 60.0:7.2f}m] {message}"
        print(line, flush=True)
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def write_args(self, args):
        args_path = self.outdir / "training_args.txt"
        with open(args_path, "w", encoding="utf-8") as f:
            for key, value in sorted(vars(args).items()):
                f.write(f"{key}: {value}\n")
        self.write(f"Saved arguments to {args_path}")

    def write_metrics(self, **kwargs):
        row = {field: kwargs.get(field, "") for field in self.csv_fields}
        row["elapsed_sec"] = f"{self.elapsed():.6f}"
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow(row)



# Initialize the run settings and hyerparameters
# ============================================================

def initialize_triangles_from_grid(
    target_cpu,
    grid_step_px=16,
    tri_scale=0.95,
    sigma_init=1.16,
    opacity_init=0.88,
    depth_jitter=1.0,
):

# for very grid center we initialize one triangle
    _, _, h, w = target_cpu.shape
    target_img = target_cpu[0]  # [3,H,W]

    # one traingle for every pixel is too much, gridstep ensure they are spaced out
    xs = list(range(grid_step_px // 2, w, grid_step_px))
    ys = list(range(grid_step_px // 2, h, grid_step_px))

    centers = []
    for y in ys:
        for x in xs:
            centers.append((x, y))

    if len(centers) == 0:
        centers = [(w // 2, h // 2)]

    # Radius in normalized coordinates.
    radius = tri_scale * grid_step_px * 2.0 / max(w - 1, h - 1)

    vertices = []
    colors = []
    opacity_logits = []
    depths = []

    for x, y in centers:
        cx = (x / max(w - 1, 1)) * 2.0 - 1.0
        cy = (y / max(h - 1, 1)) * 2.0 - 1.0

        base_angle = random.random() * 2.0 * math.pi

        tri = []
        for k in range(3):
            a = base_angle + k * 2.0 * math.pi / 3.0
            tri.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])

        vertices.append(tri)
        colors.append(target_img[:, y, x])
        opacity_logits.append(make_logit(torch.tensor(opacity_init, dtype=torch.float32)))

        # Larger depth = closer = rendered later = appears in front.
        # we set a random depth to allow for more alphablending
        z = random.random()
        if depth_jitter != 1.0:
            z = z * depth_jitter
        depths.append(z)

    vertices = torch.tensor(vertices, dtype=torch.float32)
    colors = torch.stack(colors, dim=0).float()
    color_logits = make_logit(colors)
    opacity_logits = torch.stack(opacity_logits, dim=0).float()

    sigma_base = max(float(sigma_init) - 1.0, 1e-4)
    sigma_logits = torch.full(
        (vertices.shape[0],),
        float(inv_softplus(sigma_base)),
        dtype=torch.float32,
    )

    depths = torch.tensor(depths, dtype=torch.float32)

    return vertices, color_logits, opacity_logits, sigma_logits, depths



# Triangle window function 
# ============================================================

def signed_edge_distance_grid(p, a, b):

    edge = b - a
    ex = edge[:, 0].view(-1, 1, 1)
    ey = edge[:, 1].view(-1, 1, 1)

    apx = p[None, :, :, 0] - a[:, 0].view(-1, 1, 1)
    apy = p[None, :, :, 1] - a[:, 1].view(-1, 1, 1)

    cross = ex * apy - ey * apx
    denom = torch.sqrt(edge[:, 0] ** 2 + edge[:, 1] ** 2).view(-1, 1, 1).clamp_min(1e-8)

    return cross / denom


def signed_edge_distance_points(p, a, b):
    """
    p: [B,2]
    a: [B,2]
    b: [B,2]
    returns: [B]
    """
    edge = b - a
    ap = p - a
    cross = edge[:, 0] * ap[:, 1] - edge[:, 1] * ap[:, 0]
    denom = torch.linalg.norm(edge, dim=1).clamp_min(1e-8)
    return cross / denom

def triangle_window(vertices, xy_grid, sigma):
 
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]

    # ------------------------------------------------------------
    # 2. Determine triangle orientation
    # ------------------------------------------------------------
    area2 = (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )

    orientation_sign = torch.where(area2 >= 0.0, 1.0, -1.0)
    orientation_sign_grid = orientation_sign.view(-1, 1, 1)

    # Compute edge functions L_i(p)
    L0 = -signed_edge_distance_grid(xy_grid, v0, v1) * orientation_sign_grid
    L1 = -signed_edge_distance_grid(xy_grid, v1, v2) * orientation_sign_grid
    L2 = -signed_edge_distance_grid(xy_grid, v2, v0) * orientation_sign_grid

    # Signed distance field (SGD):
    # phi(p) = max_i L_i(p)
    # This gives:
    #     phi(p) < 0 inside
    #     phi(p) = 0 on boundary
    #     phi(p) > 0 outside
    phi_p = torch.maximum(torch.maximum(L0, L1), L2)

    inside = phi_p < 0.0

    # ------------------------------------------------------------
    # 5. Compute triangle incenter s
    # ------------------------------------------------------------
    side_01 = torch.linalg.norm(v1 - v0, dim=1)
    side_12 = torch.linalg.norm(v2 - v1, dim=1)
    side_20 = torch.linalg.norm(v0 - v2, dim=1)

    perimeter = (side_01 + side_12 + side_20).clamp_min(1e-8)

    incenter = (
        side_12[:, None] * v0
        + side_20[:, None] * v1
        + side_01[:, None] * v2
    ) / perimeter[:, None]

    # ------------------------------------------------------------
    # 6. Compute phi(s), using the same paper sign convention
    # ------------------------------------------------------------
    L0_s = -signed_edge_distance_points(incenter, v0, v1) * orientation_sign
    L1_s = -signed_edge_distance_points(incenter, v1, v2) * orientation_sign
    L2_s = -signed_edge_distance_points(incenter, v2, v0) * orientation_sign

    phi_s = torch.maximum(torch.maximum(L0_s, L1_s), L2_s)

    # phi_s should be negative inside the triangle.
    # Clamp its magnitude to avoid division by zero for degenerate triangles.
    phi_s = phi_s.clamp_max(-1e-8).view(-1, 1, 1)

    # ÁCtual Window function:
    #I(p) = ReLU(phi(p) / phi(s))^sigma
    # Since both phi(p) and phi(s) are negative inside the triangle,
    # their ratio is positive inside.

    # phi(p) / phi(s) = 1

    # phi(p) = 0, so I(p) = 0

    # Outside:
    # phi(p) > 0 and phi(s) < 0, so the ratio is negative
    # and ReLU sets it to 0.
    w = torch.relu(phi_p / phi_s).clamp(0.0, 1.0)

    # sigma
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), dtype=vertices.dtype, device=vertices.device)

    if sigma.ndim == 0:
        sigma = sigma.view(1, 1, 1)
    elif sigma.ndim == 1:
        sigma = sigma.view(-1, 1, 1)

    sigma = sigma.clamp_min(1.0)

    # apply smoothness through sigma
    w = w.pow(sigma)
    w = torch.where(inside, w, torch.zeros_like(w))

    return w

# def triangle_window(vertices, xy_grid, sigma):
#     """
#     Paper-inspired triangle window function.
#     vertices: [B,3,2]
#     xy_grid: [H,W,2]
#     sigma: [B] or scalar
#     returns: [B,H,W]
#     """
#     v0 = vertices[:, 0, :]
#     v1 = vertices[:, 1, :]
#     v2 = vertices[:, 2, :]

#     area2 = (
#         (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
#         - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
#     )

#     sign_scalar = torch.where(area2 >= 0.0, 1.0, -1.0)
#     sign_grid = sign_scalar.view(-1, 1, 1)

#     d0 = signed_edge_distance_grid(xy_grid, v0, v1) * sign_grid
#     d1 = signed_edge_distance_grid(xy_grid, v1, v2) * sign_grid
#     d2 = signed_edge_distance_grid(xy_grid, v2, v0) * sign_grid

#     nearest_edge_dist = torch.minimum(torch.minimum(d0, d1), d2)
#     inside = nearest_edge_dist > 0.0

#     edge_window = nearest_edge_dist.clamp_min(0.0)

#     side_01 = torch.linalg.norm(v1 - v0, dim=1)
#     side_12 = torch.linalg.norm(v2 - v1, dim=1)
#     side_20 = torch.linalg.norm(v0 - v2, dim=1)

#     perimeter = (side_01 + side_12 + side_20).clamp_min(1e-8)

#     incenter = (
#         side_12[:, None] * v0
#         + side_20[:, None] * v1
#         + side_01[:, None] * v2
#     ) / perimeter[:, None]

#     s0 = signed_edge_distance_points(incenter, v0, v1) * sign_scalar
#     s1 = signed_edge_distance_points(incenter, v1, v2) * sign_scalar
#     s2 = signed_edge_distance_points(incenter, v2, v0) * sign_scalar

#     incenter_window = torch.minimum(torch.minimum(s0, s1), s2).clamp_min(1e-8)
#     incenter_window = incenter_window.view(-1, 1, 1)

#     w = (edge_window / incenter_window).clamp(0.0, 1.0)

#     if not torch.is_tensor(sigma):
#         sigma = torch.tensor(float(sigma), dtype=vertices.dtype, device=vertices.device)

#     if sigma.ndim == 0:
#         sigma = sigma.view(1, 1, 1)
#     elif sigma.ndim == 1:
#         sigma = sigma.view(-1, 1, 1)

#     sigma = sigma.clamp_min(1.0)
#     w = w.pow(sigma)
#     w = torch.where(inside, w, torch.zeros_like(w))

#     return w


# Rendering Tme ==============================


def render_triangles_tile(
    vertices,
    color_logits,
    opacity_logits,
    sigma_logits,
    depths,
    xy_tile,
    tri_chunk_size=8,
    background=1.0,
):
    # Here we render one tile at a time, we also make sure the triangles are rendered in order of depth

    device = vertices.device
    dtype = vertices.dtype
    th, tw, _ = xy_tile.shape
    n = vertices.shape[0]

    image = torch.full((th, tw, 3), float(background), device=device, dtype=dtype)

    # small depth = far away
    order = torch.argsort(depths, descending=False)

    vertices = vertices[order]
    color_logits = color_logits[order]
    opacity_logits = opacity_logits[order]
    sigma_logits = sigma_logits[order]

    for start in range(0, n, tri_chunk_size):
        end = min(start + tri_chunk_size, n)

        v = vertices[start:end]
        c = color_logits[start:end]
        o = opacity_logits[start:end]
        s_log = sigma_logits[start:end]

        colors = torch.sigmoid(c).view(-1, 1, 1, 3)
        opacities = torch.sigmoid(o).view(-1, 1, 1)
        sigmas = F.softplus(s_log) + 1.0

        masks = triangle_window(v, xy_tile, sigmas)
        alphas = (opacities * masks).clamp(0.0, 1.0)

        batch_size = end - start
        for i in range(batch_size):
            a = alphas[i].unsqueeze(-1)
            src = colors[i]
            image = image * (1.0 - a) + src * a

    return image.clamp(0.0, 1.0)


# Model, stores all train params
class TriangleImageModel(nn.Module):
    def __init__(
        self,
        vertices,
        color_logits,
        opacity_logits,
        sigma_logits,
        depths,
        tri_chunk_size=8,
        background=1.0,
    ):
        #initilizae neurla netweork and store params
        super().__init__()
        self.vertices = nn.Parameter(vertices)
        self.color_logits = nn.Parameter(color_logits)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.sigma_logits = nn.Parameter(sigma_logits)

        self.register_buffer("depths", depths)

        n = vertices.shape[0]
        self.register_buffer("grad_accum", torch.zeros(n, device=vertices.device))
        self.register_buffer("grad_count", torch.zeros(n, device=vertices.device))

        self.tri_chunk_size = tri_chunk_size
        self.background = background

    def forward_tile(self, xy_tile):
        return render_triangles_tile(
            vertices=self.vertices,
            color_logits=self.color_logits,
            opacity_logits=self.opacity_logits,
            sigma_logits=self.sigma_logits,
            depths=self.depths,
            xy_tile=xy_tile,
            tri_chunk_size=self.tri_chunk_size,
            background=self.background,
        )

    def sigma_values(self):
        return F.softplus(self.sigma_logits) + 1.0


# Full render no grad, just for saving images

def render_full_no_grad(model, xy_grid, tile_height):
    was_training = model.training
    model.eval()
    chunks = []

    with torch.no_grad():
        h = xy_grid.shape[0]
        for y0 in range(0, h, tile_height):
            y1 = min(y0 + tile_height, h)
            tile = model.forward_tile(xy_grid[y0:y1])
            chunks.append(tile.detach().cpu())

    if was_training:
        model.train()

    full = torch.cat(chunks, dim=0)
    return full.permute(2, 0, 1)



# Loss

# gets SSIM between current render and target y
def ssim_index(x, y, window=11):

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, window, stride=1, padding=window // 2)
    mu_y = F.avg_pool2d(y, window, stride=1, padding=window // 2)

    sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=window // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=window // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=window // 2) - mu_x * mu_y

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)

    return (numerator / denominator.clamp_min(1e-8)).mean()


def get_ssim_weight(step, start_step, final_weight, ramp_steps):
    if final_weight <= 0.0:
        return 0.0
    if step < start_step:
        return 0.0
    if ramp_steps <= 0:
        return final_weight
    t = (step - start_step) / float(ramp_steps)
    t = min(1.0, max(0.0, t))
    return final_weight * t


# Optimizer adam

def make_optimizer(model, args):
    return torch.optim.Adam(
        [
            {"params": [model.vertices], "lr": args.lr_vertices},
            {"params": [model.color_logits], "lr": args.lr_colors},
            {"params": [model.opacity_logits], "lr": args.lr_opacity},
            {"params": [model.sigma_logits], "lr": args.lr_sigma},
        ]
    )


def reset_topology(model):
    n = model.vertices.shape[0]
    device = model.vertices.device
    model.grad_accum = torch.zeros(n, device=device)
    model.grad_count = torch.zeros(n, device=device)


def accumulate_vertex_grads(model):
    if model.vertices.grad is None:
        return
    with torch.no_grad():
        # [N,3,2] -> [N]
        g = model.vertices.grad.detach().norm(dim=2).mean(dim=1)
        if model.grad_accum.numel() != g.numel(): # reset if number of triangles have changed
            reset_topology(model)
        model.grad_accum += g
        model.grad_count += 1.0


def split_triangles(vertices):
    """
    vertices: [K,3,2]
    returns: [K*4,3,2]
    """
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]

    m01 = 0.5 * (v0 + v1)
    m12 = 0.5 * (v1 + v2)
    m20 = 0.5 * (v2 + v0)

    c0 = torch.stack([v0, m01, m20], dim=1)
    c1 = torch.stack([m01, v1, m12], dim=1)
    c2 = torch.stack([m20, m12, v2], dim=1)
    c3 = torch.stack([m01, m12, m20], dim=1)

    return torch.cat([c0, c1, c2, c3], dim=0)


def replace_model_tensors(model, vertices, color_logits, opacity_logits, sigma_logits, depths):
    model.vertices = nn.Parameter(vertices)
    model.color_logits = nn.Parameter(color_logits)
    model.opacity_logits = nn.Parameter(opacity_logits)
    model.sigma_logits = nn.Parameter(sigma_logits)
    model.depths = depths.detach()
    reset_topology(model)


def densify_and_prune(model, args, step, log_fn=None):
    
    #Prune low-contribution triangles and split high-gradient triangles.
    # Returns True if topology changed, so the optimizer can be rebuilt.
    device = model.vertices.device

    with torch.no_grad():
        vertices = model.vertices.detach()
        color_logits = model.color_logits.detach()
        opacity_logits = model.opacity_logits.detach()
        sigma_logits = model.sigma_logits.detach()
        depths = model.depths.detach()

        n_before = vertices.shape[0]
        opacities = torch.sigmoid(opacity_logits)
        areas = triangle_areas(vertices)

        
        # Pruning ===
        
        pruned = 0
        if args.enable_prune and step % args.prune_every == 0:
            prune_mask = (opacities < args.prune_opacity) | (areas < args.prune_area)

            if args.min_triangles > 0:
                keep_count = int((~prune_mask).sum().item())
                if keep_count < args.min_triangles:
                    prune_mask[:] = False

            keep_mask = ~prune_mask
            vertices = vertices[keep_mask]
            color_logits = color_logits[keep_mask]
            opacity_logits = opacity_logits[keep_mask]
            sigma_logits = sigma_logits[keep_mask]
            depths = depths[keep_mask]
            pruned = n_before - vertices.shape[0]

            if model.grad_accum.numel() == n_before:
                grad_accum = model.grad_accum.detach()[keep_mask]
                grad_count = model.grad_count.detach()[keep_mask]
            else:
                grad_accum = None
                grad_count = None
        else:
            grad_accum = model.grad_accum.detach() if model.grad_accum.numel() == n_before else None
            grad_count = model.grad_count.detach() if model.grad_count.numel() == n_before else None

        n_after_prune = vertices.shape[0]

        # Densification ====
        added = 0
        can_densify = (
            args.enable_densify
            and step >= args.densify_start
            and step <= args.densify_stop
            and step % args.densify_every == 0
            and n_after_prune < args.max_triangles
            and grad_accum is not None
            and grad_count is not None
            and grad_accum.numel() == n_after_prune
        )

        if can_densify:
            grad_score = grad_accum / grad_count.clamp_min(1.0)
            opacities_now = torch.sigmoid(opacity_logits)
            areas_now = triangle_areas(vertices)

            eligible = (
                (grad_score > args.densify_grad_threshold)
                & (areas_now > args.densify_min_area)
                & (opacities_now > args.densify_min_opacity)
            )
            eligible_idx = torch.nonzero(eligible, as_tuple=False).flatten()

            if eligible_idx.numel() > 0:
                scores = grad_score[eligible_idx]
                order = torch.argsort(scores, descending=True)
                eligible_idx = eligible_idx[order]

                remaining_capacity = args.max_triangles - n_after_prune
                # One split replaces 1 big traingle with 4 smaller ones
                max_splits_by_capacity = max(0, remaining_capacity // 3)
                max_splits_by_fraction = max(
                    1,
                    int(math.ceil(float(n_after_prune) * args.densify_top_frac)),
                )

                num_splits = min(
                    int(eligible_idx.numel()),
                    int(max_splits_by_capacity),
                    int(max_splits_by_fraction),
                    int(args.densify_max_splits),
                )

                if num_splits > 0:
                    split_idx = eligible_idx[:num_splits]
                    split_mask = torch.zeros(n_after_prune, dtype=torch.bool, device=device)
                    split_mask[split_idx] = True
                    nonsplit_mask = ~split_mask

                    parent_vertices = vertices[split_idx]
                    child_vertices = split_triangles(parent_vertices)

                    if args.densify_vertex_jitter > 0.0:
                        child_vertices = child_vertices + torch.randn_like(child_vertices) * args.densify_vertex_jitter
                    child_vertices = child_vertices.clamp(-1.25, 1.25)

                    parent_color = color_logits[split_idx]
                    parent_opacity = opacity_logits[split_idx]
                    parent_sigma = sigma_logits[split_idx]
                    parent_depth = depths[split_idx]

                    # new triangles inherit properties of original triangle
                    child_color = parent_color.repeat_interleave(4, dim=0)
                    child_opacity = parent_opacity.repeat_interleave(4, dim=0)
                    child_opacity = child_opacity + args.densify_opacity_logit_offset
                    child_sigma = parent_sigma.repeat_interleave(4, dim=0)

                    child_depth = parent_depth.repeat_interleave(4)
                    if args.densify_depth_jitter > 0.0:
                        child_depth = child_depth + torch.randn_like(child_depth) * args.densify_depth_jitter

                    vertices = torch.cat([vertices[nonsplit_mask], child_vertices], dim=0)
                    color_logits = torch.cat([color_logits[nonsplit_mask], child_color], dim=0)
                    opacity_logits = torch.cat([opacity_logits[nonsplit_mask], child_opacity], dim=0)
                    sigma_logits = torch.cat([sigma_logits[nonsplit_mask], child_sigma], dim=0)
                    depths = torch.cat([depths[nonsplit_mask], child_depth], dim=0)

                    added = int(child_vertices.shape[0] - num_splits)

        changed = pruned > 0 or added > 0

        if changed:
            replace_model_tensors(
                model=model,
                vertices=vertices.to(device),
                color_logits=color_logits.to(device),
                opacity_logits=opacity_logits.to(device),
                sigma_logits=sigma_logits.to(device),
                depths=depths.to(device),
            )
            message = (
                f"[topology] step={step} | pruned={pruned} | added={added} | "
                f"N {n_before} -> {model.vertices.shape[0]}"
            )
            if log_fn is not None:
                log_fn(message)
            else:
                print(message)

        return changed


# Render save schedule for GIFs


# this method was generated by AI, I didnt know how to make a GIF lol
def should_save_render_step(step, args):
    """
    GIF-friendly save schedule.

    Default behavior:
      - save every step from 1 to --save-all-until
      - save every --save-mid-every steps until --save-mid-until
      - save every --save-every steps after that

    With default args this saves:
      1..50, then 55/60/.../150, then 175/200/...
    """
    if step <= args.save_all_until:
        return True
    if step <= args.save_mid_until:
        return (step % args.save_mid_every) == 0
    return (step % args.save_every) == 0


# Training =====

def train(args):
    set_seed(args.seed)

    # use GPU if possible
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(outdir)
    logger.write_args(args)

    target = load_target_image(args.image, args.max_size).to(device)
    _, _, h, w = target.shape

    vertices, color_logits, opacity_logits, sigma_logits, depths = initialize_triangles_from_grid(
        target_cpu=target.cpu(),
        grid_step_px=args.grid,
        tri_scale=args.tri_scale,
        sigma_init=args.sigma_init,
        opacity_init=args.opacity_init,
        depth_jitter=args.depth_jitter,
    )

    model = TriangleImageModel(
        vertices=vertices.to(device),
        color_logits=color_logits.to(device),
        opacity_logits=opacity_logits.to(device),
        sigma_logits=sigma_logits.to(device),
        depths=depths.to(device),
        tri_chunk_size=args.tri_chunk_size,
        background=args.background,
    ).to(device)

    xy_grid = make_xy_grid(h, w, device)
    #adam
    optimizer = make_optimizer(model, args)

    # logging by AI, obv
    logger.write(f"Loaded target: {args.image}")
    logger.write(f"Image: {w}x{h} | triangles: {model.vertices.shape[0]} | device: {device}")
    logger.write(f"Output folder: {outdir}")
    logger.write(f"Triangle chunk size: {args.tri_chunk_size}")
    logger.write(f"Pixel tile height: {args.tile_height}")
    logger.write("Triangle window function: ON")
    logger.write("Random initial orientation: ON")
    logger.write("Pseudo depth sorting: ON, fixed back-to-front depth")
    logger.write(f"Densify: {'ON' if args.enable_densify else 'OFF'} | Prune: {'ON' if args.enable_prune else 'OFF'}")
    logger.write(f"D-SSIM schedule: start={args.ssim_start}, ramp={args.ssim_ramp}, final_weight={args.ssim_weight}")
    logger.write(
        f"Render save schedule: every step until {args.save_all_until}, "
        f"every {args.save_mid_every} steps until {args.save_mid_until}, "
        f"then every {args.save_every} steps"
    )

    save_image_tensor(target[0], outdir / "target.png")
    render0 = render_full_no_grad(model, xy_grid, args.tile_height)
    save_image_tensor(render0, outdir / "initial.png")
    save_image_tensor(render0, outdir / "render_step_00000.png")
    del render0

    best_loss = float("inf")
    total_pixels = h * w

    for step in range(1, args.steps + 1):
        step_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        ssim_weight = get_ssim_weight(
            step=step,
            start_step=args.ssim_start,
            final_weight=args.ssim_weight,
            ramp_steps=args.ssim_ramp,
        )

        loss_sum_print = 0.0
        l1_sum_print = 0.0
        dssim_sum_print = 0.0
        ssim_sum_print = 0.0

        # Pixel-tiled training keeps memory under control.
        # We accumulate gradients over all tiles, then take one optimizer step.
        for y0 in range(0, h, args.tile_height):
            y1 = min(y0 + args.tile_height, h)
            xy_tile = xy_grid[y0:y1]
            target_tile = target[:, :, y0:y1, :]

            render_tile = model.forward_tile(xy_tile)
            render_bchw = render_tile.permute(2, 0, 1).unsqueeze(0)

            l1 = F.l1_loss(render_bchw, target_tile)

            if ssim_weight > 0.0:
                ssim_val = ssim_index(render_bchw, target_tile, window=11)
                dssim = 0.5 * (1.0 - ssim_val)
            else:
                ssim_val = torch.tensor(0.0, device=device)
                dssim = torch.tensor(0.0, device=device)

            pixel_weight = ((y1 - y0) * w) / float(total_pixels)
            tile_loss = ((1.0 - ssim_weight) * l1 + ssim_weight * dssim) * pixel_weight
            tile_loss.backward()

            loss_sum_print += float(tile_loss.detach().cpu())
            l1_sum_print += float(l1.detach().cpu()) * pixel_weight
            dssim_sum_print += float(dssim.detach().cpu()) * pixel_weight
            ssim_sum_print += float(ssim_val.detach().cpu()) * pixel_weight

            del xy_tile, target_tile, render_tile, render_bchw, l1, dssim, ssim_val, tile_loss

        opacities = torch.sigmoid(model.opacity_logits)
        opacity_reg = (opacities * (1.0 - opacities)).mean()

        areas = triangle_areas(model.vertices)
        size_reg = areas.mean()

        sigmas = model.sigma_values()
        sigma_reg = ((sigmas - args.sigma_init) ** 2).mean()

        reg_loss = (
            args.opacity_reg * opacity_reg
            + args.size_reg * size_reg
            + args.sigma_reg * sigma_reg
        )
        reg_loss.backward()

        total_loss_print = loss_sum_print + float(reg_loss.detach().cpu())

        # Densification uses this accumulated per-triangle vertex gradient.
        accumulate_vertex_grads(model)

        optimizer.step()

        with torch.no_grad():
            model.vertices.clamp_(-1.25, 1.25)

        topology_changed = False
        if (
            (args.enable_prune and step % args.prune_every == 0)
            or (
                args.enable_densify
                and step >= args.densify_start
                and step <= args.densify_stop
                and step % args.densify_every == 0
            )
        ):
            topology_changed = densify_and_prune(model, args, step, log_fn=logger.write)

        if topology_changed:
            optimizer = make_optimizer(model, args)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if total_loss_print < best_loss:
            best_loss = total_loss_print
            best_render = render_full_no_grad(model, xy_grid, args.tile_height)
            save_image_tensor(best_render, outdir / "best.png")
            del best_render

        step_time = time.time() - step_start_time

        if step % args.print_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad():
                sigma_now = model.sigma_values()
                zmin = model.depths.min().item()
                zmax = model.depths.max().item()
                zmean = model.depths.mean().item()
                n_tris = model.vertices.shape[0]
                sigma_mean = sigma_now.mean().item()
                sigma_min = sigma_now.min().item()
                sigma_max = sigma_now.max().item()

            logger.write(
                f"Step {step:5d} | "
                f"loss={total_loss_print:.6f} | "
                f"l1={l1_sum_print:.6f} | "
                f"dssim={dssim_sum_print:.6f} | "
                f"ssim={ssim_sum_print:.6f} | "
                f"ssim_w={ssim_weight:.3f} | "
                f"op={opacity_reg.item():.6f} | "
                f"size={size_reg.item():.6f} | "
                f"sigma_reg={sigma_reg.item():.6f} | "
                f"sigma_mean={sigma_mean:.4f} | "
                f"sigma_min={sigma_min:.4f} | "
                f"sigma_max={sigma_max:.4f} | "
                f"depth[min/mean/max]=({zmin:.3f}/{zmean:.3f}/{zmax:.3f}) | "
                f"N={n_tris} | "
                f"step_time={step_time:.3f}s"
            )

            logger.write_metrics(
                step=step,
                loss=f"{total_loss_print:.8f}",
                l1=f"{l1_sum_print:.8f}",
                dssim=f"{dssim_sum_print:.8f}",
                ssim=f"{ssim_sum_print:.8f}",
                ssim_weight=f"{ssim_weight:.8f}",
                opacity_reg=f"{opacity_reg.item():.8f}",
                size_reg=f"{size_reg.item():.8f}",
                sigma_reg=f"{sigma_reg.item():.8f}",
                sigma_mean=f"{sigma_mean:.8f}",
                sigma_min=f"{sigma_min:.8f}",
                sigma_max=f"{sigma_max:.8f}",
                depth_min=f"{zmin:.8f}",
                depth_mean=f"{zmean:.8f}",
                depth_max=f"{zmax:.8f}",
                num_triangles=n_tris,
                step_time_sec=f"{step_time:.8f}",
            )

        if should_save_render_step(step, args) or step == args.steps:
            render_save = render_full_no_grad(model, xy_grid, args.tile_height)
            save_image_tensor(render_save, outdir / f"render_step_{step:05d}.png")
            del render_save

        if device.type == "cuda" and step % 50 == 0:
            torch.cuda.empty_cache()

    final = render_full_no_grad(model, xy_grid, args.tile_height)
    save_image_tensor(final, outdir / "final.png")

    torch.save(
        {
            "vertices": model.vertices.detach().cpu(),
            "color_logits": model.color_logits.detach().cpu(),
            "opacity_logits": model.opacity_logits.detach().cpu(),
            "sigma_logits": model.sigma_logits.detach().cpu(),
            "depths": model.depths.detach().cpu(),
            "height": h,
            "width": w,
            "args": vars(args),
            "best_loss": best_loss,
            "total_time_sec": logger.elapsed(),
        },
        outdir / "triangles_2d_checkpoint.pt",
    )

    logger.write("Done.")
    logger.write(f"Best loss: {best_loss:.6f}")
    logger.write(f"Training finished in {logger.elapsed():.2f}s ({logger.elapsed() / 60.0:.2f} min)")
    logger.write(f"Text log: {logger.text_path}")
    logger.write(f"CSV log: {logger.csv_path}")



# I used AI to get some nice arguments to change setting through command line


def parse_args():
    parser = argparse.ArgumentParser(description="2D triangle splatting with window function, depth, densification, pruning, and logging")

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="renders_2d_triangles3")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--tri-scale", type=float, default=0.95)

    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--tri-chunk-size", type=int, default=8)
    parser.add_argument("--tile-height", type=int, default=64)

    parser.add_argument("--lr-vertices", type=float, default=2e-3)
    parser.add_argument("--lr-colors", type=float, default=3e-2)
    parser.add_argument("--lr-opacity", type=float, default=1e-2)
    parser.add_argument("--lr-sigma", type=float, default=1e-3)

    parser.add_argument("--sigma-init", type=float, default=1.16)
    parser.add_argument("--sigma-reg", type=float, default=1e-4)
    parser.add_argument("--opacity-init", type=float, default=0.88)
    parser.add_argument("--depth-jitter", type=float, default=1.0)

    # Safer defaults after the recent runs: start D-SSIM later, ramp slowly, keep it weaker.
    parser.add_argument("--ssim-weight", type=float, default=0.05)
    parser.add_argument("--ssim-start", type=int, default=600)
    parser.add_argument("--ssim-ramp", type=int, default=800)

    parser.add_argument("--opacity-reg", type=float, default=1e-3)
    parser.add_argument("--size-reg", type=float, default=1e-4)

    parser.add_argument("--background", type=float, default=1.0)

    parser.add_argument("--print-every", type=int, default=25)

    # GIF-friendly render saving. Defaults match:
    # save every image for first 50 steps, every 5th until 150, then every 25.
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--save-all-until", type=int, default=50)
    parser.add_argument("--save-mid-until", type=int, default=150)
    parser.add_argument("--save-mid-every", type=int, default=5)

    # ----------------------------
    # Densification / pruning
    # ----------------------------
    parser.add_argument("--enable-densify", action="store_true")
    parser.add_argument("--enable-prune", action="store_true")

    parser.add_argument("--max-triangles", type=int, default=2500)
    parser.add_argument("--min-triangles", type=int, default=64)

    parser.add_argument("--densify-start", type=int, default=200)
    parser.add_argument("--densify-stop", type=int, default=1200)
    parser.add_argument("--densify-every", type=int, default=200)
    parser.add_argument("--densify-grad-threshold", type=float, default=1e-5)
    parser.add_argument("--densify-top-frac", type=float, default=0.08)
    parser.add_argument("--densify-max-splits", type=int, default=128)
    parser.add_argument("--densify-min-area", type=float, default=1e-5)
    parser.add_argument("--densify-min-opacity", type=float, default=0.12)
    parser.add_argument("--densify-vertex-jitter", type=float, default=0.001)
    parser.add_argument("--densify-depth-jitter", type=float, default=0.001)
    parser.add_argument("--densify-opacity-logit-offset", type=float, default=-0.35)

    parser.add_argument("--prune-every", type=int, default=200)
    parser.add_argument("--prune-opacity", type=float, default=0.025)
    parser.add_argument("--prune-area", type=float, default=1e-7)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
