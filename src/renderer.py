import math

import numpy as np
import pygame
from scipy.spatial import KDTree

from render_utils import Pixel, Point3D, Triangle, load_points3D

try:
	import torch
except ImportError:
	torch = None


class Globals:
	SCREEN_WIDTH = 1200
	SCREEN_HEIGHT = 800
	focalLength = 500

	triangles: list[Triangle] = []
	pointcloudData: list[Point3D] = []

	cameraPosition = np.array([0.0, 0.0, -3.001], dtype=np.float32)
	R = np.eye(3, dtype=np.float32)
	delta = 0.1
	yaw = 0.05
	pitch = 0.05

	colorAccumulator = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.float32)
	transmittanceBuffer = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.float32)

	show_debug = True
	chunk_k = 100
	chunk_indices = []


def vertex_shader(v: np.ndarray, p: Pixel) -> None:
	p_camera = np.array(v, dtype=np.float32) - Globals.cameraPosition
	x, y, z = p_camera @ Globals.R

	p.zinv = 1.0 / z if z != 0 else 0.0
	if z > 1e-6:
		p.x = int(Globals.focalLength * x / z) + (Globals.SCREEN_WIDTH // 2)
		p.y = int(Globals.focalLength * y / z) + (Globals.SCREEN_HEIGHT // 2)
	else:
		p.x = -1
		p.y = -1


def interpolate_pixel(a: Pixel, b: Pixel, num_results: int) -> list[Pixel]:
	if num_results <= 0:
		return []
	if num_results == 1:
		return [Pixel(a.x, a.y, a.zinv)]

	x_step = (b.x - a.x) / (num_results - 1)
	y_step = (b.y - a.y) / (num_results - 1)
	z_step = (b.zinv - a.zinv) / (num_results - 1)

	results = []
	for i in range(num_results):
		x = int(round(a.x + x_step * i))
		y = int(round(a.y + y_step * i))
		zinv = a.zinv + z_step * i
		results.append(Pixel(x, y, zinv))
	return results


def update_rotation(yaw: float, pitch: float) -> None:
	yawMat = np.array(
		[
			[math.cos(yaw), 0, -math.sin(yaw)],
			[0, 1, 0],
			[math.sin(yaw), 0, math.cos(yaw)],
		],
		dtype=np.float32,
	)
	Globals.R = Globals.R @ yawMat

	if pitch != 0:
		cos_p = math.cos(pitch)
		sin_p = math.sin(pitch)
		pitchMat = np.array(
			[[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]],
			dtype=np.float32,
		)
		Globals.R = Globals.R @ pitchMat


def ComputePolygonRows(vertexPixels: list[Pixel]):
	minY = min(p.y for p in vertexPixels)
	maxY = max(p.y for p in vertexPixels)
	rows = int(maxY - minY) + 1

	leftPixels = [Pixel(float("inf"), int(minY) + i, float("inf")) for i in range(rows)]
	rightPixels = [Pixel(-float("inf"), int(minY) + i, -float("inf")) for i in range(rows)]

	for i in range(len(vertexPixels)):
		j = (i + 1) % len(vertexPixels)
		start = vertexPixels[i]
		end = vertexPixels[j]
		if start.y == end.y:
			continue

		if start.y < end.y:
			top = start
			bottom = end
		else:
			top = end
			bottom = start

		steps = int(bottom.y - top.y) + 1
		edgePixels = interpolate_pixel(top, bottom, steps)

		for p in edgePixels:
			row = int(p.y - minY)
			if 0 <= row < rows:
				if p.x < leftPixels[row].x or (p.x == leftPixels[row].x and p.zinv > leftPixels[row].zinv):
					leftPixels[row].x = p.x
					leftPixels[row].zinv = p.zinv
				if p.x > rightPixels[row].x or (p.x == rightPixels[row].x and p.zinv > rightPixels[row].zinv):
					rightPixels[row].x = p.x
					rightPixels[row].zinv = p.zinv
	return leftPixels, rightPixels


def SDF(vertexPixels: list[Pixel]):
	vertices = [p.to2dVector() for p in vertexPixels]
	Ls = []

	for i in range(len(vertices)):
		edge = vertices[i - 1] - vertices[i]
		normal = np.array([-edge[1], edge[0]], dtype=np.float32)

		if normal @ (vertices[i - 2] - vertices[i]) > 0:
			normal = -normal

		norm = np.linalg.norm(normal)
		if norm <= 1e-7:
			return []

		ni = normal / norm
		di = -(ni) @ vertices[i - 1]
		Ls.append((ni, di))
	return Ls


def incenter(vertexPixels: list[Pixel]):
	v = [p.to2dVector() for p in vertexPixels]
	a = np.linalg.norm(v[0] - v[1])
	b = np.linalg.norm(v[0] - v[2])
	c = np.linalg.norm(v[2] - v[1])
	perimeter = a + b + c
	if perimeter < 1e-7:
		return None
	return (a * v[2] + b * v[1] + c * v[0]) / perimeter


def DrawRows(leftPixels: list[Pixel], rightPixels: list[Pixel], triangle: Triangle, vertexPixels: list[Pixel]):
	color_rgb = (triangle.color * 255).astype(np.float32)
	Ls = SDF(vertexPixels)
	if not Ls:
		return

	s = incenter(vertexPixels)
	if s is None:
		return

	ns = np.array([ls[0] for ls in Ls], dtype=np.float32)
	ds = np.array([ls[1] for ls in Ls], dtype=np.float32)

	phiS = np.max(s @ ns.T + ds)
	if phiS >= 0:
		return

	for row in range(len(leftPixels)):
		left = leftPixels[row]
		right = rightPixels[row]
		if left.x > right.x:
			continue

		y = int(left.y)
		if y < 0 or y >= Globals.SCREEN_HEIGHT:
			continue

		x_start = max(0, int(round(left.x)))
		x_end = min(Globals.SCREEN_WIDTH - 1, int(round(right.x)))
		if x_start > x_end:
			continue

		xs = np.arange(x_start, x_end + 1)
		points = np.stack([xs, np.full_like(xs, y)], axis=1)
		phiP_all = np.max(points @ ns.T + ds, axis=1)
		influence = np.power(np.clip(phiP_all / phiS, 0.0, 1.0), triangle.sigma)
		alpha_span = triangle.opacity * influence

		T_span = Globals.transmittanceBuffer[y, x_start : x_end + 1]
		color_contribution = color_rgb * (alpha_span * T_span)[:, np.newaxis]
		Globals.colorAccumulator[y, x_start : x_end + 1] += color_contribution
		Globals.transmittanceBuffer[y, x_start : x_end + 1] *= (1.0 - alpha_span)


def DrawPolygon(triangle: Triangle):
	vertices = triangle.vertices
	vertexPixels = []
	valid_vertices = len(vertices)

	for i in range(len(vertices)):
		projected = Pixel(0, 0, 0.0)
		vertex_shader(vertices[i], projected)
		if (
			projected.x == -1
			or projected.x < 0
			or projected.x >= Globals.SCREEN_WIDTH
			or projected.y < 0
			or projected.y >= Globals.SCREEN_HEIGHT
		):
			valid_vertices -= 1
		vertexPixels.append(projected)

	if valid_vertices == 0:
		return

	for p in vertexPixels:
		p.x = int(np.clip(p.x, 0, Globals.SCREEN_WIDTH - 1))
		p.y = int(np.clip(p.y, 0, Globals.SCREEN_HEIGHT - 1))

	leftPixels, rightPixels = ComputePolygonRows(vertexPixels)
	DrawRows(leftPixels, rightPixels, triangle, vertexPixels)


def initializeTriangle(point: Point3D, neighbours):
	q = np.array([point.x, point.y, point.z], dtype=np.float32)
	U = []
	for _ in range(3):
		u = np.random.uniform(-1.0, 1.0, 3)
		U.append(u / (np.linalg.norm(u) + 1e-8))

	d = (neighbours[0][1] + neighbours[1][1] + neighbours[2][1]) / 3.0
	k = 2.2
	V = [q + k * d * u for u in U]

	return Triangle(
		vertices=V,
		color=np.array([point.r, point.g, point.b]),
		opacity=0.9,
		sigma=1.16,
	)


def initialize_triangles(points: list[Point3D]) -> list[Triangle]:
	print("Start triangle initialization")
	coords = np.array([[p.x, p.y, p.z] for p in points])
	kdtree = KDTree(coords)

	triangles = []
	for point in points:
		distances, indices = kdtree.query([point.x, point.y, point.z], k=4)
		neighbours = [(points[idx], dist) for idx, dist in zip(indices[1:], distances[1:])]
		triangles.append(initializeTriangle(point, neighbours))

	print("Triangles initialized")
	return triangles


def get_depth(tri: Triangle) -> float:
	centroid = np.mean(tri.vertices, axis=0)
	v_cam = (centroid - Globals.cameraPosition) @ Globals.R
	return float(v_cam[2])


def get_chunk_indices(k: int = 100):
	coords = np.array([[p.x, p.y, p.z] for p in Globals.pointcloudData])
	tree = KDTree(coords)
	_, indices = tree.query(coords[0], k=k)
	return list(indices)


def export_chunk_to_torch(indices):
	"""
	Next training step bridge:
	export exactly the currently rendered chunk into tensors.
	"""
	if torch is None:
		return None

	verts = np.array([Globals.triangles[i].vertices for i in indices], dtype=np.float32)
	colors = np.array([Globals.triangles[i].color for i in indices], dtype=np.float32)
	opacities = np.array([Globals.triangles[i].opacity for i in indices], dtype=np.float32)
	sigmas = np.array([Globals.triangles[i].sigma for i in indices], dtype=np.float32)

	return {
		"vertices": torch.tensor(verts, dtype=torch.float32),
		"colors": torch.tensor(colors, dtype=torch.float32),
		"opacities": torch.tensor(opacities, dtype=torch.float32),
		"sigmas": torch.tensor(sigmas, dtype=torch.float32),
	}


def draw_debug_info(surface: pygame.Surface):
	if not Globals.show_debug:
		return

	font = pygame.font.SysFont("Arial", 20)
	lines = [
		f"Camera Position: {Globals.cameraPosition}",
		"Controls: WASD move, SPACE/SHIFT up-down, arrows rotate",
		f"Points: {len(Globals.pointcloudData)}",
		f"Chunk triangles: {len(Globals.chunk_indices)}",
		"F1 toggle debug",
	]

	for i, line in enumerate(lines):
		text = font.render(line, True, (255, 255, 255))
		surface.blit(text, (10, 10 + i * 25))


def main():
	Globals.pointcloudData = load_points3D("south-building/sparse/points3D.txt")
	Globals.triangles = initialize_triangles(Globals.pointcloudData)
	Globals.chunk_indices = get_chunk_indices(Globals.chunk_k)

	torch_chunk = export_chunk_to_torch(Globals.chunk_indices)
	if torch_chunk is None:
		print("Torch not installed: rendering works, tensor export skipped.")
	else:
		print("Torch export ready for training with current rendered chunk.")
		print(f"vertices tensor shape: {tuple(torch_chunk['vertices'].shape)}")

	pygame.init()
	pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
	pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
	clock = pygame.time.Clock()

	screen = pygame.display.set_mode((Globals.SCREEN_WIDTH, Globals.SCREEN_HEIGHT))
	pygame.display.set_caption("Triangle Splatting - Chunk Viewer")

	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
				Globals.show_debug = not Globals.show_debug

		keys = pygame.key.get_pressed()
		forward = Globals.R @ np.array([0, 0, 1], dtype=np.float32)
		right = Globals.R @ np.array([1, 0, 0], dtype=np.float32)
		up = np.array([0, 1, 0], dtype=np.float32)

		if keys[pygame.K_w]:
			Globals.cameraPosition += forward * Globals.delta
		if keys[pygame.K_s]:
			Globals.cameraPosition -= forward * Globals.delta
		if keys[pygame.K_a]:
			Globals.cameraPosition -= right * Globals.delta
		if keys[pygame.K_d]:
			Globals.cameraPosition += right * Globals.delta
		if keys[pygame.K_LSHIFT]:
			Globals.cameraPosition += up * Globals.delta
		if keys[pygame.K_SPACE]:
			Globals.cameraPosition -= up * Globals.delta

		dyaw = 0.0
		dpitch = 0.0
		if keys[pygame.K_LEFT]:
			dyaw = Globals.yaw
		if keys[pygame.K_RIGHT]:
			dyaw = -Globals.yaw
		if keys[pygame.K_UP]:
			dpitch = Globals.pitch
		if keys[pygame.K_DOWN]:
			dpitch = -Globals.pitch
		update_rotation(dyaw, dpitch)

		screen.fill((0, 0, 0))
		Globals.colorAccumulator.fill(0)
		Globals.transmittanceBuffer.fill(1.0)

		visible_indices = [idx for idx in Globals.chunk_indices if get_depth(Globals.triangles[idx]) > 0.1]
		sorted_indices = sorted(visible_indices, key=lambda idx: get_depth(Globals.triangles[idx]))

		for idx in sorted_indices:
			DrawPolygon(Globals.triangles[idx])

		final_image = np.clip(Globals.colorAccumulator, 0, 255).astype(np.uint8)
		final_image = np.transpose(final_image, (1, 0, 2))
		pygame.surfarray.blit_array(screen, final_image)

		draw_debug_info(screen)
		pygame.display.flip()
		clock.tick(60)

	pygame.quit()
	print("Pygame window closed.")


if __name__ == "__main__":
	main()