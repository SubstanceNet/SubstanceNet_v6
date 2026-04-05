"""
System Classification: src.data.dynamic_primitives
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Dynamic 2D Primitives Generator
Generates sequences of frames with moving geometric primitives.
Each sequence has exact ground truth for motion parameters.
Format: [B, T, 1, H, W]

Changelog:
    2026-03-17 v1.0.0
"""

import math
import torch
import numpy as np
from typing import Dict, Tuple, Optional


def draw_circle(canvas, cx, cy, radius):
    H, W = canvas.shape
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    canvas[dist <= radius] = 1.0
    return canvas


def draw_square(canvas, cx, cy, size, theta=0.0):
    H, W = canvas.shape
    half = size / 2
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = []
    for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
        rx = cx + dx * cos_t - dy * sin_t
        ry = cy + dx * sin_t + dy * cos_t
        corners.append((rx, ry))
    y, x = np.ogrid[:H, :W]
    inside = np.ones((H, W), dtype=bool)
    for i in range(4):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 4]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        inside &= (cross >= 0)
    canvas[inside] = 1.0
    return canvas


def draw_triangle(canvas, cx, cy, size, theta=0.0):
    H, W = canvas.shape
    r = size / math.sqrt(3)
    corners = []
    for k in range(3):
        angle = theta + 2 * math.pi * k / 3 - math.pi / 2
        vx = cx + r * math.cos(angle)
        vy = cy + r * math.sin(angle)
        corners.append((vx, vy))
    y, x = np.ogrid[:H, :W]
    inside = np.ones((H, W), dtype=bool)
    for i in range(3):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % 3]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        inside &= (cross >= 0)
    canvas[inside] = 1.0
    return canvas


def draw_line(canvas, cx, cy, length, theta=0.0, width=2.0):
    H, W = canvas.shape
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_len = length / 2
    y, x = np.ogrid[:H, :W]
    dx = x - cx
    dy = y - cy
    along = dx * cos_t + dy * sin_t
    perp = -dx * sin_t + dy * cos_t
    mask = (np.abs(along) <= half_len) & (np.abs(perp) <= width / 2)
    canvas[mask] = 1.0
    return canvas


DRAW_FUNCS = {
    0: ("circle", draw_circle),
    1: ("square", draw_square),
    2: ("triangle", draw_triangle),
    3: ("line", draw_line),
}


def generate_sequence(
    primitive_type=0, num_frames=8, image_size=32,
    init_cx=None, init_cy=None, init_size=6.0, init_theta=0.0,
    dx=1.5, dy=0.5, dtheta=0.1, dscale=1.0, noise_std=0.02,
):
    """
    Generate a sequence of frames with a moving primitive.
    Returns: (frames [T,1,H,W], motion dict, label int)
    """
    H = W = image_size
    if init_cx is None: init_cx = W / 2
    if init_cy is None: init_cy = H / 2
    name, draw_func = DRAW_FUNCS[primitive_type]

    frames = []
    motion = {
        "cx": [], "cy": [], "size": [], "theta": [],
        "dx": dx, "dy": dy, "dtheta": dtheta, "dscale": dscale,
        "primitive": name,
    }

    cx, cy = init_cx, init_cy
    size = init_size
    theta = init_theta

    for t in range(num_frames):
        canvas = np.zeros((H, W), dtype=np.float32)
        if primitive_type == 0:
            draw_circle(canvas, cx, cy, size / 2)
        elif primitive_type == 1:
            draw_square(canvas, cx, cy, size, theta)
        elif primitive_type == 2:
            draw_triangle(canvas, cx, cy, size, theta)
        elif primitive_type == 3:
            draw_line(canvas, cx, cy, size, theta)

        if noise_std > 0:
            canvas += np.random.randn(H, W).astype(np.float32) * noise_std
            canvas = np.clip(canvas, 0, 1)

        frames.append(canvas)
        motion["cx"].append(cx)
        motion["cy"].append(cy)
        motion["size"].append(size)
        motion["theta"].append(theta)

        cx += dx
        cy += dy
        theta += dtheta
        size *= dscale
        cx = cx % W
        cy = cy % H

    frames_tensor = torch.from_numpy(np.stack(frames)).float().unsqueeze(1)
    return frames_tensor, motion, primitive_type


def generate_batch(
    batch_size=16, num_frames=8, image_size=32,
    random_motion=True, motion_types="all",
):
    """
    Generate batch of dynamic sequences.
    Returns: (frames [B,T,1,H,W], motions list, labels [B])
    """
    all_frames, all_motions, all_labels = [], [], []

    for _ in range(batch_size):
        prim = np.random.randint(0, 4)
        if random_motion:
            if motion_types == "translate":
                _dx = np.random.uniform(-2.5, 2.5)
                _dy = np.random.uniform(-2.5, 2.5)
                _dtheta, _dscale = 0.0, 1.0
            elif motion_types == "rotate":
                _dx, _dy = 0.0, 0.0
                _dtheta = np.random.uniform(-0.3, 0.3)
                _dscale = 1.0
            elif motion_types == "scale":
                _dx, _dy, _dtheta = 0.0, 0.0, 0.0
                _dscale = np.random.uniform(0.9, 1.1)
            else:
                _dx = np.random.uniform(-2.0, 2.0)
                _dy = np.random.uniform(-2.0, 2.0)
                _dtheta = np.random.uniform(-0.2, 0.2)
                _dscale = np.random.uniform(0.95, 1.05)
        else:
            _dx, _dy, _dtheta, _dscale = 1.5, 0.5, 0.1, 1.0

        init_size = np.random.uniform(4.0, 10.0)
        init_theta = np.random.uniform(0, 2 * math.pi)
        margin = image_size * 0.3
        init_cx = np.random.uniform(margin, image_size - margin)
        init_cy = np.random.uniform(margin, image_size - margin)

        frames, motion, label = generate_sequence(
            primitive_type=prim, num_frames=num_frames,
            image_size=image_size,
            init_cx=init_cx, init_cy=init_cy,
            init_size=init_size, init_theta=init_theta,
            dx=_dx, dy=_dy, dtheta=_dtheta, dscale=_dscale,
        )
        all_frames.append(frames)
        all_motions.append(motion)
        all_labels.append(label)

    batch_frames = torch.stack(all_frames)
    batch_labels = torch.tensor(all_labels, dtype=torch.long)
    return batch_frames, all_motions, batch_labels
