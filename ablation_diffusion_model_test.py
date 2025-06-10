import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from matplotlib.collections import LineCollection
import multiprocessing
from scipy.spatial import cKDTree

multiprocessing.set_start_method("spawn", force=True)
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
import sys
import os
import imageio
from matplotlib.colors import to_rgb

from scipy.interpolate import splprep, splev
import shutil
from typing import Optional, List, Tuple
import pandas as pd


# Configuration
SHAPE_SIZE = 2.0  # radius
NOISE_SCALE_FACTOR = 0.5 * SHAPE_SIZE  # for both training and inference
PLOT_LIMIT = SHAPE_SIZE * 1.5
SAVE_FLOW_VIDEO = True

NUM_INFERENCE_SAMPLES = 5000
EARLY_STOPPING_PATIENCE = np.inf  # number of "non improving" epochs to wait before early stopping. Use np.inf to disable early stopping.

USE_CLUSTERING = False
NUM_CLUSTERS = 100
T_NUM_TIMESTEPS = 100

BATCH_SIZE_DEFAULT = 4096  # 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
EVAL_EVERY_N_EPOCHS = 10
N_VAL_POINTS = 1000

MODEL_CONFIGS = {
    "all_models": [
        (32, # hidden layer size
        1,  # num of hidden layers
        ),
        (128,  # hidden layer size
        3,  # num of hidden layers
        ),
        (256,  # hidden layer size
        5,  # num of hidden layers
        ),
        (1024,  # hidden layer size
        10,  # num of hidden layers
        )
    ],
    "simple_models": [
        (128,  # hidden layer size
        3,  # num of hidden layers
        ),
        (1024,  # hidden layer size
        10,  # num of hidden layers
        )
    ],
    "test": [
        (16, # hidden layer size
        1,  # num of hidden layers
        ),
        (32,  # hidden layer size
        2,  # num of hidden layers
        ),
    ],
}

DATA_CONFIGS = {
    # same amount of data, increasing num of training steps
    "test": [
        (1e-4, # learning_rate
        1, # data_repeat_factor
        2, # num epochs
        2, # num data points
        "test", # model config name
        ),
    ],
    # same amount of data, increasing num of training steps
    "same_data_increasing_steps": [
        (1e-4, # learning_rate
        100, # data_repeat_factor
        100, # num epochs
        20, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        100, # data_repeat_factor
        1000, # num epochs
        20,  # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        100, # data_repeat_factor
        10000, # num epochs
        20, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        100, # data_repeat_factor
        100000, # num epochs
        20, # num data points
        "all_models", # model config name
        ),
    ],
    # same amount of data, increasing num of training steps
    "simple_same_data_increasing_steps": [
        (1e-4, # learning_rate
        100, # data_repeat_factor
        1000, # num epochs
        20,  # num data points
        "simple_models", # model config name
        ),
        (1e-4, # learning_rate
        100, # data_repeat_factor
        10000, # num epochs
        20, # num data points
        "simple_models", # model config name
        ),
    ],
    # same amount of data, increasing num of training steps
    "same_high_data_increasing_steps": [
        (1e-4, # learning_rate
        1, # data_repeat_factor
        1, # num epochs
        200000, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        10, # num epochs
        200000,  # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        100, # num epochs
        200000, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        1000, # num epochs
        200000, # num data points
        "all_models", # model config name
        ),
    ],
    # same number of steps, increasing amount of data
    "increasing_data_same_steps": [
        (1e-4, # learning_rate
        1, # data_repeat_factor
        500, # num epochs
        200, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        500, # num epochs
        2000, # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        100, # num epochs
        20000,  # num data points
        "all_models", # model config name
        ),
        (1e-4, # learning_rate
        1, # data_repeat_factor
        10, # num epochs
        200000, # num data points
        "all_models", # model config name
        ),
    ],
}


def compute_grid_coverage_metrics(gen_points: np.ndarray, ref_points: np.ndarray, grid_size: int = 64, bounds: tuple = None):
    """
    Computes grid-based precision, recall, and F1 score between generated and reference 2D point sets.

    Args:
        gen_points (np.ndarray): Generated samples, shape (N, 2).
        ref_points (np.ndarray): Ground truth/reference samples, shape (M, 2).
        grid_size (int): Number of bins per axis (e.g., 64x64 grid).
        bounds (tuple): Optional (xmin, xmax, ymin, ymax); if None, derived from both sets.

    Returns:
        dict: {precision, recall, f1, intersection_bins, gen_bins, ref_bins}
    """
    if bounds is None:
        all_points = np.vstack([gen_points, ref_points])
        xmin, ymin = np.min(all_points, axis=0)
        xmax, ymax = np.max(all_points, axis=0)
    else:
        xmin, xmax, ymin, ymax = bounds

    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)

    def get_bin_indices(points):
        x_idx = np.digitize(points[:, 0], x_edges) - 1
        y_idx = np.digitize(points[:, 1], y_edges) - 1
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)
        return set(zip(x_idx, y_idx))

    gen_bins = get_bin_indices(gen_points)
    ref_bins = get_bin_indices(ref_points)
    intersection = gen_bins & ref_bins

    precision = len(intersection) / len(gen_bins) if gen_bins else 0.0
    recall = len(intersection) / len(ref_bins) if ref_bins else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "intersection_bins": len(intersection),
        "gen_bins": len(gen_bins),
        "ref_bins": len(ref_bins),
        "intersection_over_ref_bins": len(intersection)/len(ref_bins),
    }


def chamfer_distance_kdtree(set_a: np.ndarray, set_b: np.ndarray, tree_b: cKDTree) -> float:
    """
    Compute the symmetric Chamfer Distance between two point clouds using KDTree.

    Args:
        set_a (np.ndarray): Generated point cloud of shape (N, D)
        set_b (np.ndarray): Reference point cloud (e.g., ground truth) of shape (M, D)
        tree_b (cKDTree): KDTree for reference data

    Returns:
        float: Chamfer Distance
    """
    # Nearest neighbor from A to B
    dists_a_to_b, _ = tree_b.query(set_a, k=1)

    # Nearest neighbor from B to A
    tree_a = cKDTree(set_a)
    dists_b_to_a, _ = tree_a.query(set_b, k=1)

    # Average squared distances in both directions
    chamfer = np.mean(dists_a_to_b ** 2) + np.mean(dists_b_to_a ** 2)
    return chamfer

def cluster_trajectories(trajectories, num_clusters=100):

    final_points = trajectories[:, -1, :]  # (N, 2)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(final_points)

    # Find one representative trajectory for each cluster (closest one to centroid)
    cluster_centers = kmeans.cluster_centers_

    selected_indices = []
    for c in range(num_clusters):
        cluster_mask = labels == c
        cluster_members = final_points[cluster_mask]
        cluster_trajs = trajectories[cluster_mask]

        # remove small clusters
        if len(cluster_members) < 10:
            continue

        if len(cluster_members) == 0:
            continue

        dists = np.linalg.norm(cluster_members - cluster_centers[c], axis=1)
        best_idx = np.argmin(dists)
        selected_indices.append(cluster_mask.nonzero()[0][best_idx])

    selected_trajectories = trajectories[selected_indices]
    return selected_trajectories


def ellipse_func(t, size):
    long_axis = size  # x-axis stretched 2x
    short_axis = size * 0.5  # y-axis normal
    x = long_axis * np.cos(t)
    y = short_axis * np.sin(t)
    return x, y


def heart_func(t, size):
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x *= size / 20
    y *= size / 20
    return x, y


def sample_segments_along_curve(
    curve_func, size, n_segments, n_samples, segment_fraction=0.03, keep_prob=0.7
):
    total_angle = 2 * np.pi
    segment_length = segment_fraction * total_angle

    samples = []

    # Evenly spaced starting points for each segment
    t_starts = np.linspace(0, total_angle, n_segments, endpoint=False)

    # Randomly decide which segments to keep
    keep_mask = np.random.rand(n_segments) < keep_prob
    kept_t_starts = t_starts[keep_mask]

    if len(kept_t_starts) == 0:
        raise ValueError("No segments kept! Try increasing keep_prob.")

    samples_per_segment = n_samples // len(kept_t_starts)

    for t0 in kept_t_starts:
        t1 = t0 + segment_length
        t = np.random.uniform(t0, t1, size=(samples_per_segment,))
        x, y = curve_func(t, size)
        pts = np.stack([x, y], axis=1)
        samples.append(pts)

    samples = np.concatenate(samples, axis=0)

    # Clean Resampling: Duplicate if needed
    if samples.shape[0] < n_samples:
        print(
            f"Warning: {samples.shape[0]} samples generated, but {n_samples} required. Resampling..."
        )
        indices = np.random.choice(samples.shape[0], size=n_samples, replace=True)
        samples = samples[indices]
    else:
        samples = samples[:n_samples]

    return samples


def sample_along_edges_uniformly(vertices, n_samples):
    edges = [
        (vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))
    ]
    lengths = [np.linalg.norm(b - a) for a, b in edges]
    total_length = sum(lengths)
    distances = np.linspace(0, total_length, n_samples, endpoint=False)

    samples = []
    acc = 0
    i = 0
    for d in distances:
        while d > acc + lengths[i]:
            acc += lengths[i]
            i += 1
        a, b = edges[i]
        t = (d - acc) / lengths[i]
        samples.append((1 - t) * a + t * b)
    return np.stack(samples)


def sample_shape_points(
    shape: str, size: float, n_samples: int, near_vertex_edges=False
):
    if shape == "ellipse":
        if near_vertex_edges:
            data = sample_segments_along_curve(
                ellipse_func,
                size,
                n_segments=20,
                n_samples=n_samples,
                segment_fraction=0.01,
                keep_prob=0.5,
            )
        else:
            angles = np.random.rand(n_samples) * 2 * np.pi
            long_axis = size
            short_axis = size * 0.5
            x = long_axis * np.cos(angles)
            y = short_axis * np.sin(angles)
            data = np.stack([x, y], axis=1)

    elif shape == "star":
        R = size
        r = 0.5 * size
        outer_angles = np.linspace(np.pi / 2, 5 * np.pi / 2, 6)[:-1]
        inner_angles = outer_angles + np.pi / 5
        outer_pts = np.stack(
            [R * np.cos(outer_angles), R * np.sin(outer_angles)], axis=1
        )
        inner_pts = np.stack(
            [r * np.cos(inner_angles), r * np.sin(inner_angles)], axis=1
        )
        vertices = []
        for i in range(5):
            vertices.append(outer_pts[i])
            vertices.append(inner_pts[i])
        vertices = np.array(vertices)

        if near_vertex_edges:
            data = sample_near_vertex_edges(
                vertices,
                n_samples,
                min_spread=0.2,
                max_spread=0.2,
                vertex_keep_prob=0.7,
            )
            # convex_vertices = vertices[::2]  # take every second point: 0, 2, 4, 6, 8
            # data = sample_near_vertex_edges(convex_vertices, n_samples, size)
        else:
            data = sample_along_edges_uniformly(vertices, n_samples)

    elif shape == "heart":
        if near_vertex_edges:
            data = sample_segments_along_curve(
                heart_func,
                size,
                n_segments=25,
                n_samples=n_samples,
                segment_fraction=0.02,
                keep_prob=0.5,
            )
        else:
            # Full random t for smooth heart
            t = np.random.rand(n_samples) * 2 * np.pi
            x = 16 * np.sin(t) ** 3
            y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
            x *= size / 20  # scaling down
            y *= size / 20
            data = np.stack([x, y], axis=1)

    elif shape == "rectangle":
        # Rectangle with width and height = 2 * size
        half_w = size
        half_h = size
        vertices = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float32,
        )

        if near_vertex_edges:
            # Small data: sample near 4 corners
            data = sample_near_vertex_edges(vertices, n_samples, max_spread=0.1)
        else:
            # Big data: sample uniformly along rectangle edges
            data = sample_along_edges_uniformly(vertices, n_samples)

    else:
        raise ValueError(f"Unknown shape_type: {shape}")

    return data.astype(np.float32)


# Model
class DiffusionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 2,
        hidden_size: int = 128,
        hidden_layers: int = 3,
    ):
        super(DiffusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Training
# with early stopping and best model restoration
def train_model(
    model: torch.nn.Module,
    points: torch.Tensor,
    num_epochs: int,
    learning_rate: float,
    alpha_bars: torch.Tensor,
    label: Optional[str] = None,
    patience: int = 50,
    batch_size: Optional[int] = None,
    shape_type: Optional[str] = None,
    device = DEVICE,
    val_points: torch.Tensor = None,
) -> Tuple[List[float], int, List[float]]:
    """
    Trains a diffusion model using denoising score matching with cosine learning rate decay and early stopping.

    Args:
        model (torch.nn.Module): The neural network model to train.
        points (torch.Tensor): The training data, shape (N, D).
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Initial learning rate for the optimizer.
        alpha_bars (torch.Tensor): Precomputed alpha_bar schedule of shape (T,).
        label (Optional[str]): Descriptive label for logging (e.g. experiment name).
        patience (int): Early stopping patience (in epochs without improvement).
        batch_size (Optional[int]): Mini-batch size. Must be set.
        shape_type (Optional[str]): Shape name (used for logging).
        val_points (Optional [torch.Tensor]): Validation points (N_val, D).

    Returns:
        num_learning_steps: Number of updates.
        logs: Dict with metrics across training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    loss_fn = nn.MSELoss()
    print(f"[{shape_type.upper()}] -> Starting training: {label}")
    model.train()
    epoch_losses = []
    epoch_val_losses = []
    epoch_cds = []
    epoch_precisions = []
    epoch_recalls = []
    epoch_f1s = []
    log_epochs = []

    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    num_learning_steps = 0

    # for cd calculation
    ref_points = points.cpu().numpy()
    tree_b = cKDTree(ref_points)

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(points.shape[0])
        points_shuffled = points[perm]
        running_loss = 0.0

        for i in range(0, points_shuffled.size(0), batch_size):
            x0 = points_shuffled[i : i + batch_size]
            t = torch.randint(low=1, high=T_NUM_TIMESTEPS + 1, size=(x0.size(0),), device=device)
            alpha_bar_t = alpha_bars[t - 1].unsqueeze(1)
            epsilon = NOISE_SCALE_FACTOR * torch.randn_like(x0)
            x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
            t_norm = (t - 1).unsqueeze(1).float() / (T_NUM_TIMESTEPS - 1)
            model_input = torch.cat([x_t, t_norm], dim=1)
            pred_epsilon = model(model_input)
            loss = loss_fn(pred_epsilon, epsilon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_learning_steps += 1
            running_loss += loss.item() * x0.size(0)

        scheduler.step()  # Update learning rate

        epoch_loss = running_loss / points.shape[0]

        # Optional: log learning rate
        current_lr = scheduler.get_last_lr()[0]

        # === Evaluation Block ===
        if epoch % EVAL_EVERY_N_EPOCHS == 0 or epoch == 1:
            # === Evaluate Metrics at end of epoch ===
            NUM_EVAL_SAMPLES = 100
            model.eval()
            with torch.no_grad():
                if val_points is not None:
                    # Validation Loss
                    val_loss = 0.0
                    val_perm = torch.randperm(val_points.shape[0])
                    val_shuffled = val_points[val_perm]

                    for i in range(0, val_shuffled.size(0), batch_size):
                        x0 = val_shuffled[i : i + batch_size]
                        t = torch.randint(1, T_NUM_TIMESTEPS + 1, (x0.size(0),), device=device)
                        alpha_bar_t = alpha_bars[t - 1].unsqueeze(1)
                        epsilon = NOISE_SCALE_FACTOR * torch.randn_like(x0)
                        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon
                        t_norm = (t - 1).unsqueeze(1).float() / (T_NUM_TIMESTEPS - 1)
                        model_input = torch.cat([x_t, t_norm], dim=1)

                        pred_epsilon = model(model_input)
                        val_loss += loss_fn(pred_epsilon, epsilon).item() * x0.size(0)

                    val_loss /= val_points.shape[0]
                    epoch_val_losses.append(val_loss)

                # Chamfer, Precision, Recall
                initial_noise = torch.randn((NUM_EVAL_SAMPLES, 2), device=device)  # sampling noise
                trajectories = sample_points_with_trajectory(
                    model=model,
                    num_samples=NUM_EVAL_SAMPLES,
                    initial_noise=initial_noise,
                    alpha_bars=alpha_bars,
                    device=device,
                )
                gen_points = trajectories[:, -1, :]  # final denoised outputs

                # Calculate Chamfer distance
                cd = chamfer_distance_kdtree(gen_points, ref_points, tree_b)
                epoch_cds.append(cd)

                # Calculate precision and recall
                coverage_metrics = compute_grid_coverage_metrics(gen_points, ref_points)
                epoch_f1s.append(coverage_metrics["f1"])
                epoch_precisions.append(coverage_metrics["precision"])
                epoch_recalls.append(coverage_metrics["recall"])
                epoch_losses.append(epoch_loss)
                log_epochs.append(epoch)
            log_msg = f"[Shape: {shape_type}]: Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.6f}, LR: {current_lr:.6f}, Chamfer Distance: {cd:.6f}"
            if val_points is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
            print(log_msg)

        # === Early Stopping ===
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[Shape: {shape_type}]: Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
            )
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[Shape: {shape_type}]: Restored best model with loss {best_loss:.6f}")
    else:
        print(
            "Error: [Shape: {shape_type}]: No best model state found. Training may not have been successful."
        )
        sys.exit(1)

    logs = {
        "training_losses": epoch_losses,
        "chamfer_distances": epoch_cds,
        "f1s": epoch_f1s,
        "precisions": epoch_precisions,
        "recalls": epoch_recalls,
        "epochs": log_epochs
    }
    if val_points is not None:
        logs["val_losses"] = epoch_val_losses
    return num_learning_steps, logs


def sample_points_with_trajectory(
    model: torch.nn.Module,
    num_samples: int,
    initial_noise: torch.Tensor,
    alpha_bars: torch.Tensor,
    num_denoising_timesteps: int = T_NUM_TIMESTEPS,
    device=DEVICE,
) -> np.ndarray:
    """
    Generate denoising trajectories from a diffusion model starting from initial noise.

    This function performs the full reverse diffusion process by iteratively denoising
    the samples using the given model, while tracking the trajectory of each sample.

    Args:
        model (torch.nn.Module): The diffusion model which predicts epsilon (noise).
        num_samples (int): Number of samples to generate.
        initial_noise (torch.Tensor): Tensor of shape (num_samples, 2) containing
            the initial Gaussian noise from which to start denoising.
        num_denoising_timesteps: number of denoising steps.
        alpha_bars: Precomputed alpha_bar schedule of shape (T,).

    Returns:
        np.ndarray: An array of shape (num_samples, T + 1, 2) representing the full
            denoising trajectories for each sample from t = T down to t = 0.
            If clustering is enabled, returns clustered trajectories.
    """
    x_t = initial_noise.to(device)
    trajectories = [x_t.cpu().numpy()]
    for t in range(num_denoising_timesteps, 0, -1):
        t_cur = torch.full((num_samples,), t, device=device, dtype=torch.long)
        t_norm = (t_cur - 1).unsqueeze(1).float() / (num_denoising_timesteps - 1)
        model_input = torch.cat([x_t, t_norm], dim=1)
        with torch.no_grad():
            pred_epsilon = model(model_input)
        alpha_bar_t = alpha_bars[t - 1]
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_epsilon) / torch.sqrt(
            alpha_bar_t
        )
        if t > 1:
            z = torch.randn_like(x_t)
            alpha_bar_prev = (
                alpha_bars[t - 2] if t - 2 >= 0 else torch.tensor(1.0, device=device)
            )
            x_t = (
                torch.sqrt(alpha_bar_prev) * pred_x0
                + torch.sqrt(1 - alpha_bar_prev) * z
            )
        else:
            x_t = pred_x0
        trajectories.append(x_t.cpu().numpy())
    trajectories = np.stack(trajectories, axis=1)  # (num_samples, num_steps+1, 2)

    if USE_CLUSTERING:
        clustered_trajectories = cluster_trajectories(trajectories, num_clusters=NUM_CLUSTERS)
    else:
        clustered_trajectories = trajectories
    return clustered_trajectories


def plot_trajectories(
    trajectories,
    points_np,
    title=None,
    last_steps=30,
    ax=None,
    plot_denoising_lines=False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    num_trajectories, num_steps, _ = trajectories.shape
    steps_to_plot = min(last_steps, num_steps)

    if plot_denoising_lines:

        # Plot fading lines for the denoising steps
        # only plot 1000 trajectories, sampled evenly
        step_size = max(1, num_trajectories // 1000)
        for i in range(0, num_trajectories, step_size):

            traj = trajectories[
                i, -steps_to_plot::2, :
            ]  # take last steps_to_plot steps, every 5th step
            x, y = traj[:, 0], traj[:, 1]

            try:
                # Fit a spline through the points
                tck, u = splprep([x, y], s=1.0, k=2)  # smoothing factor s, degree k
                u_fine = np.linspace(0, 1, 100)  # more points for smoothness
                x_fine, y_fine = splev(u_fine, tck)
                ax.plot(
                    x_fine, y_fine, color="cyan", alpha=0.2, linewidth=0.2, zorder=1
                )
            except Exception as e:
                # fallback if spline fitting fails
                ax.plot(x, y, color="cyan", alpha=0.15, linewidth=0.2, zorder=1)

    # Plot
    start_points = trajectories[:, 0, :]
    end_points = trajectories[:, -1, :]
    connect_segments = np.stack([start_points, end_points], axis=1)
    # lc_connect = LineCollection(connect_segments, colors='green', linewidths=0.15, alpha=0.8, zorder=1)
    lc_connect = LineCollection(
        connect_segments, colors="cyan", linewidths=0.15, alpha=0.8, zorder=1
    )
    ax.add_collection(lc_connect)

    # init noise points
    ax.scatter(
        start_points[:, 0],
        start_points[:, 1],
        color="gray",
        edgecolors="gray",
        s=1,
        alpha=0.2,
        label="Input Points",
        zorder=3,
    )

    # after denoising
    ax.scatter(
        end_points[:, 0],
        end_points[:, 1],
        color="blue",
        s=40,
        edgecolors="white",
        linewidths=0.5,
        zorder=10,
        alpha=0.25,
    )

    # gound truth points
    marker_size_cur = 30 if points_np.shape[0] < 100000 else 7
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        c="orange",
        s=marker_size_cur,
        linewidths=0.1,
        alpha=1.0,
        zorder=100,
        label="Ground Truth",
    )

    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.legend(loc="upper right", fontsize=12)

    ax.axis("equal")
    ax.axis("off")

    # Fix the plot plot_limits based on shape size
    ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
    ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)


def train_and_plot_all(
    shape_type: str,
    cur_time_str: str,
    experiment_name: str,
    device: torch.device = DEVICE,
    output_dir: str = None,
) -> None:
    """
    Run training and evaluation for multiple model/data configurations on a given 2D shape.

    This function performs the full experimental pipeline:
        - Loads a set of model and data configs for a given experiment
        - Trains each model using a diffusion score-matching objective
        - Evaluates generalization using Chamfer distance and precision/recall
        - Plots results including trajectories, training curves, and evaluation metrics
        - Optionally generates denoising videos

    Args:
        shape_type (str): The name of the geometric shape (e.g. "star", "ellipse", etc.).
        cur_time_str (str): Timestamp or version string used to isolate output directory.
        experiment_name (str): Name of the ablation experiment (must match a key in DATA_CONFIGS).
        device (torch.device): Torch device to run model training (e.g. 'cuda' or 'cpu').

    Returns:
        None. Saves all plots, metrics, and optionally videos to disk under results/ablation/.
    """

    print(
        f"\n========================\nTraining on shape: {shape_type}\n========================"
    )
    # Prepare saving directory

    save_dir = f"results/ablation/{shape_type}/{experiment_name}/{cur_time_str}"

    if output_dir is not None:
        save_dir = output_dir +f"/{save_dir}"
    if os.path.exists(save_dir):
        print(f"[{shape_type.upper()}] Removing old folder: '{save_dir}'")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"------Saving results in {save_dir}")

    # Generate the configs
    experiment_configs = DATA_CONFIGS[experiment_name]
    configs = []
    for learning_rate, data_repeat_factor, num_epochs, num_data_points, model_config_name in experiment_configs:
        model_configs = MODEL_CONFIGS[model_config_name]
        for (hidden_size, hidden_layer) in model_configs:
            points = torch.from_numpy(sample_shape_points(
                shape_type, SHAPE_SIZE, num_data_points, near_vertex_edges=False)).to(device)
            configs.append(
                (
                    points,
                    hidden_size,
                    hidden_layer,
                    BATCH_SIZE_DEFAULT,
                    num_data_points,
                    learning_rate,
                    data_repeat_factor,
                    num_epochs
                ))
    # --- preempt grid plot axes
    n_model_architectures = len(model_configs)
    # trajectories grid
    fig, axs = plt.subplots(
        int(len(configs)/n_model_architectures), n_model_architectures,
        figsize=(4*n_model_architectures, 1.7*len(configs)))
    axs = axs.flatten()
    # precision and recall grid
    fig_pr, axs_pr = plt.subplots(
        int(len(configs)/n_model_architectures), n_model_architectures,
        figsize=(4*n_model_architectures, 1.7*len(configs)))
    axs_pr = axs_pr.flatten()

    # -----Run the ablation over all configs
    all_losses = []
    all_val_losses = []
    all_labels = []
    all_cds = []
    eval_results = []
    # training hyperparameters
    betas = torch.linspace(1e-4, 0.02, T_NUM_TIMESTEPS).to(device)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    # use the same noise for all the models
    shared_initial_noise = NOISE_SCALE_FACTOR * torch.randn(
        NUM_INFERENCE_SAMPLES, 2
    ).to(device)
    # Full distribution for eval
    full_points = sample_shape_points(
                shape_type, SHAPE_SIZE, 100000, near_vertex_edges=False)
    val_points = full_points[np.random.choice(full_points.shape[0], N_VAL_POINTS, replace=False)]
    val_points_tensor = torch.from_numpy(val_points).to(device)

    for idx, (
        points_tensor,
        hidden_size,
        hidden_layer,
        batch_size,
        num_data_points,
        learning_rate,
        data_repeat_factor,
        num_epochs,
    ) in enumerate(configs):

        # Prepare the input data
        points_tensor = points_tensor.repeat((data_repeat_factor, 1))
        # Prepare the model
        model = DiffusionMLP(
            input_dim=3,
            output_dim=2,
            hidden_size=hidden_size,
            hidden_layers=hidden_layer,
        ).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params: .1f}")
        label =  f"Data({num_data_points}), Model({num_params})"

        # Training
        num_learning_steps, logs = train_model(
            model,
            points_tensor,
            num_epochs,
            learning_rate,
            alpha_bars,
            label=label,
            patience=EARLY_STOPPING_PATIENCE,
            batch_size=batch_size,
            shape_type=shape_type,
            val_points=val_points_tensor,
        )
        losses = logs["training_losses"]
        val_losses = logs["val_losses"]
        chamfer_distances = logs["chamfer_distances"]
        f1s = logs["f1s"]
        precisions= logs["precisions"]
        recalls = logs["recalls"]
        epochs = logs["epochs"]

        all_losses.append(losses)
        all_val_losses.append(val_losses)
        all_cds.append(chamfer_distances)
        all_labels.append(label)

        # Eval with access to the underlying data distribution
        traj = sample_points_with_trajectory(
            model, alpha_bars=alpha_bars, num_samples=NUM_INFERENCE_SAMPLES, initial_noise=shared_initial_noise
        )
        end_points = traj[:, -1, :]
        eval_result = compute_grid_coverage_metrics(gen_points=end_points, ref_points=full_points, grid_size=128)
        precision_ = eval_result["precision"]
        recall_ = eval_result["recall"]
        intersection_over_ref_bins_ = eval_result["intersection_over_ref_bins"]
        print("Eval results:\n",eval_result)
        eval_results.append(eval_result)

        # --Plotting
        points_np = points_tensor.cpu().numpy()
        # Plot trajectory in grid plot
        ax = axs[idx]
        title_name = (
            f"{label},\n{num_epochs} epochs, \n{num_learning_steps} steps, "
            f"\n precision: {precision_:.2f}, recall: {recall_:.2f},\n "
            f"I/R_bins: {intersection_over_ref_bins_:.2f}")
        plot_trajectories(traj, points_np, title=title_name, last_steps=30, ax=ax)
        ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        # Plot precision in grid plot
        ax_pr = axs_pr[idx]
        ax_pr.plot(epochs, precisions, label="precision", linewidth=2)
        ax_pr.plot(epochs, recalls, label="recall", linewidth=2)
        ax_pr.set_xlabel("Epoch")
        ax_pr.set_ylabel("Precisions/Recall")
        ax_pr.set_ylim(-0.01, 1.01)
        ax_pr.set_title(title_name)
        ax_pr.legend(loc="upper right", fontsize=12)

        # Create a single figure for all metrics
        fig_combined, axs_combined = plt.subplots(2, 3, figsize=(18, 10))
        axs_combined = axs_combined.flatten()
        # 1. Trajectories
        plot_trajectories(traj, points_np, title=None, last_steps=30, ax=axs_combined[0])
        axs_combined[0].set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        axs_combined[0].set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        axs_combined[0].axis("equal")
        axs_combined[0].axis("off")
        axs_combined[0].set_title("Sampled Trajectories")
        # 2. MSE Loss
        axs_combined[1].plot(epochs, losses, label="training loss", linewidth=2)
        axs_combined[1].plot(epochs, val_losses, label="val loss", linewidth=2)
        axs_combined[1].set_xlabel("Epoch")
        axs_combined[1].set_ylabel("MSE Loss")
        axs_combined[1].set_title("Training Loss")
        # 3. Chamfer Distance
        axs_combined[2].plot(epochs, chamfer_distances, label=label, linewidth=2)
        axs_combined[2].set_xlabel("Epoch")
        axs_combined[2].set_ylabel("Chamfer Distance")
        axs_combined[2].set_title("Chamfer Distance")
        axs_combined[2].set_ylim(-0.01, 50)
        # 4. Precision
        axs_combined[3].plot(epochs, precisions, label=label, linewidth=2)
        axs_combined[3].set_xlabel("Epoch")
        axs_combined[3].set_ylabel("Precision")
        axs_combined[3].set_title("Precision")
        axs_combined[3].set_ylim(-0.01, 1.01)
        # 5. Recall
        axs_combined[4].plot(epochs, recalls, label=label, linewidth=2)
        axs_combined[4].set_xlabel("Epoch")
        axs_combined[4].set_ylabel("Recall")
        axs_combined[4].set_title("Recall")
        axs_combined[4].set_ylim(-0.01, 1.01)
        # 6. F1 Score
        axs_combined[5].plot(epochs, f1s, label=label, linewidth=2)
        axs_combined[5].set_xlabel("Epoch")
        axs_combined[5].set_ylabel("F1 Score")
        axs_combined[5].set_title("F1 Score")
        axs_combined[5].set_ylim(-0.01, 1.01)
        # Final formatting
        for ax_ in axs_combined[1:]:
            ax_.grid(True)
            ax_.legend()
        fig_combined.suptitle(title_name)
        # Save figure
        model_type = f"({num_params:.1f})"
        save_name = f"metrics_shape_{shape_type}_Model_is_{model_type}_Data_is_{num_data_points}_epochs_{num_epochs}_hid_layer_size_{hidden_size}_num_hid_layers_{hidden_layer}.png"
        combined_path = os.path.join(save_dir, save_name)
        plt.tight_layout()
        fig_combined.savefig(combined_path, dpi=300)
        plt.close(fig_combined)
        print(f"Saved metrics figure to: {save_name}")

        # save denoising video
        if SAVE_FLOW_VIDEO:
            video_filename = os.path.join(
                save_dir, f"video_{save_name}.mp4"
            )
            save_trajectory_video(
                traj,
                shape_type=shape_type,
                save_path=video_filename,
                ground_truth_points=points_np,
            )

    # ---Save eval results
    df = pd.DataFrame(eval_results)
    # Save to pickle
    output_path = f"{save_dir}/eval_metrics.pkl"  # You can change the path if needed
    df.to_pickle(output_path)
    print(f"Saved metrics to: {output_path}")

    # ---Save grid plots
    # save trajectories grid
    plt.tight_layout()
    save_name_full = f"{save_dir}/trajectories_grid.png"
    fig.savefig(save_name_full, dpi=200)
    print(f"Saved full traj grid figure to '{save_name_full}'.")
    plt.close(fig)
    # save precision and recall grid
    plt.tight_layout()
    save_name_full = f"{save_dir}/precision_recall_grid.png"
    fig_pr.savefig(save_name_full, dpi=200)
    print(f"Saved full P/R grid figure to '{save_name_full}'.")
    plt.close(fig)

    # --- Combined plots
    # Plot combined losses
    plt.figure(figsize=(8, 6))
    for losses, label in zip(all_losses, all_labels):
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    save_name_loss = f"loss_curves.png"
    plt.savefig(os.path.join(save_dir, save_name_loss), dpi=300)
    plt.close()
    print(f"Saved loss curves figure to '{save_name_loss}'.")

    # Plot combined Chamfer Distances
    plt.figure(figsize=(8, 6))
    for cds, label in zip(all_cds, all_labels):
        plt.plot(cds, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Chamfer Distance")
    plt.title("Training CDs")
    plt.legend()
    plt.grid(True)
    save_name_cds = f"cd_curves.png"
    plt.savefig(os.path.join(save_dir, save_name_cds), dpi=300)
    plt.close()
    print(f"Saved CD curves figure to '{save_name_cds}'.")


def interpolate_color(c1, c2, alpha):
    return tuple((1 - alpha) * a + alpha * b for a, b in zip(c1, c2))


def save_trajectory_video(
    trajectories,
    shape_type,
    save_path="diffusion.mp4",
    fps=15,
    tail_length=20,
    ground_truth_points=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.collections import LineCollection
    import imageio

    num_samples, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots(figsize=(6, 6))
    frames = []

    gray_rgb = to_rgb("gray")
    blue_rgb = to_rgb("blue")
    trail_color = (0.3, 0.3, 0.3)
    min_size = 1
    max_size = 40
    subset = min(num_samples, 300)
    marker_size_cur = 30 if ground_truth_points.shape[0] < 100000 else 7

    last_rendered_frame = None

    for t in range(1, num_steps):
        ax.clear()
        ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        ax.axis("off")
        ax.set_title(
            f"{shape_type.upper()} Diffusion Step {t}/{num_steps - 1}", fontsize=14
        )

        # trail
        segments = []
        segment_colors = []
        segment_widths = []

        overall_alpha_scale = t / (num_steps - 1)

        for i in range(subset):
            for dt in range(tail_length):
                t1 = t - dt - 1
                t2 = t1 + 1
                if t1 < 0 or t2 >= num_steps:
                    continue
                p1 = trajectories[i, t1]
                p2 = trajectories[i, t2]
                fade = dt / tail_length

                base_alpha = 0.1 + 0.2 * (1 - fade)
                alpha = base_alpha * overall_alpha_scale
                width = 0.6 + 0.5 * (1 - fade)

                segments.append([p1, p2])
                segment_colors.append((*trail_color, alpha))
                segment_widths.append(width)

        if segments:
            lc = LineCollection(
                segments, colors=segment_colors, linewidths=segment_widths, zorder=1
            )
            ax.add_collection(lc)

        # points
        alpha = t / (num_steps - 1)
        sizes = min_size + alpha * (max_size - min_size)
        colors = [
            interpolate_color(gray_rgb, blue_rgb, alpha) for _ in range(num_samples)
        ]

        point_alpha = 0.25  # consistent with plot
        ax.scatter(
            trajectories[:, t, 0],
            trajectories[:, t, 1],
            s=sizes,
            c=colors,
            edgecolors="white",
            linewidths=0.2,
            alpha=point_alpha,
            zorder=2,
        )
        # real
        if ground_truth_points is not None:
            ax.scatter(
                ground_truth_points[:, 0],
                ground_truth_points[:, 1],
                c="orange",
                s=marker_size_cur,
                linewidths=0.1,
                alpha=1.0,
                zorder=100,
            )

        # save frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_argb(), dtype="uint8")
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame)
        last_rendered_frame = frame  # save last frame

    # Repeat the last frame 10 times to create a pause
    frames.extend([last_rendered_frame] * 10)

    imageio.mimsave(save_path, frames, fps=fps)
    plt.close(fig)
    print(f"Saved diffusion video to: {save_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train single task diffusion model on 2D shapes.")
    parser.add_argument(
        "--shape",
        type=str,
        default="star",
        help="List of shapes to train on.",
        choices=["star", "ellipse", "heart", "rectangle"]
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="test",
        help="Name of the experiment for logging/checkpointing purposes.",
        choices=["increasing_data_same_steps",
                 "same_data_increasing_steps",
                 "same_high_data_increasing_steps",
                 "test",
                 "simple_same_data_increasing_steps"]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
    )
    args = parser.parse_args()

    shape = args.shape
    experiment_name = args.experiment_name

    # reset seed for reproducibility
    np.random.seed(SEED)

    processes = []
    cur_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    train_and_plot_all(shape, cur_time_str, experiment_name, output_dir=args.output_dir)

    print("All shapes finished successfully.")
