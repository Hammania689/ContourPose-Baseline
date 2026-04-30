"""
Visualization utilities for training monitoring using torchvision (fast GPU ops).
Leverages eval.py for pose estimation via PnP and point cloud projection.
"""

import torch
import torchvision
from torchvision.utils import make_grid
import numpy as np
import wandb
from typing import Dict, Tuple, List, Optional
import cv2
from utils.utils import project, load_ply
import os
from PIL import Image, ImageDraw, ImageFont
from transforms3d.euler import mat2euler, euler2mat
from transforms3d.axangles import mat2axangle

def add_text_to_image(image: torch.Tensor, text: str, font_size: int = 60, height: int = 100) -> torch.Tensor:
    """
    Add text label to top of image using PIL.

    Args:
        image: [3, H, W] tensor in [0, 1] range
        text: Text to add
        font_size: Font size
        height: Height of text bar

    Returns:
        [3, H+height, W] tensor with text label
    """
    # Save original device before CPU conversion
    device = image.device

    # Convert to PIL
    img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    # Create text bar
    text_bar = Image.new('RGB', (W, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(text_bar)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageDraw.Load().getfont()

    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (W - text_width) // 2
    text_y = (height - text_height) // 2

    # Draw text
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

    # Concatenate
    text_bar_np = np.array(text_bar)
    combined = np.vstack([text_bar_np, img_np])

    # Convert back to tensor and move to original device
    combined_tensor = torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0
    return combined_tensor.to(device)


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize image from training normalization (GPU operation).

    Args:
        img: [C, H, W] or [B, C, H, W] tensor

    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = torch.tensor([0.419, 0.427, 0.424], device=img.device).view(-1, 1, 1)
    std = torch.tensor([0.184, 0.206, 0.197], device=img.device).view(-1, 1, 1)

    if img.dim() == 4:  # [B, C, H, W]
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    img = img * std + mean
    return torch.clamp(img, 0, 1)


def overlay_heatmap_on_image(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4) -> torch.Tensor:
    """Overlay heatmap on image (GPU operation)."""
    img_denorm = denormalize_image(image)

    # Sum heatmap across keypoints and normalize
    heatmap_sum = heatmap.sum(dim=1, keepdim=True)  # [B, 1, H, W]
    heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-8)

    # Apply colormap (red for high, blue for low)
    heatmap_rgb = torch.zeros_like(img_denorm)
    heatmap_rgb[:, 0] = heatmap_norm.squeeze(1)  # Red
    heatmap_rgb[:, 2] = 1 - heatmap_norm.squeeze(1)  # Blue

    overlay = alpha * heatmap_rgb + (1 - alpha) * img_denorm
    return torch.clamp(overlay, 0, 1)


def extract_keypoints_from_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Extract 2D keypoint coordinates from heatmap (argmax).
    Matches eval.py:map_2_points() logic.

    Args:
        heatmap: [B, K, H, W] heatmap

    Returns:
        [B, K, 2] keypoint coordinates (x, y)
    """
    flat_map = heatmap.view(heatmap.shape[0], heatmap.shape[1], -1)
    max_idx = torch.argmax(flat_map, dim=2)
    width = heatmap.shape[3]
    x = (max_idx / width).int().unsqueeze(dim=2)
    y = (max_idx % width).unsqueeze(dim=2)
    return torch.cat((y, x), dim=2)  # [B, K, 2]


def pnp_pose_estimation(keypoints_3d: np.ndarray, keypoints_2d: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray=np.zeros(shape=[5, 1], dtype="float64")) -> np.ndarray:
    """
    Estimate pose using PnP (matches eval.py logic).

    Args:
        keypoints_3d: [K, 3] 3D keypoints
        keypoints_2d: [K, 2] 2D keypoints
        K: [3, 3] camera intrinsics
        dist_coeffs: [5,1] Distortion coeffecients

    Returns:
        [3, 4] pose matrix [R|t]
    """
    _, R_exp, t = cv2.solvePnP(
        keypoints_3d,
        keypoints_2d.astype(np.float64),
        K,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )
    R, _ = cv2.Rodrigues(R_exp)
    pose = np.concatenate([R, t], axis=-1)
    return pose



# =============================================================================
# Keypoint and Edge Metrics (for model selection)
# =============================================================================

def compute_keypoint_metrics(
    pred_keypoints_2d: torch.Tensor,
    gt_keypoints_2d: torch.Tensor,
    pck_thresholds: List[float] = [1.0, 2.0]
) -> Dict[str, float]:
    """
    Compute keypoint detection metrics: RMSE and PCK at multiple thresholds.

    Args:
        pred_keypoints_2d: [B, N, 2] predicted keypoint coordinates (x, y)
        gt_keypoints_2d: [B, N, 2] ground truth keypoint coordinates (x, y)
        pck_thresholds: List of pixel thresholds for PCK computation

    Returns:
        Dictionary with:
        - keypoint_rmse: Root mean squared error in pixels
        - pck@{threshold}px: Percentage of correct keypoints for each threshold
    """
    # Compute per-keypoint L2 distances [B, N]
    distances = torch.norm(pred_keypoints_2d.float() - gt_keypoints_2d.float(), dim=2)

    # RMSE across all keypoints and samples
    rmse = torch.sqrt(torch.mean(distances ** 2)).item()

    metrics = {"keypoint_rmse": rmse}

    # PCK at each threshold
    for threshold in pck_thresholds:
        correct = (distances < threshold).float()
        pck = torch.mean(correct).item() * 100  # Percentage
        metrics[f"pck@{threshold}px"] = pck

    return metrics


def compute_edge_metrics(
    pred_edges: torch.Tensor,
    gt_edges: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute edge detection metrics: IoU and F1 score.

    Args:
        pred_edges: [B, 1, H, W] predicted edge logits (before sigmoid)
        gt_edges: [B, 1, H, W] ground truth binary edges
        threshold: Threshold for binarizing predictions (default: 0.5)

    Returns:
        Dictionary with:
        - edge_iou: Intersection over Union
        - edge_f1: F1 score (Dice coefficient)
        - edge_precision: Precision
        - edge_recall: Recall
    """
    # Apply sigmoid and threshold to predictions
    pred_binary = (torch.sigmoid(pred_edges) > threshold).float()
    gt_binary = (gt_edges > threshold).float()

    # Flatten spatial dimensions
    pred_flat = pred_binary.view(-1)
    gt_flat = gt_binary.view(-1)

    # Compute intersection and union
    intersection = (pred_flat * gt_flat).sum()
    pred_sum = pred_flat.sum()
    gt_sum = gt_flat.sum()
    union = pred_sum + gt_sum - intersection

    # IoU (handle edge case of empty masks)
    iou = (intersection / (union + 1e-8)).item()

    # Precision and Recall
    precision = (intersection / (pred_sum + 1e-8)).item()
    recall = (intersection / (gt_sum + 1e-8)).item()

    # F1 score
    f1 = (2 * precision * recall / (precision + recall + 1e-8))

    return {
        "edge_iou": iou,
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
    }


def compute_detection_metrics(
    pred_heatmaps: torch.Tensor,
    pred_edges: torch.Tensor,
    gt_keypoints_2d: torch.Tensor,
    gt_edges: torch.Tensor,
    pck_thresholds: List[float] = [1.0, 2.0]
) -> Dict[str, float]:
    """
    Compute all detection metrics for model selection.

    This is the main function for computing validation metrics that directly
    measure the network's prediction quality (keypoint localization and edge detection),
    rather than derived metrics like pose error which depend on PnP solving.

    Args:
        pred_heatmaps: [B, N, H, W] predicted heatmaps
        pred_edges: [B, 1, H, W] predicted edge logits
        gt_keypoints_2d: [B, N, 2] ground truth 2D keypoints
        gt_edges: [B, 1, H, W] ground truth edges

    Returns:
        Dictionary with all keypoint and edge metrics
    """
    # Extract predicted keypoints from heatmaps
    pred_keypoints_2d = extract_keypoints_from_heatmap(pred_heatmaps)

    # Compute keypoint metrics
    kp_metrics = compute_keypoint_metrics(pred_keypoints_2d, gt_keypoints_2d, pck_thresholds)

    # Compute edge metrics
    edge_metrics = compute_edge_metrics(pred_edges, gt_edges)

    # Merge all metrics
    all_metrics = {**kp_metrics, **edge_metrics}

    return all_metrics


def compute_selection_score(metrics: Dict[str, float], edge_weight: float = 1.0) -> float:
    """
    Compute combined score for model selection.

    Lower is better. Combines keypoint RMSE with edge IoU penalty.

    Formula: keypoint_rmse + edge_weight * (1 - edge_iou)

    Args:
        metrics: Dictionary containing 'keypoint_rmse' and 'edge_iou'
        edge_weight: Weight for edge IoU term (default: 1.0)

    Returns:
        Combined selection score (lower is better)
    """
    rmse = metrics.get("keypoint_rmse", float('inf'))
    iou = metrics.get("edge_iou", 0.0)

    # Selection score: lower is better
    # RMSE is in pixels (typically 0-10 for good models)
    # (1 - IoU) is in [0, 1]
    # With edge_weight=1.0, we're saying 1 pixel RMSE = 1.0 IoU penalty
    score = rmse + edge_weight * (1.0 - iou)

    return score


def compute_pose_metrics(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    """
    Compute translation and rotation errors between predicted and ground truth poses.
    
    Args:
        pred_pose: [3, 4] or [4, 4] predicted pose matrix
        gt_pose: [3, 4] or [4, 4] ground truth pose matrix
    
    Returns:
        Dictionary with translation and rotation errors
    """
    # Handle both [3, 4] and [4, 4] inputs
    if pred_pose.shape[0] == 4:
        pred_pose = pred_pose[:3, :]
    if gt_pose.shape[0] == 4:
        gt_pose = gt_pose[:3, :]
    
    # Translation error (in mm, assuming input is in mm)
    pred_t = pred_pose[:, 3]
    gt_t = gt_pose[:, 3]
    trans_error = np.linalg.norm(pred_t - gt_t)
    
    # Rotation error (geodesic distance on SO(3))
    pred_R = pred_pose[:, :3]
    gt_R = gt_pose[:, :3]
    
    # Compute relative rotation
    R_diff = pred_R @ gt_R.T
    
    # Geodesic distance (angle of rotation between the two orientations)
    # Formula: theta = arccos((trace(R) - 1) / 2)
    # Clamp to handle numerical errors
    trace_val = np.trace(R_diff)
    cos_angle = (trace_val - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rot_error_rad = np.arccos(cos_angle)
    rot_error_deg = rot_error_rad * 180 / np.pi
    
    # Euler angles (for additional diagnostics, but less reliable)
    # Using transforms3d for Euler angle conversion
    # mat2euler returns (ai, aj, ak) for the specified axes
    # Default is 'sxyz' (static xyz) which is equivalent to 'xyz' in scipy
    gt_ai, gt_aj, gt_ak = mat2euler(gt_R, axes='sxyz')
    pred_ai, pred_aj, pred_ak = mat2euler(pred_R, axes='sxyz')
    
    # Convert to degrees
    gt_euler = np.array([gt_ai, gt_aj, gt_ak]) * 180 / np.pi
    pred_euler = np.array([pred_ai, pred_aj, pred_ak]) * 180 / np.pi
    
    # Compute angle differences with proper wrapping
    def angle_diff(a1, a2):
        """Compute the smallest angle difference between two angles."""
        diff = (a1 - a2 + 180) % 360 - 180
        return abs(diff)
    
    alpha_error = angle_diff(pred_euler[0], gt_euler[0])
    beta_error = angle_diff(pred_euler[1], gt_euler[1])
    gamma_error = angle_diff(pred_euler[2], gt_euler[2])
    
    return {
        "translation_error_mm": float(trans_error),
        "rotation_error_deg": float(rot_error_deg),
        "rotation_error_rad": float(rot_error_rad),
        "alpha_error_deg": float(alpha_error),
        "beta_error_deg": float(beta_error),
        "gamma_error_deg": float(gamma_error),
    }


def compute_pose_metrics_alternative(pred_pose: np.ndarray, gt_pose: np.ndarray) -> Dict[str, float]:
    """
    Alternative implementation using transforms3d axis-angle representation.
    This is more robust and handles edge cases better.
    
    Args:
        pred_pose: [3, 4] or [4, 4] predicted pose matrix
        gt_pose: [3, 4] or [4, 4] ground truth pose matrix
    
    Returns:
        Dictionary with translation and rotation errors
    """
    # Handle both [3, 4] and [4, 4] inputs
    if pred_pose.shape[0] == 4:
        pred_pose = pred_pose[:3, :]
    if gt_pose.shape[0] == 4:
        gt_pose = gt_pose[:3, :]
    
    # Translation error
    pred_t = pred_pose[:, 3]
    gt_t = gt_pose[:, 3]
    trans_error = np.linalg.norm(pred_t - gt_t)
    
    # Rotation error using axis-angle representation
    pred_R = pred_pose[:, :3]
    gt_R = gt_pose[:, :3]
    
    # Compute relative rotation
    R_diff = pred_R @ gt_R.T
    
    # Convert to axis-angle representation
    # mat2axangle returns (axis, angle) where angle is in radians
    # The angle can be negative, so we take absolute value
    axis, rot_error_rad = mat2axangle(R_diff)
    rot_error_rad = abs(rot_error_rad)  # Ensure positive angle
    rot_error_deg = rot_error_rad * 180 / np.pi
    
    return {
        "translation_error_mm": float(trans_error),
        "rotation_error_deg": float(rot_error_deg),
        "rotation_error_rad": float(rot_error_rad),
    }


def render_point_cloud_with_depth(
    image: np.ndarray,
    pts_3d: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    point_size: int = 2,
    alpha: float = 0.7,
    debug_save_path: str = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Render 3D point cloud on image by drawing on black canvas then overlaying.

    Args:
        image: [H, W, 3] numpy array in [0, 255] RGB format
        pts_3d: [N, 3] 3D points (in mm)
        K: [3, 3] camera intrinsics
        pose: [3, 4] pose matrix [R|t]
        color: RGB color tuple for points (e.g., (0, 255, 0) for green, (255, 0, 0) for red)
               NOTE: Use RGB values since image is in RGB format, not BGR!
        point_size: Point size
        alpha: Blending factor
        debug_save_path: Optional path to save intermediate canvas

    Returns:
        [H, W, 3] image with point cloud rendered (RGB format)
    """
    H, W = image.shape[:2]

    # DETAILED DEBUG: Check pose and intrinsics
    if debug:
        print(f"  [Render Debug] === PROJECTION ANALYSIS ===")
        print(f"  [Render Debug] 3D points shape: {pts_3d.shape}")
        print(f"  [Render Debug] 3D points range: X[{pts_3d[:, 0].min():.2f}, {pts_3d[:, 0].max():.2f}], "
              f"Y[{pts_3d[:, 1].min():.2f}, {pts_3d[:, 1].max():.2f}], Z[{pts_3d[:, 2].min():.2f}, {pts_3d[:, 2].max():.2f}]")
        print(f"  [Render Debug] Camera intrinsics K:\n{K}")
        print(f"  [Render Debug] Pose RT (original):\n{pose}")

    # All units are in mm (pts_3d and pose translation)
    if debug:
        print(f"  [Render Debug] Pose RT:\n{pose}")

    # Use the project function from utils.utils
    pts_2d = project(pts_3d, K, pose)  # [N, 2]

    # Also compute depth for filtering
    R = pose[:, :3]
    t = pose[:, 3:].reshape(3, 1)
    pts_cam = np.dot(pts_3d, R.T) + t.T  # [N, 3]
    depths = pts_cam[:, 2]

    # DETAILED DEBUG: Check projection results
    if debug:
        print(f"  [Render Debug] Projected 2D points shape: {pts_2d.shape}")
        print(f"  [Render Debug] 2D points range: X[{pts_2d[:, 0].min():.2f}, {pts_2d[:, 0].max():.2f}], "
              f"Y[{pts_2d[:, 1].min():.2f}, {pts_2d[:, 1].max():.2f}]")
        print(f"  [Render Debug] Depth range (all): [{depths.min():.2f}, {depths.max():.2f}]")
        print(f"  [Render Debug] Image bounds: [0, {W}] x [0, {H}]")

    # Check how many points fall in different regions
    depth_positive = (depths > 0).sum()
    x_in_bounds = ((pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W)).sum()
    y_in_bounds = ((pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)).sum()
    if debug:
        print(f"  [Render Debug] Points with positive depth: {depth_positive}/{len(pts_3d)}")
        print(f"  [Render Debug] Points with X in bounds [0, {W}]: {x_in_bounds}/{len(pts_3d)}")
        print(f"  [Render Debug] Points with Y in bounds [0, {H}]: {y_in_bounds}/{len(pts_3d)}")

    # Check if points are clustering
    unique_x = len(np.unique(np.round(pts_2d[:, 0])))
    unique_y = len(np.unique(np.round(pts_2d[:, 1])))
    if debug:
        print(f"  [Render Debug] Unique X pixel locations: {unique_x}")
        print(f"  [Render Debug] Unique Y pixel locations: {unique_y}")

    # Filter valid points (in front of camera and within image bounds)
    valid_mask = (depths > 0)  # In front of camera
    valid_mask &= (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W)
    valid_mask &= (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)

    # Get valid points
    valid_pts_2d = pts_2d[valid_mask]
    valid_depths = depths[valid_mask]

    if debug:
        print(f"  [Render Debug] === FINAL VALID POINTS: {valid_mask.sum()}/{len(pts_3d)} ===")
        if valid_mask.sum() > 0:
            if debug:
                print(f"  [Render Debug] First 20 valid 2D points (x,y):")
            for idx in range(min(20, len(valid_pts_2d))):
                print(f"    [{idx}] ({valid_pts_2d[idx, 0]:.2f}, {valid_pts_2d[idx, 1]:.2f}) depth={valid_depths[idx]:.2f}")
            print(f"  [Render Debug] Image shape: {H}x{W}")

        if valid_mask.sum() == 0:
            print(f"  [Render Debug] No valid points! All filtered out.")
            print(f"  [Render Debug] All depths range: [{depths.min():.2f}, {depths.max():.2f}]")
            print(f"  [Render Debug] All 2D x range: [{pts_2d[:, 0].min():.2f}, {pts_2d[:, 0].max():.2f}]")
            print(f"  [Render Debug] All 2D y range: [{pts_2d[:, 1].min():.2f}, {pts_2d[:, 1].max():.2f}]")
            return image

    # Create BLACK canvas for point cloud
    point_cloud_canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Sort points by depth (far to near for proper occlusion)
    sorted_indices = np.argsort(-valid_depths)  # Descending order (far to near)

    # Draw all points
    points_drawn = 0
    if debug:
        print(f"  [Render Debug] Starting to draw {len(sorted_indices)} points...")
    for i, idx in enumerate(sorted_indices):
        x, y = valid_pts_2d[idx]
        x_int, y_int = int(round(x)), int(round(y))

        # Debug first few points
        if i < 5 and debug:
            print(f"    Point {i}: 2D=({x:.2f}, {y:.2f}) -> pixel=({x_int}, {y_int}), depth={valid_depths[idx]:.2f}")

        # Ensure within bounds
        if 0 <= x_int < W and 0 <= y_int < H:
            # Draw circle on black canvas
            cv2.circle(point_cloud_canvas, (x_int, y_int), point_size, color, -1, cv2.LINE_AA)
            points_drawn += 1

    if debug:
        print(f"  [Render Debug] Drew {points_drawn} points on canvas")

    # Check how many non-zero pixels in canvas
    non_zero_pixels = np.count_nonzero(point_cloud_canvas.sum(axis=2))
    if debug:
        print(f"  [Render Debug] Non-zero pixels in canvas: {non_zero_pixels}")

    # Save intermediate canvas if path provided
    if debug_save_path is not None:
        cv2.imwrite(debug_save_path, cv2.cvtColor(point_cloud_canvas, cv2.COLOR_RGB2BGR))
        if debug:
            print(f"  [Render Debug] Saved point cloud canvas to: {debug_save_path}")

    # Overlay point cloud canvas on original image
    # Where point_cloud_canvas is non-black, blend it with the image
    mask = (point_cloud_canvas.sum(axis=2) > 0).astype(np.float32)
    mask = mask[:, :, np.newaxis]  # [H, W, 1]

    # Blend: where mask=1, use alpha blending; where mask=0, use original image
    result = (alpha * point_cloud_canvas + (1 - alpha) * image) * mask + image * (1 - mask)
    return result.astype(np.uint8)


def visualize_batch_with_pose(
    batch_data: Dict[str, torch.Tensor],
    pred_heatmaps: torch.Tensor,
    pred_edges: torch.Tensor,
    keypoints_3d: np.ndarray,
    pts_3d: np.ndarray,
    num_samples: int = 4,
    output_dir: str = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Create comprehensive visualization with pose estimation.

    Visualization layout (per sample row):
    [Input] [Edge Input*] [GT Heatmap] [Pred Heatmap] [GT Edge] [Pred Edge] [GT Edge Overlay] [Pred Edge Overlay]

    *Edge Input column only shown when batch_data contains "edges_input".

    Args:
        batch_data: Dict with "images", "heatmaps", "edges", "K", "pose",
                    and optionally "edges_input" [B, 1, H, W]
        pred_heatmaps: [B, K, H, W]
        pred_edges: [B, 1, H, W]
        keypoints_3d: [K, 3] 3D keypoints for PnP
        pts_3d: [N, 3] CAD model point cloud for visualization
        num_samples: Number of samples to visualize
        output_dir: Optional directory to save intermediate point cloud images

    Returns:
        Grid tensor and metrics dictionary
    """
    device = batch_data["images"].device
    batch_size = min(num_samples, batch_data["images"].shape[0])

    has_edge_input = "edges_input" in batch_data and batch_data["edges_input"] is not None

    # Slice to num_samples
    images = batch_data["images"][:batch_size]
    gt_heatmaps = batch_data["heatmaps"][:batch_size]
    gt_edges = batch_data["edges"][:batch_size]
    K_batch = batch_data["K"][:batch_size]
    gt_pose_batch = batch_data["pose"][:batch_size]
    if has_edge_input:
        edges_input = batch_data["edges_input"][:batch_size]

    pred_heatmaps = pred_heatmaps[:batch_size]
    pred_edges = pred_edges[:batch_size]

    # Column labels
    labels = ["Input"]
    if has_edge_input:
        labels.append("Edge Input")
    labels += ["GT Heatmap", "Pred Heatmap", "GT Edge", "Pred Edge", "GT Edge Overlay", "Pred Edge Overlay"]
    ncols = len(labels)

    # Build visualization row by row
    all_rows = []

    all_trans_errors = []
    all_rot_errors = []

    for i in range(batch_size):
        # 1. Input image
        img_denorm = denormalize_image(images[i:i+1]).squeeze(0)

        # Build row
        row_images = [img_denorm]

        # 1b. Edge input (Laplacian fed to CNN branch) — optional
        if has_edge_input:
            edge_input_vis = edges_input[i].repeat(3, 1, 1)  # [1,H,W] → [3,H,W]
            row_images.append(edge_input_vis)

        # 2. GT heatmap overlay
        gt_heatmap_overlay = overlay_heatmap_on_image(images[i:i+1], gt_heatmaps[i:i+1]).squeeze(0)
        row_images.append(gt_heatmap_overlay)

        # 3. Pred heatmap overlay
        pred_heatmap_overlay = overlay_heatmap_on_image(images[i:i+1], pred_heatmaps[i:i+1]).squeeze(0)
        row_images.append(pred_heatmap_overlay)

        # 4. GT edge
        gt_edge_rgb = gt_edges[i].repeat(3, 1, 1)
        row_images.append(gt_edge_rgb)

        # 5. Pred edge
        pred_edge_sigmoid = torch.sigmoid(pred_edges[i]).repeat(3, 1, 1)
        row_images.append(pred_edge_sigmoid)

        # 6 & 7. Edge overlays on input image
        K_np = K_batch[i].cpu().numpy()
        gt_pose_np = gt_pose_batch[i].cpu().numpy()

        # Extract predicted keypoints for pose estimation
        pred_keypoints_2d = extract_keypoints_from_heatmap(pred_heatmaps[i:i+1])
        pred_kpts_2d = pred_keypoints_2d[0].cpu().numpy()

        # Create edge overlays (green for GT, red for pred)
        # GT edge overlay
        gt_edge_mask = gt_edges[i, 0]  # [H, W]
        gt_edge_overlay = img_denorm.clone()
        gt_edge_overlay[1] = torch.where(gt_edge_mask > 0.5, torch.ones_like(gt_edge_mask), gt_edge_overlay[1])  # Green channel
        row_images.append(gt_edge_overlay)

        # Pred edge overlay
        pred_edge_mask = torch.sigmoid(pred_edges[i, 0])  # [H, W]
        pred_edge_overlay = img_denorm.clone()
        pred_edge_overlay[0] = torch.where(pred_edge_mask > 0.5, torch.ones_like(pred_edge_mask), pred_edge_overlay[0])  # Red channel
        row_images.append(pred_edge_overlay)

        # Compute pose metrics
        try:
            pred_pose_np = pnp_pose_estimation(keypoints_3d, pred_kpts_2d, K_np)
            metrics = compute_pose_metrics(pred_pose_np, gt_pose_np)
            all_trans_errors.append(metrics["translation_error_mm"])
            all_rot_errors.append(metrics["rotation_error_deg"])
        except Exception as e:
            if debug:
                print(f"  PnP failed for sample {i}: {e}")
            all_trans_errors.append(0.0)
            all_rot_errors.append(0.0)

        # Add labels for first row, padding for others (to keep dimensions consistent)
        if i == 0:
            row_images = [add_text_to_image(img, label) for img, label in zip(row_images, labels)]
        else:
            # Add empty padding to match label height
            row_images = [add_text_to_image(img, "") for img in row_images]

        all_rows.append(torch.stack(row_images))

    # Stack all rows: [num_samples, ncols, 3, H, W]
    all_images = torch.cat(all_rows, dim=0)

    # Create grid
    grid = make_grid(all_images, nrow=ncols, padding=10, pad_value=1.0)

    # Aggregate metrics
    avg_metrics = {
        "avg_translation_error_mm": np.mean(all_trans_errors) if all_trans_errors else 0.0,
        "avg_rotation_error_deg": np.mean(all_rot_errors) if all_rot_errors else 0.0,
    }

    return grid, avg_metrics


def log_batch_visualization_wandb(
    model,
    fixed_batch: Dict[str, torch.Tensor],
    random_batch: Dict[str, torch.Tensor],
    class_type: str,
    epoch: int,
    device: torch.device
):
    """
    Log batch visualizations to wandb with pose estimation.

    Args:
        model: ContourPose model
        fixed_batch: Fixed batch (consistent across epochs)
        random_batch: Random batch (different each epoch)
        class_type: Object class (e.g., "obj1")
        epoch: Current epoch
        device: Device
    """
    # Load 3D keypoints and CAD model (already in mm)
    keypoints_3d = np.loadtxt(os.path.join(os.getcwd(), f"keypoints/{class_type}.txt"))
    cad_path = os.path.join(os.getcwd(), f"cad/{class_type}.ply")
    mesh_model = load_ply(cad_path)
    pts_3d = mesh_model["pts"]  # Already in mm

    model.eval()

    with torch.no_grad():
        # Fixed batch
        images_fixed = fixed_batch["images"].to(device)
        heatmaps_fixed = fixed_batch["heatmaps"].to(device)
        edges_fixed = fixed_batch["edges"].to(device)
        K_fixed = fixed_batch["K"].to(device)
        pose_fixed = fixed_batch["pose"].to(device)

        batch_fixed = {
            "images": images_fixed,
            "heatmaps": heatmaps_fixed,
            "edges": edges_fixed,
            "K": K_fixed,
            "pose": pose_fixed
        }

        edge_input_fixed = fixed_batch.get("edges_input")
        if edge_input_fixed is not None:
            edge_input_fixed = edge_input_fixed.to(device)
            batch_fixed["edges_input"] = edge_input_fixed
        pred_heatmaps_fixed, pred_edges_fixed = model(images_fixed, edge_input_fixed)
        grid_fixed, metrics_fixed = visualize_batch_with_pose(
            batch_fixed, pred_heatmaps_fixed, pred_edges_fixed, keypoints_3d, pts_3d, num_samples=4
        )

        # Random batch
        images_random = random_batch["images"].to(device)
        heatmaps_random = random_batch["heatmaps"].to(device)
        edges_random = random_batch["edges"].to(device)
        K_random = random_batch["K"].to(device)
        pose_random = random_batch["pose"].to(device)

        batch_random = {
            "images": images_random,
            "heatmaps": heatmaps_random,
            "edges": edges_random,
            "K": K_random,
            "pose": pose_random
        }

        edge_input_random = random_batch.get("edges_input")
        if edge_input_random is not None:
            edge_input_random = edge_input_random.to(device)
            batch_random["edges_input"] = edge_input_random
        pred_heatmaps_random, pred_edges_random = model(images_random, edge_input_random)
        grid_random, metrics_random = visualize_batch_with_pose(
            batch_random, pred_heatmaps_random, pred_edges_random, keypoints_3d, pts_3d, num_samples=4
        )

        # Log to wandb
        wandb.log({
            "viz/fixed_batch": wandb.Image(grid_fixed.cpu().numpy().transpose(1, 2, 0)),
            "viz/random_batch": wandb.Image(grid_random.cpu().numpy().transpose(1, 2, 0)),
            "viz/fixed_trans_error_mm": metrics_fixed["avg_translation_error_mm"],
            "viz/fixed_rot_error_deg": metrics_fixed["avg_rotation_error_deg"],
            "viz/random_trans_error_mm": metrics_random["avg_translation_error_mm"],
            "viz/random_rot_error_deg": metrics_random["avg_rotation_error_deg"],
        }, step=epoch)

    model.train()
