#!/usr/bin/env python3
"""
DALIDataset.py

NVIDIA DALI pipeline for ContourPose synthetic data.
Accelerates image loading and augmentations on GPU.

Prerequisites:
    - Run precompute_keypoints_2d.py first to generate 2D keypoint projections

Storage comparison (per sample, 17 keypoints @ 480x640):
    - Full heatmaps (uint8):  5.2 MB
    - 2D keypoints (float16): 68 bytes  (76,000x smaller!)

Augmentations:
    - Background mixing (SUN397 dataset)
    - Color jitter (brightness/contrast)
    - Note: Geometric augmentations (rotation/translation) are NOT applied
      because they would invalidate the pose ground truth labels.

Usage:
    from dataset.DALIDataset import get_dali_loader, generate_heatmaps_gpu

    loader = get_dali_loader(data_root, class_type, batch_size)
    for batch in loader:
        images = batch["images"]           # [B, 3, H, W]
        keypoints_2d = batch["keypoints_2d"]  # [B, N, 2]
        edges = batch["edges"]             # [B, 1, H, W]

        # Generate heatmaps on GPU (fast: ~0.3ms per batch)
        heatmaps = generate_heatmaps_gpu(keypoints_2d)  # [B, N, H, W]

Author: ContourPose
"""

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dali_math
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
import numpy as np
from pathlib import Path
import random
import pickle


# Blender camera intrinsics for synthetic renders
BLENDER_K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def generate_heatmaps_gpu(
    keypoints_2d: torch.Tensor,
    height: int = 480,
    width: int = 640,
    sigma: float = 25.0,
) -> torch.Tensor:
    """
    Generate Gaussian heatmaps on GPU from 2D keypoint coordinates.

    This is a vectorized GPU operation that generates all heatmaps for a batch
    in a single pass. Typically takes <0.5ms for a batch of 32.

    Args:
        keypoints_2d: [B, N, 2] tensor of (x, y) keypoint coordinates
        height: Image height (default: 480)
        width: Image width (default: 640)
        sigma: Gaussian standard deviation in pixels (default: 25.0)

    Returns:
        [B, N, H, W] tensor of Gaussian heatmaps normalized to [0, 1]
    """
    device = keypoints_2d.device
    B, N, _ = keypoints_2d.shape

    # Create coordinate grids [H, W]
    y_coords = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, height, 1)
    x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, width)

    # Extract keypoint coordinates [B, N, 1, 1]
    kp_x = keypoints_2d[:, :, 0].view(B, N, 1, 1)
    kp_y = keypoints_2d[:, :, 1].view(B, N, 1, 1)

    # Compute squared distances [B, N, H, W]
    dist_sq = (x_coords - kp_x) ** 2 + (y_coords - kp_y) ** 2

    # Gaussian: exp(-d^2 / (2 * sigma^2))
    heatmaps = torch.exp(-dist_sq / (2 * sigma * sigma))

    # Handle out-of-bounds keypoints (set those heatmaps to zero)
    valid_mask = (
        (kp_x >= 0) & (kp_x < width) &
        (kp_y >= 0) & (kp_y < height)
    ).float()  # [B, N, 1, 1]
    heatmaps = heatmaps * valid_mask

    return heatmaps


class ContourPoseDALIPipeline(Pipeline):
    """
    DALI pipeline for ContourPose synthetic renders.

    GPU-accelerated operations:
    - Image/mask/edge loading and decoding
    - Background mixing (mask compositing with SUN397)
    - Color jitter (brightness/contrast)
    - Normalization
    - 2D keypoint loading

    Note: Geometric augmentations (rotation/translation) are NOT applied
    because they would invalidate the pose ground truth labels. The synthetic
    dataset should already contain diverse viewpoints from rendering.
    """

    def __init__(self, data_root, class_type, batch_size, num_threads, device_id,
                 seed=42, file_indices=None, compute_edge_input=False):
        super().__init__(batch_size, num_threads, device_id, seed=seed)

        self.compute_edge_input = compute_edge_input
        self.class_type = class_type
        self.file_indices = file_indices
        self.img_height = 480
        self.img_width = 640

        # Set random seed for file shuffling
        random.seed(seed)
        np.random.seed(seed)

        # Build file lists using pathlib
        data_root = Path(data_root)
        render_dir = data_root / "train" / "renders" / class_type
        render_edge_dir = data_root / "train" / "renders" / "gtEdge" / class_type
        keypoints_2d_dir = render_dir / "keypoints_2d"
        bg_dir = data_root / "SUN2012pascalformat" / "JPEGImages"

        # Verify keypoints_2d directory exists
        if not keypoints_2d_dir.exists():
            raise FileNotFoundError(
                f"Keypoints 2D directory not found: {keypoints_2d_dir}\n"
                f"Please run: python scripts/precompute_keypoints_2d.py --class_type {class_type}"
            )

        # Get synthetic render paths
        img_files = sorted(render_dir.glob("*.jpg"))

        self.img_files = []
        self.edge_files = []
        self.keypoints_2d_files = []
        self.poses = []

        # Load keypoints to know num_keypoints
        keypoints_path = Path.cwd() / "keypoints" / f"{class_type}.txt"
        self.num_keypoints = len(np.loadtxt(keypoints_path)) if keypoints_path.exists() else 8

        for img_path in img_files:
            idx = img_path.stem
            kp_file = keypoints_2d_dir / f"{idx}.npy"

            # Skip if keypoints file doesn't exist
            if not kp_file.exists():
                continue

            self.img_files.append(str(img_path))
            self.edge_files.append(str(render_edge_dir / f"{idx}.png"))
            self.keypoints_2d_files.append(str(kp_file))

            # Load pose from pkl file
            pose_path = render_dir / f"{idx}_RT.pkl"
            with open(pose_path, "rb") as f:
                pose = pickle.load(f)["RT"]  # 3x4 matrix [R|t]
            self.poses.append(pose.astype(np.float32))

        # Preload all 2D keypoints into memory (they're tiny: ~68 bytes each)
        self.keypoints_2d_data = []
        for kp_file in self.keypoints_2d_files:
            kp = np.load(kp_file).astype(np.float32)  # [N, 2]
            self.keypoints_2d_data.append(kp)

        # Subset files if file_indices provided (for cross-validation)
        if file_indices is not None:
            self.img_files = [self.img_files[i] for i in file_indices]
            self.edge_files = [self.edge_files[i] for i in file_indices]
            self.keypoints_2d_data = [self.keypoints_2d_data[i] for i in file_indices]
            self.poses = [self.poses[i] for i in file_indices]
            print(f"[DALI] Subsetted to {len(file_indices)} samples from file_indices")

        self.num_samples = len(self.img_files)

        # Shuffle all file lists together using same random state
        if file_indices is None:
            indices = list(range(self.num_samples))
            random.shuffle(indices)

            self.img_files = [self.img_files[i] for i in indices]
            self.edge_files = [self.edge_files[i] for i in indices]
            self.keypoints_2d_data = [self.keypoints_2d_data[i] for i in indices]
            self.poses = [self.poses[i] for i in indices]
            print(f"[DALI] File lists shuffled with seed={seed}")

        # Background images (shuffle separately - we want random backgrounds)
        self.bg_files = [str(p) for p in sorted(bg_dir.glob("*.jpg"))]
        random.shuffle(self.bg_files)

        print(f"[DALI] Loaded {self.num_samples} synthetic renders for {class_type}")
        print(f"[DALI] Loaded {len(self.bg_files)} background images")
        print(f"[DALI] Using 2D keypoints from {keypoints_2d_dir}")
        print(f"[DALI] Keypoints: {self.num_keypoints}")

    def get_keypoints_2d(self, sample_info):
        """
        External source callback for 2D keypoints.
        Returns [N, 2] keypoint array for the current sample.
        """
        idx = sample_info.idx_in_epoch % self.num_samples
        return self.keypoints_2d_data[idx].copy()

    def get_camera_intrinsics(self, sample_info):
        """
        External source callback for camera intrinsics.
        Returns 3x3 K matrix.
        """
        return BLENDER_K.copy()

    def get_pose(self, sample_info):
        """
        External source callback for object pose.
        Returns 3x4 pose matrix [R|t] for the current sample.
        """
        idx = sample_info.idx_in_epoch % self.num_samples
        return self.poses[idx].copy()

    def define_graph(self):
        """Define DALI computation graph"""

        # === External sources ===

        # 2D keypoints [N, 2]
        keypoints_2d = fn.external_source(
            source=self.get_keypoints_2d,
            batch=False,
            device="cpu"
        )

        # Camera intrinsics (3x3 matrix)
        K = fn.external_source(
            source=self.get_camera_intrinsics,
            batch=False,
            device="cpu"
        )

        # Object pose (3x4 matrix [R|t])
        pose = fn.external_source(
            source=self.get_pose,
            batch=False,
            device="cpu"
        )

        # === File reading ===

        img_encoded, _ = fn.readers.file(
            files=self.img_files,
            random_shuffle=False,
            name="img_reader"
        )
        img = fn.decoders.image(img_encoded, device="mixed", output_type=types.RGB)

        # Generate mask from rendered image (object on black background)
        img_gray = fn.color_space_conversion(img, image_type=types.RGB, output_type=types.GRAY)
        mask = (img_gray > 5) * 255

        # Read edges
        edge_encoded, _ = fn.readers.file(
            files=self.edge_files,
            random_shuffle=False,
            name="edge_reader"
        )
        edge = fn.decoders.image(edge_encoded, device="mixed", output_type=types.GRAY)

        # Read background images
        bg_encoded, _ = fn.readers.file(
            files=self.bg_files,
            random_shuffle=False,
            name="bg_reader"
        )
        bg = fn.decoders.image(bg_encoded, device="mixed", output_type=types.RGB)
        bg = fn.resize(bg, size=(480, 640))

        # === Background mixing ===

        mask_float = fn.cast(mask, dtype=types.FLOAT) / 255.0
        mask_rgb = fn.cat(mask_float, mask_float, mask_float, axis=2)

        img_float = fn.cast(img, dtype=types.FLOAT)
        bg_float = fn.cast(bg, dtype=types.FLOAT)

        composited = mask_rgb * img_float + (1.0 - mask_rgb) * bg_float
        composited = fn.cast(composited, dtype=types.UINT8)

        # === Color jitter ===

        composited = fn.brightness_contrast(
            composited,
            brightness=fn.random.uniform(range=(0.8, 1.2)),
            contrast=fn.random.uniform(range=(0.8, 1.2))
        )

        # === Edge input (Laplacian on augmented RGB, before normalization) ===
        if self.compute_edge_input:
            img_gray_for_edge = fn.color_space_conversion(composited, image_type=types.RGB, output_type=types.GRAY)
            img_gray_f = fn.cast(img_gray_for_edge, dtype=types.FLOAT) / 255.0
            edge_input = fn.laplacian(img_gray_f, window_size=3)
            edge_input = dali_math.abs(edge_input)
            edge_input = dali_math.min(edge_input, 1.0)
            edge_input = fn.transpose(edge_input, perm=[2, 0, 1])  # HW1 → 1HW

        # === Normalization ===

        composited = fn.cast(composited, dtype=types.FLOAT) / 255.0
        composited = (composited - np.array([0.419, 0.427, 0.424], dtype=np.float32)) / np.array([0.184, 0.206, 0.197], dtype=np.float32)
        composited = fn.transpose(composited, perm=[2, 0, 1])

        edge = fn.cast(edge, dtype=types.FLOAT) / 255.0
        edge = fn.transpose(edge, perm=[2, 0, 1])

        # Ensure proper types
        K = fn.cast(K, dtype=types.FLOAT)
        pose = fn.cast(pose, dtype=types.FLOAT)
        keypoints_2d = fn.cast(keypoints_2d, dtype=types.FLOAT)

        if self.compute_edge_input:
            return composited, keypoints_2d, edge, K, pose, edge_input
        return composited, keypoints_2d, edge, K, pose

    def __len__(self):
        return self.num_samples


def get_dali_loader(data_root, class_type, batch_size, num_threads=4, device_id=0,
                    seed=42, file_indices=None, compute_edge_input=False):
    """
    Create DALI data loader for ContourPose.

    Args:
        data_root: Path to data directory
        class_type: Object class (e.g., "obj1")
        batch_size: Batch size
        num_threads: CPU threads for DALI (default: 4)
        device_id: GPU device ID (default: 0)
        seed: Random seed (default: 42)
        file_indices: Optional list of file indices to use (for cross-validation split)

    Returns:
        DALIGenericIterator yielding dicts with keys:
        - "images": [B, 3, H, W] normalized RGB
        - "keypoints_2d": [B, N, 2] 2D keypoint coordinates
        - "edges": [B, 1, H, W]
        - "K": [B, 3, 3] camera intrinsics
        - "pose": [B, 3, 4] object pose [R|t]

    Example:
        loader = get_dali_loader(data_root, "obj1", batch_size=32)
        for batch in loader:
            images = batch["images"]
            keypoints_2d = batch["keypoints_2d"]

            # Generate heatmaps on GPU
            heatmaps = generate_heatmaps_gpu(keypoints_2d)
    """

    pipeline = ContourPoseDALIPipeline(
        data_root=data_root,
        class_type=class_type,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        file_indices=file_indices,
        compute_edge_input=compute_edge_input,
    )

    pipeline.build()

    output_map = ["images", "keypoints_2d", "edges", "K", "pose"]
    if compute_edge_input:
        output_map.append("edges_input")

    loader = DALIGenericIterator(
        pipelines=[pipeline],
        output_map=output_map,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        reader_name="img_reader"
    )

    return loader
