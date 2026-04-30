#!/usr/bin/env python3
"""
BOPDALIDataset.py

NVIDIA DALI pipeline for ContourPose synthetic datasets.
Accelerates image loading and augmentations on GPU.

Dataset Structure (flat format):
    dataset_dir/
    ├── rgb/000000.png, 000001.png, ...
    ├── mask/000000.png, ...
    ├── edges/000000.png, ...
    ├── keypoints/obj1.txt
    ├── scene_camera.json
    └── scene_gt.json

Usage:
    from dataset.BOPDALIDataset import get_bop_dali_loader
    loader = get_bop_dali_loader(data_dir, obj_id, keypoints_dir, batch_size)

Returns:
    Dict with keys: "images", "heatmaps", "edges", "K", "pose"
    - images: [B, 3, H, W] normalized RGB
    - heatmaps: [B, num_keypoints, H, W] Gaussian probability maps
    - edges: [B, 1, H, W] ground truth contours
    - K: [B, 3, 3] camera intrinsics matrix
    - pose: [B, 3, 4] object pose (R|t)
"""

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as dali_math
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import numpy as np
from pathlib import Path
import json
import random
import os


class BOPDALIPipeline(Pipeline):
    """
    DALI pipeline for ContourPose synthetic data.

    GPU-accelerated operations:
    - Image/mask/edge loading and decoding
    - Background mixing (optional)
    - Color jitter (brightness/contrast)
    - Normalization
    - Heatmap generation (CPU) or loading (if precomputed)

    Args:
        data_dir: Dataset directory containing rgb/, mask/, edges/, scene_gt.json, etc.
        obj_id: Target object ID (1-indexed)
        keypoints_path: Path to keypoints file for this object
        batch_size: Batch size
        num_threads: Number of CPU threads
        device_id: GPU device ID
        seed: Random seed
        img_size: Output image size (H, W)
        heatmap_dir: Directory with precomputed heatmaps (optional)
        background_dir: Directory with background images (optional)
        file_indices: Optional list of sample indices to use
    """

    def __init__(
        self,
        data_dir,
        obj_id,
        keypoints_path,
        batch_size,
        num_threads,
        device_id,
        seed=42,
        img_size=(480, 640),
        heatmap_dir=None,
        background_dir=None,
        file_indices=None,
        compute_edge_input=False,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=seed)

        self.compute_edge_input = compute_edge_input
        self.data_dir = Path(data_dir)
        self.obj_id = obj_id
        self.img_size = img_size  # (H, W)
        self.heatmap_dir = Path(heatmap_dir) if heatmap_dir else None
        self.use_precomputed_heatmaps = heatmap_dir is not None and Path(heatmap_dir).exists()

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Load 3D keypoints
        self.keypoints_3d = np.loadtxt(keypoints_path).astype(np.float32) * 1000  # m → mm (consistent with BOP cam_t_m2c)
        self.num_keypoints = len(self.keypoints_3d)
        print(f"[BOP DALI] Loaded {self.num_keypoints} keypoints from {keypoints_path}")

        # Build sample index from BOP structure
        self.samples = self._build_sample_index()

        # Subset if file_indices provided
        if file_indices is not None:
            self.samples = [self.samples[i] for i in file_indices]
            print(f"[BOP DALI] Subsetted to {len(file_indices)} samples")

        self.num_samples = len(self.samples)

        # Shuffle samples
        if file_indices is None:
            random.shuffle(self.samples)
            print(f"[BOP DALI] Samples shuffled with seed={seed}")

        # Build file lists from samples
        self.img_files = [str(s['rgb_path']) for s in self.samples]
        self.mask_files = [str(s['mask_path']) if s['mask_path'] else "" for s in self.samples]
        self.edge_files = [str(s['edge_path']) if s.get('edge_path') else "" for s in self.samples]
        self.cam_K_list = [s['cam_K'] for s in self.samples]
        self.pose_list = [s['pose'] for s in self.samples]

        # Check if we have precomputed edges
        self.use_precomputed_edges = any(s.get('edge_path') for s in self.samples)
        if self.use_precomputed_edges:
            print(f"[BOP DALI] Using precomputed edge images from edges/ directory")

        # Precomputed heatmap files (if available)
        # Support both PNG (GPU-accelerated) and NPY (legacy) formats
        self.heatmap_format = None
        if self.use_precomputed_heatmaps:
            self.heatmap_files = []

            # Check format by looking for first file
            first_sample = self.samples[0]
            scene_output_dir = self.heatmap_dir / first_sample['scene_id']

            # Try PNG first (GPU-efficient), then NPY (legacy)
            png_path = scene_output_dir / f"{first_sample['frame_id']:06d}.png"
            npy_path = scene_output_dir / f"{first_sample['frame_id']:06d}.npy"

            if png_path.exists():
                self.heatmap_format = "png"
                ext = ".png"
            elif npy_path.exists():
                self.heatmap_format = "npy"
                ext = ".npy"
            else:
                # Try old naming convention
                old_name = f"{first_sample['scene_id']}_{first_sample['frame_id']:06d}_{first_sample['obj_instance']:06d}"
                if (self.heatmap_dir / f"{old_name}.png").exists():
                    self.heatmap_format = "png_flat"
                    ext = ".png"
                elif (self.heatmap_dir / f"{old_name}.npy").exists():
                    self.heatmap_format = "npy_flat"
                    ext = ".npy"
                else:
                    print(f"[BOP DALI] Warning: No precomputed heatmaps found, will generate on CPU")
                    self.use_precomputed_heatmaps = False

            if self.use_precomputed_heatmaps:
                for s in self.samples:
                    if self.heatmap_format in ["png", "npy"]:
                        # New format: bop_root/heatmaps_split/obj_id/scene_id/frame.ext
                        heatmap_path = self.heatmap_dir / s['scene_id'] / f"{s['frame_id']:06d}{ext}"
                        if len([gt for gt in self.samples if gt['scene_id'] == s['scene_id'] and gt['frame_id'] == s['frame_id']]) > 1:
                            # Multiple instances - use instance index
                            heatmap_path = self.heatmap_dir / s['scene_id'] / f"{s['frame_id']:06d}_{s['obj_instance']:02d}{ext}"
                    else:
                        # Old flat format
                        heatmap_name = f"{s['scene_id']}_{s['frame_id']:06d}_{s['obj_instance']:06d}{ext}"
                        heatmap_path = self.heatmap_dir / heatmap_name

                    self.heatmap_files.append(str(heatmap_path))

                print(f"[BOP DALI] Using precomputed heatmaps ({self.heatmap_format.upper()}) from {heatmap_dir}")

        # Background images (optional)
        self.bg_files = []
        if background_dir and Path(background_dir).exists():
            bg_dir = Path(background_dir)
            self.bg_files = [str(p) for p in sorted(bg_dir.glob("*.jpg"))]
            self.bg_files.extend([str(p) for p in sorted(bg_dir.glob("*.png"))])
            random.shuffle(self.bg_files)
            print(f"[BOP DALI] Loaded {len(self.bg_files)} background images")

        print(f"[DALI] Pipeline initialized:")
        print(f"  Data dir: {self.data_dir}")
        print(f"  Object ID: {self.obj_id}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Image size: {self.img_size}")

    def _build_sample_index(self):
        """
        Build index of samples from dataset directory.
        Parses scene_gt.json and scene_camera.json.

        Expected format (flat):
           data_dir/
           ├── rgb/000000.png
           ├── mask/000000.png
           ├── edges/000000.png
           ├── scene_gt.json
           └── scene_camera.json
        """
        samples = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Flat format: scene_gt.json directly in data_dir
        gt_path = self.data_dir / "scene_gt.json"
        camera_path = self.data_dir / "scene_camera.json"

        if not gt_path.exists():
            raise FileNotFoundError(f"scene_gt.json not found in {self.data_dir}")
        if not camera_path.exists():
            raise FileNotFoundError(f"scene_camera.json not found in {self.data_dir}")

        scene_dirs = [(self.data_dir, "000000")]

        for scene_dir, scene_id in scene_dirs:
            gt_path = scene_dir / "scene_gt.json"
            camera_path = scene_dir / "scene_camera.json"

            if not gt_path.exists() or not camera_path.exists():
                continue

            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            with open(camera_path, 'r') as f:
                scene_camera = json.load(f)

            for frame_id_str, gt_list in scene_gt.items():
                frame_id = int(frame_id_str)

                for inst_idx, gt in enumerate(gt_list):
                    if gt['obj_id'] != self.obj_id:
                        continue

                    # Camera intrinsics
                    cam_data = scene_camera[frame_id_str]
                    cam_K = np.array(cam_data['cam_K'], dtype=np.float32).reshape(3, 3)

                    # Pose: R and t
                    R = np.array(gt['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
                    t = np.array(gt['cam_t_m2c'], dtype=np.float32).reshape(3, 1)
                    pose = np.hstack([R, t])  # 3x4 matrix

                    # File paths - try multiple naming conventions
                    rgb_path = scene_dir / "rgb" / f"{frame_id:06d}.png"
                    if not rgb_path.exists():
                        rgb_path = scene_dir / "rgb" / f"{frame_id:06d}.jpg"

                    # Mask path - try different conventions:
                    # 1. Simple: mask/000000.png (flat format)
                    # 2. With instance: mask/000000_000000.png (BOP format)
                    # 3. mask_visib directory (standard BOP)
                    mask_path = scene_dir / "mask" / f"{frame_id:06d}.png"
                    if not mask_path.exists():
                        mask_path = scene_dir / "mask" / f"{frame_id:06d}_{inst_idx:06d}.png"
                    if not mask_path.exists():
                        mask_path = scene_dir / "mask_visib" / f"{frame_id:06d}_{inst_idx:06d}.png"
                    if not mask_path.exists():
                        mask_path = None

                    # Edge path: try simple name, then instance-indexed name
                    edge_path = scene_dir / "edges" / f"{frame_id:06d}.png"
                    if not edge_path.exists():
                        edge_path = scene_dir / "edges" / f"{frame_id:06d}_{inst_idx:06d}.png"
                    if not edge_path.exists():
                        edge_path = None

                    if not rgb_path.exists():
                        continue

                    samples.append({
                        'scene_id': scene_id,
                        'frame_id': frame_id,
                        'obj_instance': inst_idx,
                        'rgb_path': rgb_path,
                        'mask_path': mask_path,
                        'edge_path': edge_path,
                        'cam_K': cam_K,
                        'pose': pose,
                    })

        return samples

    def get_camera_intrinsics(self, sample_info):
        """External source callback for camera intrinsics."""
        idx = sample_info.idx_in_epoch % self.num_samples
        return self.cam_K_list[idx].copy()

    def get_pose(self, sample_info):
        """External source callback for pose matrix."""
        idx = sample_info.idx_in_epoch % self.num_samples
        return self.pose_list[idx].copy()

    def get_heatmap(self, sample_info):
        """
        External source callback for heatmaps.
        If precomputed heatmaps exist, load them. Otherwise, generate on CPU.

        Note: For PNG format, this is used as fallback. The define_graph method
        uses fn.readers.file + fn.decoders.image for GPU-accelerated loading.
        """
        idx = sample_info.idx_in_epoch % self.num_samples

        if self.use_precomputed_heatmaps:
            heatmap_path = self.heatmap_files[idx]

            if self.heatmap_format in ["npy", "npy_flat"]:
                # Load NPY format
                heatmap = np.load(heatmap_path)
                return (heatmap / 255.0).astype(np.float32)

            elif self.heatmap_format in ["png", "png_flat"]:
                # Load PNG format (stacked grayscale)
                import cv2
                stacked = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
                if stacked is None:
                    print(f"[BOP DALI] Warning: Failed to load heatmap {heatmap_path}")
                    return self._generate_heatmap_cpu(idx)

                H, W = self.img_size
                # Reshape from (H * num_kp, W) to (num_kp, H, W)
                heatmap = stacked.reshape(self.num_keypoints, H, W)
                return (heatmap / 255.0).astype(np.float32)

        # Generate heatmap on CPU from pose and keypoints
        return self._generate_heatmap_cpu(idx)

    def _generate_heatmap_cpu(self, idx):
        """Generate heatmap on CPU from pose and keypoints."""
        sample = self.samples[idx]
        cam_K = sample['cam_K']
        pose = sample['pose']

        # Project 3D keypoints to 2D
        R, t = pose[:, :3], pose[:, 3:]
        pts_cam = (R @ self.keypoints_3d.T).T + t.flatten()
        pts_2d = (cam_K @ pts_cam.T).T
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]

        # Generate Gaussian heatmaps
        H, W = self.img_size
        sigma = 5.0
        heatmaps = np.zeros((self.num_keypoints, H, W), dtype=np.float32)

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        for i, (x, y) in enumerate(pts_2d):
            if 0 <= x < W and 0 <= y < H:
                heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

        return heatmaps

    def define_graph(self):
        """Define DALI computation graph."""

        # === External sources for metadata ===
        K = fn.external_source(
            source=self.get_camera_intrinsics,
            batch=False,
            device="cpu"
        )

        pose = fn.external_source(
            source=self.get_pose,
            batch=False,
            device="cpu"
        )

        # Heatmaps
        # Use GPU-accelerated loading for PNG format, fallback to external source for NPY/CPU
        if self.use_precomputed_heatmaps and self.heatmap_format in ["png", "png_flat"]:
            # GPU-accelerated PNG loading
            heatmap_encoded, _ = fn.readers.file(
                files=self.heatmap_files,
                random_shuffle=False,
                name="heatmap_reader"
            )
            # Decode as grayscale on GPU
            heatmap_stacked = fn.decoders.image(
                heatmap_encoded,
                device="mixed",
                output_type=types.GRAY
            )
            # Normalize to [0, 1]
            heatmap_stacked = fn.cast(heatmap_stacked, dtype=types.FLOAT) / 255.0

            # Reshape from (H * num_kp, W, 1) to (num_kp, H, W)
            # First squeeze the channel dimension
            heatmap_stacked = fn.squeeze(heatmap_stacked, axes=[2])
            # Then reshape: (H * num_kp, W) -> (num_kp, H, W)
            H, W = self.img_size
            heatmap = fn.reshape(heatmap_stacked, shape=[self.num_keypoints, H, W])
        else:
            # Fallback: external source (NPY format or CPU generation)
            heatmap = fn.external_source(
                source=self.get_heatmap,
                batch=False,
                device="cpu"
            )
            heatmap = heatmap.gpu()

        # === File reading ===

        # Read RGB images
        img_encoded, _ = fn.readers.file(
            files=self.img_files,
            random_shuffle=False,
            name="img_reader"
        )
        img = fn.decoders.image(img_encoded, device="mixed", output_type=types.RGB)

        # Resize image if needed
        img = fn.resize(img, size=self.img_size)

        # Read masks (for background mixing)
        has_masks = any(self.mask_files)
        if has_masks:
            # Filter to only valid mask files, use placeholder for missing
            valid_mask_files = []
            for mf in self.mask_files:
                if mf and os.path.exists(mf):
                    valid_mask_files.append(mf)
                else:
                    # Use first valid mask as placeholder (will be zeroed out)
                    valid_mask_files.append(self.mask_files[0] if self.mask_files[0] else self.img_files[0])

            mask_encoded, _ = fn.readers.file(
                files=valid_mask_files,
                random_shuffle=False,
                name="mask_reader"
            )
            mask = fn.decoders.image(mask_encoded, device="mixed", output_type=types.GRAY)
            mask = fn.resize(mask, size=self.img_size, interp_type=types.INTERP_NN)
            mask_float = fn.cast(mask, dtype=types.FLOAT) / 255.0
        else:
            # No masks - create dummy mask from non-black pixels
            img_gray = fn.color_space_conversion(img, image_type=types.RGB, output_type=types.GRAY)
            mask_float = fn.cast(img_gray > 5, dtype=types.FLOAT)

        # Load edges - prefer precomputed edges, fallback to Laplacian from mask
        if self.use_precomputed_edges:
            # Use precomputed edge images from edges/ directory
            valid_edge_files = []
            for ef in self.edge_files:
                if ef and os.path.exists(ef):
                    valid_edge_files.append(ef)
                else:
                    # Use first valid edge as placeholder
                    valid_edge_files.append(self.edge_files[0] if self.edge_files[0] else self.img_files[0])

            edge_encoded, _ = fn.readers.file(
                files=valid_edge_files,
                random_shuffle=False,
                name="edge_reader"
            )
            edge = fn.decoders.image(edge_encoded, device="mixed", output_type=types.GRAY)
            edge = fn.resize(edge, size=self.img_size, interp_type=types.INTERP_NN)
            edge = fn.cast(edge, dtype=types.FLOAT) / 255.0
        elif has_masks:
            # Generate edge from mask boundary using Laplacian
            mask_f = fn.cast(mask, dtype=types.FLOAT) / 255.0
            edge = fn.laplacian(mask_f, window_size=3)
            edge = dali_math.abs(edge)
            edge = fn.cast(edge > 0.1, dtype=types.FLOAT)
        else:
            # Fallback: Laplacian on grayscale image
            img_gray = fn.color_space_conversion(img, image_type=types.RGB, output_type=types.GRAY)
            img_gray_f = fn.cast(img_gray, dtype=types.FLOAT) / 255.0
            edge = fn.laplacian(img_gray_f, window_size=3)
            edge = dali_math.abs(edge)
            edge = fn.cast(edge > 0.1, dtype=types.FLOAT)

        # === Background mixing (optional) ===
        if self.bg_files:
            bg_encoded, _ = fn.readers.file(
                files=self.bg_files,
                random_shuffle=True,
                name="bg_reader"
            )
            bg = fn.decoders.image(bg_encoded, device="mixed", output_type=types.RGB)
            bg = fn.resize(bg, size=self.img_size)

            # Composite: bg * (1 - mask) + img * mask
            # mask_float is always defined above — either from mask files (has_masks=True)
            # or from thresholded non-black pixels (has_masks=False, PBR renders on black bg)
            mask_rgb = fn.cat(mask_float, mask_float, mask_float, axis=2)
            img_float = fn.cast(img, dtype=types.FLOAT)
            bg_float = fn.cast(bg, dtype=types.FLOAT)
            composited = mask_rgb * img_float + (1.0 - mask_rgb) * bg_float
            img = fn.cast(composited, dtype=types.UINT8)

        # === Color jitter ===
        # HSV jitter applied with 60% probability
        do_hsv = fn.cast(fn.random.coin_flip(probability=0.6), dtype=types.FLOAT)
        img_orig_f = fn.cast(img, dtype=types.FLOAT)
        img_hsv_f = fn.hsv(
            img_orig_f / 255.0,
            hue=fn.random.uniform(range=(-15.0, 15.0)),
            saturation=fn.random.uniform(range=(0.7, 1.3)),
        ) * 255.0
        img = fn.cast(do_hsv * img_hsv_f + (1.0 - do_hsv) * img_orig_f, dtype=types.UINT8)

        # Brightness/contrast (always applied, range spans identity at 1.0)
        img = fn.brightness_contrast(
            img,
            brightness=fn.random.uniform(range=(0.8, 1.2)),
            contrast=fn.random.uniform(range=(0.8, 1.2)),
        )

        # Gaussian blur applied with 40% probability
        do_blur = fn.cast(fn.random.coin_flip(probability=0.4), dtype=types.FLOAT)
        blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        img_f = fn.cast(img, dtype=types.FLOAT)
        blurred_f = fn.cast(blurred, dtype=types.FLOAT)
        img = fn.cast(do_blur * blurred_f + (1.0 - do_blur) * img_f, dtype=types.UINT8)

        # === Edge input (Laplacian on augmented RGB, before normalization) ===
        if self.compute_edge_input:
            img_gray_for_edge = fn.color_space_conversion(img, image_type=types.RGB, output_type=types.GRAY)
            img_gray_f = fn.cast(img_gray_for_edge, dtype=types.FLOAT) / 255.0
            edge_input = fn.laplacian(img_gray_f, window_size=3)
            edge_input = dali_math.abs(edge_input)
            edge_input = dali_math.min(edge_input, 1.0)
            edge_input = fn.transpose(edge_input, perm=[2, 0, 1])  # HW1 → 1HW

        # === Normalization ===
        img = fn.cast(img, dtype=types.FLOAT) / 255.0

        # Normalize with ImageNet-like stats (same as original DALIDataset)
        mean = np.array([0.419, 0.427, 0.424], dtype=np.float32)
        std = np.array([0.184, 0.206, 0.197], dtype=np.float32)
        img = (img - mean) / std

        # Transpose to CHW
        img = fn.transpose(img, perm=[2, 0, 1])

        # Edge: ensure float and CHW format [1, H, W]
        edge = fn.cast(edge, dtype=types.FLOAT)
        # Edge from decoder is (H, W, 1), transpose to (1, H, W)
        edge = fn.transpose(edge, perm=[2, 0, 1])

        # Ensure types
        heatmap = fn.cast(heatmap, dtype=types.FLOAT)
        K = fn.cast(K, dtype=types.FLOAT)
        pose = fn.cast(pose, dtype=types.FLOAT)

        if self.compute_edge_input:
            return img, heatmap, edge, K, pose, edge_input
        return img, heatmap, edge, K, pose

    def __len__(self):
        return self.num_samples


def get_bop_dali_loader(
    data_dir,
    obj_id,
    keypoints_dir="keypoints",
    batch_size=8,
    num_threads=4,
    device_id=0,
    seed=42,
    img_size=(480, 640),
    heatmap_dir=None,
    background_dir=None,
    file_indices=None,
    compute_edge_input=False,
):
    """
    Create DALI data loader for ContourPose synthetic data.

    Args:
        data_dir: Path to dataset directory (contains rgb/, mask/, edges/, scene_gt.json, etc.)
        obj_id: Target object ID (1-indexed)
        keypoints_dir: Directory containing keypoint files
        batch_size: Batch size
        num_threads: CPU threads for DALI
        device_id: GPU device ID
        seed: Random seed
        img_size: Output image size (H, W)
        heatmap_dir: Directory with precomputed heatmaps (optional)
        background_dir: Directory with background images (optional)
        file_indices: Optional list of sample indices for subset

    Returns:
        DALIGenericIterator yielding dicts with keys:
        - "images": [B, 3, H, W] normalized RGB
        - "heatmaps": [B, num_keypoints, H, W]
        - "edges": [B, 1, H, W]
        - "K": [B, 3, 3] camera intrinsics
        - "pose": [B, 3, 4] object pose [R|t]
    """
    # Find keypoints file (resolve relative to data_dir)
    keypoints_path = None
    data_path = Path(data_dir)

    # Handle both relative and absolute keypoints_dir paths
    kp_dir = Path(keypoints_dir)
    if kp_dir.is_absolute() or kp_dir.exists():
        base_dir = kp_dir
    else:
        base_dir = data_path / keypoints_dir

    candidates = [
        base_dir / f"obj_{obj_id:06d}.txt",
        base_dir / f"obj{obj_id}.txt",
        base_dir / f"{obj_id}.txt",
    ]

    # Also try finding any .txt file if there's only one (custom naming like gear.txt)
    if base_dir.exists():
        txt_files = list(base_dir.glob("*.txt"))
        if len(txt_files) == 1:
            candidates.insert(0, txt_files[0])

    for candidate in candidates:
        if candidate.exists():
            keypoints_path = candidate
            break

    if keypoints_path is None:
        raise FileNotFoundError(
            f"Keypoints file not found for obj_id={obj_id}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    pipeline = BOPDALIPipeline(
        data_dir=data_dir,
        obj_id=obj_id,
        keypoints_path=str(keypoints_path),
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        img_size=img_size,
        heatmap_dir=heatmap_dir,
        background_dir=background_dir,
        file_indices=file_indices,
        compute_edge_input=compute_edge_input,
    )

    pipeline.build()

    output_map = ["images", "heatmaps", "edges", "K", "pose"]
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
