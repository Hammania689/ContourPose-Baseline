#!/usr/bin/env python3
"""
BOPTestDataset.py

NVIDIA DALI pipeline for evaluating ContourPose on BOP-format test data.
Handles multi-scene, multi-sensor, multi-instance structure from
converted ROBI/TROBI datasets.

Extends BOPDALIPipeline with:
- Multi-scene directory traversal (test/000001/, test/000002/, ...)
- Per-sensor GT files (scene_gt_{sensor}.json, scene_camera_{sensor}.json)
- Multi-instance support (same object class, multiple poses per image)
- No data augmentation (no bg mixing, color jitter, shuffling)
- Metadata tracking for BOP CSV output

Dataset Structure (BOP multi-sensor):
    bop_root/
    ├── test/
    │   ├── 000001/
    │   │   ├── rgb_{sensor}/000000.png
    │   │   ├── mask_visib_{sensor}/000000_000000.png
    │   │   ├── scene_gt_{sensor}.json
    │   │   └── scene_camera_{sensor}.json
    │   └── 000002/
    │       └── ...
    ├── models/
    │   └── models_info.json
    └── keypoints/
        └── obj_000001.txt

Usage:
    from dataset.BOPTestDataset import get_bop_test_dali_loader
    loader, metadata = get_bop_test_dali_loader(
        bop_root="data/ROBI_BOP/Gear_BOP",
        obj_id=1,
        sensor="ensenso",
        keypoints_dir="keypoints",
    )
"""

import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import numpy as np
from pathlib import Path
import json
import os


class BOPTestDALIPipeline(Pipeline):
    """
    DALI pipeline for BOP test evaluation.

    No augmentation — deterministic inference only.
    Parses multi-scene, multi-sensor BOP directory structure.
    """

    def __init__(
        self,
        bop_root,
        obj_id,
        keypoints_path,
        sensor="ensenso",
        scene_ids=None,
        batch_size=1,
        num_threads=4,
        device_id=0,
        img_size=(480, 640),
        use_masks=True,
        split="test",
        compute_edge_input=False,
    ):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.bop_root = Path(bop_root)
        self.obj_id = obj_id
        self.sensor = sensor
        self.img_size = img_size
        self.use_masks = use_masks
        self.split = split
        self.compute_edge_input = compute_edge_input

        # Load 3D keypoints — DALI path is mm-native (matches BOP cam_t_m2c).
        # The keypoints file must already be in millimetres; see
        # scripts/convert_keypoints_to_mm.py for the {obj}_mm.txt convention.
        self.keypoints_3d = np.loadtxt(keypoints_path).astype(np.float32)
        assert self.keypoints_3d.ndim == 2 and self.keypoints_3d.shape[1] == 3, (
            f"[BOPTest DALI] Expected (N, 3) keypoints, got {self.keypoints_3d.shape} "
            f"from {keypoints_path}"
        )
        assert np.max(np.abs(self.keypoints_3d)) > 1.0, (
            f"[BOPTest DALI] Keypoints in {keypoints_path} look like metres "
            f"(max |xyz| = {np.max(np.abs(self.keypoints_3d)):.4f}); expected mm. "
            f"Regenerate via scripts/convert_keypoints_to_mm.py."
        )
        self.num_keypoints = len(self.keypoints_3d)
        print(f"[BOPTest DALI] Loaded {self.num_keypoints} keypoints from {keypoints_path}")

        # Build sample index from multi-scene BOP structure
        self.samples = self._build_sample_index(scene_ids)
        self.num_samples = len(self.samples)

        if self.num_samples == 0:
            raise ValueError(
                f"No samples found for obj_id={obj_id}, sensor={sensor} "
                f"in {self.bop_root / self.split}"
            )

        # Build file lists (deterministic order, no shuffle)
        self.img_files = [str(s['rgb_path']) for s in self.samples]
        self.cam_K_list = [s['cam_K'] for s in self.samples]
        self.pose_list = [s['pose'] for s in self.samples]

        # Mask files (for instance isolation)
        if self.use_masks:
            self.mask_files = []
            for s in self.samples:
                mp = s.get('mask_path')
                if mp and Path(mp).exists():
                    self.mask_files.append(str(mp))
                else:
                    # Placeholder — will be handled as all-ones mask
                    self.mask_files.append(self.img_files[0])
            self.has_valid_masks = any(
                s.get('mask_path') and Path(s['mask_path']).exists()
                for s in self.samples
            )
        else:
            self.mask_files = []
            self.has_valid_masks = False

        print(f"[BOPTest DALI] Pipeline initialized:")
        print(f"  BOP root: {self.bop_root}")
        print(f"  Split: {self.split}, Sensor: {self.sensor}")
        print(f"  Object ID: {self.obj_id}")
        print(f"  Samples: {self.num_samples}")
        print(f"  Image size: {self.img_size}")
        print(f"  Use masks: {self.use_masks} (valid: {self.has_valid_masks})")

    def _sfx(self, base):
        """Return base_{sensor} when sensor is set, else base (standard single-camera BOP)."""
        return f"{base}_{self.sensor}" if self.sensor else base

    def _build_sample_index(self, scene_ids):
        """
        Parse multi-scene BOP test directory.

        Supports two path conventions automatically:
          sensor set  → rgb_{sensor}/, mask_visib_{sensor}/, scene_gt_{sensor}.json
          sensor None → rgb/, mask_visib/, scene_gt.json  (standard single-camera BOP)

        Prefers pre-cropped patches when available, falls back to full-res images.
        """
        samples = []
        split_dir = self.bop_root / self.split

        # Fall back to bop_root if split directory doesn't exist
        # (ROBI datasets have scenes directly under bop_root, no train/test split)
        if not split_dir.exists():
            split_dir = self.bop_root
            if not split_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {split_dir}")

        # Discover scene directories
        if scene_ids is not None:
            scene_dirs = [split_dir / sid for sid in scene_ids]
        else:
            scene_dirs = sorted([
                d for d in split_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ])

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name

            # Try patch-based loading first
            patches_dir = scene_dir / self._sfx("patches")
            patch_meta_path = scene_dir / f"{self._sfx('patch_meta')}.json"

            if patches_dir.exists() and patch_meta_path.exists():
                samples.extend(self._load_patch_samples(scene_dir, scene_id))
                continue

            # Fall back to full-resolution loading
            gt_path = scene_dir / f"{self._sfx('scene_gt')}.json"
            cam_path = scene_dir / f"{self._sfx('scene_camera')}.json"

            if not gt_path.exists():
                print(f"[BOPTest DALI] Warning: {gt_path.name} not found in {scene_dir}")
                continue
            if not cam_path.exists():
                print(f"[BOPTest DALI] Warning: {cam_path.name} not found in {scene_dir}")
                continue

            with open(gt_path, 'r') as f:
                scene_gt = json.load(f)
            with open(cam_path, 'r') as f:
                scene_camera = json.load(f)

            # Load visibility info if available
            gt_info_path = scene_dir / f"{self._sfx('scene_gt_info')}.json"
            scene_gt_info = None
            if gt_info_path.exists():
                with open(gt_info_path, 'r') as f:
                    scene_gt_info = json.load(f)

            for frame_id_str, gt_list in scene_gt.items():
                frame_id = int(frame_id_str)

                for inst_idx, gt in enumerate(gt_list):
                    if gt['obj_id'] != self.obj_id:
                        continue

                    # Camera intrinsics
                    cam_data = scene_camera[frame_id_str]
                    cam_K = np.array(cam_data['cam_K'], dtype=np.float32).reshape(3, 3)

                    # Pose
                    R = np.array(gt['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
                    t = np.array(gt['cam_t_m2c'], dtype=np.float32).reshape(3, 1)
                    pose = np.hstack([R, t])  # 3x4

                    # RGB path (rgb_{sensor}/ or rgb/)
                    rgb_dir = scene_dir / self._sfx("rgb")
                    rgb_path = rgb_dir / f"{frame_id:06d}.png"
                    if not rgb_path.exists():
                        rgb_path = rgb_dir / f"{frame_id:06d}.jpg"
                    if not rgb_path.exists():
                        continue

                    # Mask path: mask_visib_{sensor}/ → mask_visib/ → mask_{sensor}/ → mask/
                    mask_path = None
                    if self.use_masks:
                        for mask_dir_name in [self._sfx("mask_visib"), self._sfx("mask")]:
                            for fmt in [f"{frame_id:06d}_{inst_idx:06d}.png",
                                        f"{frame_id:06d}_{inst_idx:02d}.png"]:
                                candidate = scene_dir / mask_dir_name / fmt
                                if candidate.exists():
                                    mask_path = candidate
                                    break
                            if mask_path:
                                break

                    # Visibility fraction
                    visibility = -1.0
                    if scene_gt_info and frame_id_str in scene_gt_info:
                        info_list = scene_gt_info[frame_id_str]
                        if inst_idx < len(info_list):
                            visibility = info_list[inst_idx].get('visib_fract', -1.0)

                    samples.append({
                        'scene_id': scene_id,
                        'frame_id': frame_id,
                        'instance_idx': inst_idx,
                        'obj_id': self.obj_id,
                        'rgb_path': rgb_path,
                        'mask_path': mask_path,
                        'cam_K': cam_K,
                        'pose': pose,
                        'visibility': visibility,
                    })

        # Sort for deterministic ordering
        samples.sort(key=lambda s: (s['scene_id'], s['frame_id'], s['instance_idx']))
        return samples

    def _load_patch_samples(self, scene_dir, scene_id):
        """Load pre-cropped patch samples from patches_{sensor}/ or patches/ directory."""
        patches_dir = scene_dir / self._sfx("patches")
        patch_meta_path = scene_dir / f"{self._sfx('patch_meta')}.json"

        with open(patch_meta_path, 'r') as f:
            patch_meta = json.load(f)

        samples = []
        for patch_key, meta in patch_meta.items():
            if meta['obj_id'] != self.obj_id:
                continue

            # Use masked RGB if available, else plain RGB
            if self.use_masks:
                rgb_path = patches_dir / "rgb_masked" / f"{patch_key}.png"
                if not rgb_path.exists():
                    rgb_path = patches_dir / "rgb" / f"{patch_key}.png"
            else:
                rgb_path = patches_dir / "rgb" / f"{patch_key}.png"

            if not rgb_path.exists():
                continue

            cam_K = np.array(meta['cam_K'], dtype=np.float32).reshape(3, 3)
            R = np.array(meta['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            t = np.array(meta['cam_t_m2c'], dtype=np.float32).reshape(3, 1)
            pose = np.hstack([R, t])

            mask_path = patches_dir / "mask_visib" / f"{patch_key}.png"
            if not mask_path.exists():
                mask_path = None

            samples.append({
                'scene_id': scene_id,
                'frame_id': meta['frame_id'],
                'instance_idx': meta['instance_idx'],
                'obj_id': meta['obj_id'],
                'rgb_path': rgb_path,
                'mask_path': mask_path,
                'cam_K': cam_K,
                'pose': pose,
                'visibility': meta.get('visibility', -1.0),
                'is_patch': True,
            })

        print(f"[BOPTest DALI] Loaded {len(samples)} pre-cropped patches from {scene_id}")
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
        """Generate heatmap on CPU from pose and keypoints."""
        idx = sample_info.idx_in_epoch % self.num_samples
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

        # Generate Gaussian heatmaps (same as heatmap.py: sigma=(25,25))
        H, W = self.img_size
        heatmaps = np.zeros((self.num_keypoints, H, W), dtype=np.float32)

        for i, (x, y) in enumerate(pts_2d):
            ix, iy = int(x), int(y)
            if 0 <= ix < W and 0 <= iy < H:
                heatmaps[i, iy, ix] = 1.0
                import cv2
                heatmaps[i] = cv2.GaussianBlur(heatmaps[i], (25, 25), 0)
                am = np.amax(heatmaps[i])
                if am > 0:
                    heatmaps[i] = heatmaps[i] / am

        return heatmaps

    def define_graph(self):
        """Define DALI computation graph — no augmentation."""

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
        heatmap = fn.external_source(
            source=self.get_heatmap,
            batch=False,
            device="cpu"
        )
        heatmap = heatmap.gpu()

        # === Read and decode RGB ===
        img_encoded, _ = fn.readers.file(
            files=self.img_files,
            random_shuffle=False,
            name="img_reader"
        )
        img = fn.decoders.image(img_encoded, device="mixed", output_type=types.RGB)
        img = fn.resize(img, size=self.img_size)

        # === Apply instance mask if available ===
        if self.use_masks and self.has_valid_masks:
            mask_encoded, _ = fn.readers.file(
                files=self.mask_files,
                random_shuffle=False,
                name="mask_reader"
            )
            mask = fn.decoders.image(mask_encoded, device="mixed", output_type=types.GRAY)
            mask = fn.resize(mask, size=self.img_size, interp_type=types.INTERP_NN)
            mask_float = fn.cast(mask, dtype=types.FLOAT) / 255.0

            # Apply mask to image: zero out non-object pixels
            mask_rgb = fn.cat(mask_float, mask_float, mask_float, axis=2)
            img_float = fn.cast(img, dtype=types.FLOAT)
            img = fn.cast(mask_rgb * img_float, dtype=types.UINT8)

        # Compute edge input from grayscale Laplacian BEFORE normalization (requires uint8)
        if self.compute_edge_input:
            import nvidia.dali.math as dali_math
            img_gray_for_edge = fn.color_space_conversion(
                img, image_type=types.RGB, output_type=types.GRAY)
            img_gray_f = fn.cast(img_gray_for_edge, dtype=types.FLOAT) / 255.0
            edge_input = fn.laplacian(img_gray_f, window_size=3)
            edge_input = dali_math.abs(edge_input)
            edge_input = dali_math.min(edge_input, 1.0)
            edge_input = fn.transpose(edge_input, perm=[2, 0, 1])  # HW1 → 1HW

        # === Normalization (no augmentation) ===
        img = fn.cast(img, dtype=types.FLOAT) / 255.0

        mean = np.array([0.419, 0.427, 0.424], dtype=np.float32)
        std = np.array([0.184, 0.206, 0.197], dtype=np.float32)
        img = (img - mean) / std

        # Transpose to CHW
        img = fn.transpose(img, perm=[2, 0, 1])

        # Ensure types
        heatmap = fn.cast(heatmap, dtype=types.FLOAT)
        K = fn.cast(K, dtype=types.FLOAT)
        pose = fn.cast(pose, dtype=types.FLOAT)

        if self.compute_edge_input:
            return img, heatmap, K, pose, edge_input
        return img, heatmap, K, pose

    def __len__(self):
        return self.num_samples


def get_bop_test_dali_loader(
    bop_root,
    obj_id,
    sensor=None,
    keypoints_dir="keypoints",
    scene_ids=None,
    batch_size=1,
    num_threads=4,
    device_id=0,
    img_size=(480, 640),
    use_masks=True,
    split="test",
    compute_edge_input=False,
):
    """
    Create DALI test loader for BOP-format evaluation.

    Args:
        bop_root: Path to BOP dataset root (e.g., data/ROBI_BOP/Gear_BOP)
        obj_id: Target object ID (1-indexed)
        sensor: Sensor name (ensenso, realsense)
        keypoints_dir: Directory containing keypoint files (relative to bop_root or absolute)
        scene_ids: List of scene IDs (e.g., ['000001']). None = all scenes.
        batch_size: Batch size (default: 1 for test)
        num_threads: CPU threads for DALI
        device_id: GPU device ID
        img_size: Output image size (H, W)
        use_masks: Apply instance visibility masks
        split: Dataset split (test, train)

    Returns:
        (loader, samples):
        - loader: DALIGenericIterator yielding dicts with keys:
            "images" [B,3,H,W], "heatmaps" [B,N,H,W], "K" [B,3,3], "pose" [B,3,4]
        - samples: list of sample metadata dicts for CSV output
    """
    bop_root = Path(bop_root)

    # Resolve keypoints path
    kp_dir = Path(keypoints_dir)
    if not kp_dir.is_absolute():
        kp_dir = bop_root / keypoints_dir

    keypoints_path = None
    # Prefer *_mm.txt (BOP mm convention); fall back to legacy metre files last
    # so the assert in the pipeline fires if only the metre file exists.
    candidates = [
        kp_dir / f"obj_{obj_id:06d}_mm.txt",
        kp_dir / f"obj{obj_id}_mm.txt",
        kp_dir / f"{obj_id}_mm.txt",
        kp_dir / f"obj_{obj_id:06d}.txt",
        kp_dir / f"obj{obj_id}.txt",
        kp_dir / f"{obj_id}.txt",
    ]
    # Also try single .txt file if only one exists (prefer _mm variant)
    if kp_dir.exists():
        mm_files = list(kp_dir.glob("*_mm.txt"))
        if len(mm_files) == 1:
            candidates.insert(0, mm_files[0])
        else:
            txt_files = list(kp_dir.glob("*.txt"))
            if len(txt_files) == 1:
                candidates.insert(0, txt_files[0])

    for c in candidates:
        if c.exists():
            keypoints_path = c
            break

    if keypoints_path is None:
        raise FileNotFoundError(
            f"Keypoints not found for obj_id={obj_id}. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    pipeline = BOPTestDALIPipeline(
        bop_root=str(bop_root),
        obj_id=obj_id,
        keypoints_path=str(keypoints_path),
        sensor=sensor,
        scene_ids=scene_ids,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        img_size=img_size,
        use_masks=use_masks,
        split=split,
        compute_edge_input=compute_edge_input,
    )

    pipeline.build()

    output_map = ["images", "heatmaps", "K", "pose"]
    if compute_edge_input:
        output_map.append("edges_input")

    loader = DALIGenericIterator(
        pipelines=[pipeline],
        output_map=output_map,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        reader_name="img_reader"
    )

    return loader, pipeline.samples
