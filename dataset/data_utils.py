"""
Data loading utilities for ContourPose.

Centralizes dataloader creation for both ContourPose and BOP formats,
supporting DALI-accelerated pipelines for efficient training.

Usage:
    from dataset.data_utils import (
        create_train_loader,
        create_validation_setup,
        create_bop_validation_setup,
        create_test_loader,
        get_keypoints_path,
        load_keypoints,
        compute_cosine_annealing_T0
    )

    # Get keypoints
    keypoints = load_keypoints(args)

    # Create training dataloader
    train_loader, bop_config = create_train_loader(args)

    # Create validation setup (ContourPose format)
    val_setup = create_validation_setup(args)

    # Create test loader
    test_loader = create_test_loader(args)
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


# =============================================================================
# Learning Rate Scheduler Utilities
# =============================================================================

def compute_scaled_lr(base_lr: float, base_batch_size: int, new_batch_size: int) -> float:
    """
    Compute scaled learning rate using linear scaling rule.

    When batch size increases by factor k, learning rate should also
    increase by factor k to maintain similar training dynamics.

    Args:
        base_lr: Original learning rate
        base_batch_size: Original batch size the base_lr was tuned for
        new_batch_size: New batch size to use

    Returns:
        Scaled learning rate
    """
    scale_factor = new_batch_size / base_batch_size
    new_lr = base_lr * scale_factor
    print(f"[LR Scaling] {base_lr} × {scale_factor:.1f}x = {new_lr} (batch {base_batch_size} → {new_batch_size})")
    return new_lr


def compute_cosine_annealing_T0(
    dataset_size: int,
    batch_size: int,
    epochs_per_cycle: int = 10,
    val_split: float = 0.2
) -> int:
    """
    Compute T_0 for CosineAnnealingWarmRestarts scheduler.

    T_0 is the number of iterations (optimizer steps) before the first restart.
    The learning rate follows a cosine curve from max to min over T_0 steps,
    then restarts.

    Args:
        dataset_size: Total number of samples in the dataset
        batch_size: Batch size used for training
        epochs_per_cycle: Number of epochs per cosine annealing cycle (default: 10)
            - Lower values (5-10): More frequent restarts, good for fine-tuning
            - Higher values (20-30): Fewer restarts, good for training from scratch
        val_split: Fraction of data used for validation (default: 0.2)
            The training set size is dataset_size * (1 - val_split)

    Returns:
        T_0: Number of iterations for first cycle

    Example:
        >>> # 8000 images, batch_size=32, 10 epochs per cycle, 20% validation
        >>> T_0 = compute_cosine_annealing_T0(8000, 32, 10, 0.2)
        >>> # train_size = 8000 * 0.8 = 6400
        >>> # iters_per_epoch = 6400 / 32 = 200
        >>> # T_0 = 200 * 10 = 2000
        >>> print(T_0)
        2000
    """
    # Account for validation split
    train_size = int(dataset_size * (1 - val_split))

    # Iterations per epoch (number of batches)
    iters_per_epoch = train_size // batch_size

    # T_0 = iterations per epoch * epochs per cycle
    T_0 = iters_per_epoch * epochs_per_cycle

    print(f"[LR Scheduler] Computed T_0 for CosineAnnealingWarmRestarts:")
    print(f"  Dataset size: {dataset_size} (train: {train_size} after {val_split*100:.0f}% val split)")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations per epoch: {iters_per_epoch}")
    print(f"  Epochs per cycle: {epochs_per_cycle}")
    print(f"  T_0 = {T_0} iterations")

    return T_0


def compute_scaled_lr(
    base_lr: float,
    base_batch_size: int,
    new_batch_size: int
) -> float:
    """
    Compute scaled learning rate using linear scaling rule.

    When batch size increases by factor k, learning rate should also
    increase by factor k to maintain similar training dynamics.

    Args:
        base_lr: Original learning rate
        base_batch_size: Original batch size the base_lr was tuned for
        new_batch_size: New batch size to use

    Returns:
        Scaled learning rate

    Example:
        >>> # Original: lr=0.1 at batch_size=8
        >>> # New batch_size=32
        >>> new_lr = compute_scaled_lr(0.1, 8, 32)
        >>> print(new_lr)
        0.4
    """
    scale_factor = new_batch_size / base_batch_size
    new_lr = base_lr * scale_factor

    print(f"[LR Scaling] Linear scaling rule:")
    print(f"  Base: lr={base_lr} at batch_size={base_batch_size}")
    print(f"  Scale factor: {scale_factor:.2f}x")
    print(f"  New: lr={new_lr} at batch_size={new_batch_size}")

    return new_lr


# =============================================================================
# Keypoint Utilities
# =============================================================================

def get_keypoints_path(args) -> Path:
    """
    Get keypoints file path.

    Resolution order:
        1. args.data_root / keypoints_dir / ...  (explicit --data_root)
        2. args.bop_root / keypoints_dir / ...    (BOP training root)
        3. cwd / keypoints / {class_type}.txt     (legacy fallback)

    Args:
        args: Argument namespace with data_root, bop_root, obj_id,
              keypoints_dir, class_type

    Returns:
        Path to keypoints file

    Raises:
        FileNotFoundError: If keypoints file not found
    """
    # Determine the base root to search under
    data_root = getattr(args, 'data_root', None)
    bop_root = getattr(args, 'bop_root', None)
    root = Path(data_root) if data_root else (Path(bop_root) if bop_root else None)

    if root is not None:
        keypoints_dir_arg = getattr(args, 'keypoints_dir', 'keypoints')
        obj_id = getattr(args, 'obj_id', 1)
        class_type = getattr(args, 'class_type', None)

        # Handle both relative and absolute keypoints_dir paths
        keypoints_dir = Path(keypoints_dir_arg)
        if keypoints_dir.is_absolute() or keypoints_dir.exists():
            base_dir = keypoints_dir
        else:
            base_dir = root / keypoints_dir

        candidates = []
        # Prefer class_type (e.g. "obj2.txt") over obj_id (which defaults to 1)
        if class_type:
            candidates.append(base_dir / f"{class_type}.txt")
        candidates.extend([
            base_dir / f"obj_{obj_id:06d}.txt",
            base_dir / f"obj{obj_id}.txt",
            base_dir / f"{obj_id}.txt",
        ])

        # If only one .txt file in dir, use it
        if base_dir.exists():
            txt_files = list(base_dir.glob("*.txt"))
            if len(txt_files) == 1:
                candidates.insert(0, txt_files[0])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Keypoints not found for obj_id={obj_id}. "
            f"Searched: {[str(c) for c in candidates]}"
        )
    else:
        # Legacy fallback: cwd/keypoints/
        return Path(os.getcwd()) / "keypoints" / f"{args.class_type}.txt"


def load_keypoints(args) -> np.ndarray:
    """
    Load keypoints from file.

    Args:
        args: Argument namespace

    Returns:
        Keypoints array of shape (N, 3)
    """
    keypoints_path = get_keypoints_path(args)
    keypoints = np.loadtxt(str(keypoints_path))
    print(f"[Keypoints] Loaded {len(keypoints)} keypoints from {keypoints_path}")
    return keypoints


# =============================================================================
# Training Dataloader Creation
# =============================================================================

def create_bop_train_loader(args) -> Tuple[Any, Dict]:
    """
    Create DALI training dataloader for synthetic data.

    Args:
        args: Argument namespace with arguments:
            - bop_root: Dataset directory (contains rgb/, mask/, edges/, scene_gt.json, etc.)
            - obj_id: Object ID (1-indexed)
            - keypoints_dir: Directory containing keypoint files
            - batch_size: Batch size
            - heatmap_dir: Optional precomputed heatmaps directory
            - background_dir: Optional background images directory

    Returns:
        Tuple of (dataloader, config dict or None)
    """
    from dataset.BOPDALIDataset import get_bop_dali_loader

    print(f"[Training] Using synthetic data from {args.bop_root}")
    print(f"[Training] Object ID: {args.obj_id}")

    # Get img_size from args or default to (256, 256) for new datasets
    img_size = getattr(args, 'img_size', (480, 640))
    print(f"[Training] Image size: {img_size}")

    train_loader = get_bop_dali_loader(
        data_dir=args.bop_root,
        obj_id=args.obj_id,
        keypoints_dir=getattr(args, 'keypoints_dir', 'keypoints'),
        batch_size=args.batch_size,
        num_threads=8,
        device_id=0,
        seed=42,
        img_size=img_size,
        heatmap_dir=getattr(args, 'heatmap_dir', None),
        background_dir=getattr(args, 'background_dir', None),
        compute_edge_input=getattr(args, 'compute_edge_input', False),
    )

    return train_loader, None


def create_contourpose_train_loader(args, num_gpus: int = 2) -> Any:
    """
    Create DALI training dataloader for ContourPose format.

    Args:
        args: Argument namespace with data_path, class_type, batch_size
        num_gpus: Number of GPUs for logging

    Returns:
        DALI dataloader

    Raises:
        FileNotFoundError: If 2D keypoints not precomputed
    """
    from dataset.DALIDataset import get_dali_loader

    # Check keypoints_2d exist
    keypoints_2d_dir = Path(args.data_path) / "train" / "renders" / args.class_type / "keypoints_2d"
    if not keypoints_2d_dir.exists():
        raise FileNotFoundError(
            f"2D keypoints not found at {keypoints_2d_dir}\n"
            f"Please run: python scripts/precompute_keypoints_2d.py --class_type {args.class_type}"
        )

    print("[Training] Using DALI pipeline with GPU heatmap generation")
    print(f"[Training] Batch size: {args.batch_size} (effective: {args.batch_size // num_gpus} per GPU)")

    train_loader = get_dali_loader(
        data_root=args.data_path,
        class_type=args.class_type,
        batch_size=args.batch_size,
        num_threads=8,
        device_id=0,
        seed=42,
        compute_edge_input=getattr(args, 'compute_edge_input', False),
    )

    return train_loader


def create_train_loader(args, num_gpus: int = 2) -> Tuple[Any, Optional[Dict]]:
    """
    Create training dataloader based on format (BOP or ContourPose).

    Args:
        args: Argument namespace
        num_gpus: Number of GPUs

    Returns:
        Tuple of (dataloader, bop_config or None)
    """
    if getattr(args, 'bop_root', None):
        return create_bop_train_loader(args)
    else:
        loader = create_contourpose_train_loader(args, num_gpus)
        return loader, None


# =============================================================================
# Cross-Validation Utilities
# =============================================================================

def create_cv_split(
    num_samples: int,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create train/val split for cross-validation.

    Args:
        num_samples: Total number of samples
        val_ratio: Fraction of data for validation (default: 0.15 = 15%)
        seed: Random seed for reproducibility

    Returns:
        train_indices: List of training sample indices
        val_indices: List of validation sample indices
    """
    np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    val_size = int(num_samples * val_ratio)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    print(f"[Cross-Validation] Total samples: {num_samples}")
    print(f"[Cross-Validation] Train: {len(train_indices)} ({100*(1-val_ratio):.1f}%)")
    print(f"[Cross-Validation] Val: {len(val_indices)} ({100*val_ratio:.1f}%)")

    return train_indices, val_indices


def create_cross_validation_folds(
    total_files: int,
    val_split: float = 0.2,
    n_folds: int = 5,
    seed: int = 42
) -> Tuple[List[int], List[List[int]]]:
    """
    Create train/validation split with multiple cross-validation folds.

    Args:
        total_files: Total number of files in dataset
        val_split: Fraction for validation
        n_folds: Number of validation folds
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, list of validation fold indices)
    """
    np.random.seed(seed)
    all_indices = np.arange(total_files)
    np.random.shuffle(all_indices)

    val_size = int(total_files * val_split)
    train_size = total_files - val_size

    train_indices = all_indices[:train_size].tolist()
    val_indices = all_indices[train_size:].tolist()

    # Split validation into folds
    fold_size = val_size // n_folds
    val_folds = [
        val_indices[i * fold_size:(i + 1) * fold_size]
        for i in range(n_folds)
    ]

    return train_indices, val_folds


def create_multi_val_loaders(
    data_root: str,
    class_type: str,
    val_indices: List[int],
    batch_size: int,
    num_diverse_loaders: int = 4,
    base_seed: int = 1000
) -> Dict[str, Any]:
    """
    Create multiple validation loaders:
    - 1 val-clean (consistent, for tracking metrics)
    - N val-diverse (different seeds, for robustness testing)

    Args:
        data_root: Path to data directory
        class_type: Object class
        val_indices: Validation sample indices
        batch_size: Batch size
        num_diverse_loaders: Number of diverse validation loaders
        base_seed: Base seed for diverse loaders

    Returns:
        Dictionary with 'clean' and 'diverse' loaders
    """
    from dataset.DALIDataset import get_dali_loader

    # Val-Clean: consistent validation
    val_clean = get_dali_loader(
        data_root=data_root,
        class_type=class_type,
        batch_size=batch_size,
        num_threads=4,
        device_id=0,
        seed=999,
        file_indices=val_indices
    )

    # Val-Diverse: multiple loaders with different seeds
    val_diverse = []
    for i in range(num_diverse_loaders):
        loader = get_dali_loader(
            data_root=data_root,
            class_type=class_type,
            batch_size=batch_size,
            num_threads=4,
            device_id=0,
            seed=base_seed + i,
            file_indices=val_indices
        )
        val_diverse.append(loader)

    print(f"[Validation Loaders] Created 1 val-clean + {num_diverse_loaders} val-diverse loaders")

    return {
        'clean': val_clean,
        'diverse': val_diverse
    }


def create_validation_setup(args, num_gpus: int = 2) -> Dict[str, Any]:
    """
    Create complete validation setup with cross-validation folds.

    Note: Currently only supports ContourPose format with DALI.
    For BOP format, validation setup needs to be implemented separately.

    Args:
        args: Argument namespace with data_path, class_type, batch_size
        num_gpus: Number of GPUs

    Returns:
        Dict with keys:
        - train_loader: Training dataloader (with train indices only)
        - fixed_val_loader: Fixed validation loader (fold 0)
        - fixed_batch: Cached batch for visualization
        - train_indices: List of training indices
        - val_folds: List of validation fold indices
        - fold_size: Number of images per fold
    """
    from dataset.DALIDataset import get_dali_loader

    # Cross-validation parameters
    total_files = 6600  # Total synthetic renders (adjust per class if needed)
    val_split = 0.2
    n_folds = 5

    print("[Cross-Validation] Setting up 5-fold validation (1 fixed + 4 rotating)")

    # Create train/val split
    train_indices, val_folds = create_cross_validation_folds(
        total_files=total_files,
        val_split=val_split,
        n_folds=n_folds,
        seed=42
    )

    train_size = len(train_indices)
    val_size = total_files - train_size
    fold_size = val_size // n_folds

    print(f"[Data Split] Total: {total_files}, Train: {train_size}, Val: {val_size}")
    print(f"[Data Split] Each val fold: {fold_size} images = ~{fold_size // args.batch_size} batches")

    compute_edge_input = getattr(args, 'compute_edge_input', False)

    # Create training loader with train indices only
    train_loader = get_dali_loader(
        data_root=args.data_path,
        class_type=args.class_type,
        batch_size=args.batch_size,
        num_threads=8,
        device_id=0,
        seed=42,
        file_indices=train_indices,
        compute_edge_input=compute_edge_input,
    )

    # Create fixed validation loader (fold 0) - deterministic ordering
    print("[Cross-Validation] Creating fixed validation loader (fold 0)")
    fixed_val_loader = get_dali_loader(
        data_root=args.data_path,
        class_type=args.class_type,
        batch_size=args.batch_size,
        num_threads=4,
        device_id=0,
        seed=1111,
        file_indices=val_folds[0],
        compute_edge_input=compute_edge_input,
    )

    # Create random validation loader - different seed each epoch for variety
    # Uses a time-based seed to get different batches each visualization
    import time
    random_seed = int(time.time()) % 10000
    print(f"[Cross-Validation] Creating random validation loader (seed={random_seed})")
    random_val_loader = get_dali_loader(
        data_root=args.data_path,
        class_type=args.class_type,
        batch_size=args.batch_size,
        num_threads=4,
        device_id=0,
        seed=random_seed,
        file_indices=val_folds[0],  # Same data, different order
        compute_edge_input=compute_edge_input,
    )

    # Cache a batch for visualization (fixed - always the same)
    print("[Cross-Validation] Pre-loading visualization batch")
    fixed_batch = next(iter(fixed_val_loader))[0]

    return {
        "train_loader": train_loader,
        "fixed_val_loader": fixed_val_loader,
        "random_val_loader": random_val_loader,
        "fixed_batch": fixed_batch,
        "train_indices": train_indices,
        "val_folds": val_folds,
        "fold_size": fold_size,
    }


# =============================================================================
# Test Dataloader Creation
# =============================================================================

def create_test_loader(args):
    """
    Create test dataloader.

    Routes to BOP DALI test loader when args.bop_root is set and
    --legacy_test is not set, otherwise uses legacy PyTorch DataLoader.

    Args:
        args: Argument namespace

    Returns:
        For BOP format: (DALIGenericIterator, list of sample metadata dicts)
        For legacy format: PyTorch DataLoader
    """
    use_legacy = getattr(args, 'legacy_test', False)
    if not use_legacy and getattr(args, 'bop_root', None) is not None:
        return create_bop_test_loader(args)

    from torch.utils.data import DataLoader
    from dataset.Dataset import MyDataset

    test_set = MyDataset(
        args.data_path,
        args.class_type,
        is_train=False,
        scene=args.scene,
        index=args.index,
        data_root=getattr(args, 'data_root', None),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return test_loader


def create_bop_test_loader(args):
    """
    Create DALI test loader for BOP-format evaluation.

    Args:
        args: Argument namespace with bop_root, obj_id, sensor,
              keypoints_dir, scene_ids, img_size, use_masks

    Returns:
        (DALIGenericIterator, list of sample metadata dicts)
    """
    from dataset.BOPTestDALIDataset import get_bop_test_dali_loader

    scene_ids = None
    if getattr(args, 'scene_ids', None):
        scene_ids = [s.strip() for s in args.scene_ids.split(',')]

    loader, samples = get_bop_test_dali_loader(
        bop_root=args.bop_root,
        obj_id=args.obj_id,
        sensor=getattr(args, 'sensor', 'ensenso'),
        keypoints_dir=getattr(args, 'keypoints_dir', 'keypoints'),
        scene_ids=scene_ids,
        batch_size=args.batch_size,
        num_threads=4,
        device_id=0,
        img_size=getattr(args, 'img_size', (480, 640)),
        use_masks=getattr(args, 'use_masks_bool', True),
        compute_edge_input=getattr(args, 'compute_edge_input', False),
    )

    return loader, samples


# =============================================================================
# BOP-Specific Utilities
# =============================================================================

def count_bop_samples(data_dir: str, obj_id: int) -> int:
    """
    Count total samples for a specific object in a BOP dataset.

    Args:
        data_dir: Dataset directory containing scene_gt.json
        obj_id: Target object ID (1-indexed)

    Returns:
        Total number of samples for the object
    """
    import json

    gt_path = Path(data_dir) / "scene_gt.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"scene_gt.json not found in {data_dir}")

    with open(gt_path, 'r') as f:
        scene_gt = json.load(f)

    count = 0
    for frame_id_str, gt_list in scene_gt.items():
        for gt in gt_list:
            if gt['obj_id'] == obj_id:
                count += 1

    return count


def create_bop_validation_setup(args, num_gpus: int = 2) -> Dict[str, Any]:
    """
    Create complete validation setup with cross-validation for BOP format.

    Args:
        args: Argument namespace with bop_root, obj_id, keypoints_dir, batch_size
        num_gpus: Number of GPUs

    Returns:
        Dict with keys:
        - train_loader: Training dataloader (with train indices only)
        - fixed_val_loader: Fixed validation loader
        - fixed_batch: Cached batch for visualization
        - train_indices: List of training indices
        - val_indices: List of validation indices
    """
    from dataset.BOPDALIDataset import get_bop_dali_loader

    # Count total samples
    total_samples = count_bop_samples(args.bop_root, args.obj_id)
    print(f"[BOP Cross-Validation] Total samples for obj_id={args.obj_id}: {total_samples}")

    # Create train/val split (80/20)
    val_split = 0.2
    train_indices, val_indices = create_cv_split(
        num_samples=total_samples,
        val_ratio=val_split,
        seed=42
    )

    print(f"[BOP Data Split] Train: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"[BOP Data Split] Val batches: ~{len(val_indices) // args.batch_size}")

    # Get img_size from args or default to (256, 256) for new datasets
    img_size = getattr(args, 'img_size', (480, 640))
    print(f"[BOP Cross-Validation] Image size: {img_size}")

    compute_edge_input = getattr(args, 'compute_edge_input', False)

    # Create training loader with train indices only
    print("[BOP Cross-Validation] Creating training loader")
    train_loader = get_bop_dali_loader(
        data_dir=args.bop_root,
        obj_id=args.obj_id,
        keypoints_dir=getattr(args, 'keypoints_dir', 'keypoints'),
        batch_size=args.batch_size,
        num_threads=8,
        device_id=0,
        seed=42,
        img_size=img_size,
        heatmap_dir=getattr(args, 'heatmap_dir', None),
        background_dir=getattr(args, 'background_dir', None),
        file_indices=train_indices,
        compute_edge_input=compute_edge_input,
    )

    # Create fixed validation loader - deterministic ordering, no background aug
    print("[BOP Cross-Validation] Creating fixed validation loader")
    fixed_val_loader = get_bop_dali_loader(
        data_dir=args.bop_root,
        obj_id=args.obj_id,
        keypoints_dir=getattr(args, 'keypoints_dir', 'keypoints'),
        batch_size=args.batch_size,
        num_threads=4,
        device_id=0,
        seed=1111,  # Fixed seed for deterministic ordering
        img_size=img_size,
        heatmap_dir=getattr(args, 'heatmap_dir', None),
        background_dir=None,  # No background aug in validation — saves GPU memory
        file_indices=val_indices,
        compute_edge_input=compute_edge_input,
    )

    # Create random validation loader - different seed each run for variety, no background aug
    import time
    random_seed = int(time.time()) % 10000
    print(f"[BOP Cross-Validation] Creating random validation loader (seed={random_seed})")
    random_val_loader = get_bop_dali_loader(
        data_dir=args.bop_root,
        obj_id=args.obj_id,
        keypoints_dir=getattr(args, 'keypoints_dir', 'keypoints'),
        batch_size=args.batch_size,
        num_threads=4,
        device_id=0,
        seed=random_seed,  # Time-based seed for variety
        img_size=img_size,
        heatmap_dir=getattr(args, 'heatmap_dir', None),
        background_dir=None,  # No background aug in validation — saves GPU memory
        file_indices=val_indices,  # Same data, different order
        compute_edge_input=compute_edge_input,
    )

    # Cache a batch for visualization (fixed - always the same)
    print("[BOP Cross-Validation] Pre-loading visualization batch")
    fixed_batch = next(iter(fixed_val_loader))[0]

    return {
        "train_loader": train_loader,
        "fixed_val_loader": fixed_val_loader,
        "random_val_loader": random_val_loader,
        "fixed_batch": fixed_batch,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }


def get_bop_object_key(obj_id: int) -> str:
    """Convert BOP object ID to ContourPose key format."""
    return f"obj{obj_id}"


def get_bop_config_for_eval(bop_root: str, obj_id: int) -> Dict[str, Any]:
    """
    Get BOP configuration needed for evaluation.

    Args:
        bop_root: BOP dataset root
        obj_id: Object ID

    Returns:
        Dict with diameter, symmetry info for the object
    """
    from dataset.bop_config import (
        load_bop_config,
        get_bop_diameter,
        get_bop_symmetry
    )

    config = load_bop_config(bop_root)
    obj_key = get_bop_object_key(obj_id)

    return {
        "diameter": get_bop_diameter(config, obj_id),
        "symmetry": get_bop_symmetry(config, obj_id),
        "obj_key": obj_key,
    }
