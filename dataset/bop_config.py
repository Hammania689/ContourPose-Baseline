"""
BOP Configuration Utilities.

Load object metadata (diameters, symmetries) from BOP models_info.json
to replace hardcoded values in config.py.

Usage:
    from dataset.bop_config import load_bop_config, get_bop_diameter, get_bop_symmetry

    config = load_bop_config("path/to/bop_dataset")
    diameter = get_bop_diameter(config, obj_id=1)
    symmetry = get_bop_symmetry(config, obj_id=1)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List


def load_bop_config(bop_root: str) -> Dict:
    """
    Load BOP models_info.json.

    Args:
        bop_root: Root directory of BOP dataset

    Returns:
        Dict with object metadata keyed by object ID (as string)
    """
    info_path = Path(bop_root) / "models" / "models_info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"models_info.json not found at {info_path}")

    with open(info_path, 'r') as f:
        return json.load(f)


def get_bop_diameter(config: Dict, obj_id: int) -> float:
    """
    Get object diameter from BOP config.

    Args:
        config: BOP config dict from load_bop_config()
        obj_id: Object ID (1-indexed)

    Returns:
        Object diameter in mm
    """
    obj_key = str(obj_id)
    if obj_key not in config:
        raise KeyError(f"Object {obj_id} not found in BOP config")

    return config[obj_key].get("diameter", 0.0)


def get_bop_symmetry(config: Dict, obj_id: int) -> Optional[Dict]:
    """
    Get symmetry information from BOP config.

    BOP models_info.json contains symmetry transforms in format:
    - "symmetries_discrete": list of 4x4 transformation matrices
    - "symmetries_continuous": dict with rotation axis info

    Args:
        config: BOP config dict
        obj_id: Object ID (1-indexed)

    Returns:
        Dict with symmetry info, or None if no symmetries
    """
    obj_key = str(obj_id)
    if obj_key not in config:
        return None

    obj_info = config[obj_key]
    symmetry = {}

    if "symmetries_discrete" in obj_info:
        # List of 4x4 transformation matrices (flattened)
        sym_matrices = []
        for sym in obj_info["symmetries_discrete"]:
            mat = np.array(sym, dtype=np.float32).reshape(4, 4)
            sym_matrices.append(mat)
        symmetry["discrete"] = sym_matrices

    if "symmetries_continuous" in obj_info:
        symmetry["continuous"] = obj_info["symmetries_continuous"]

    return symmetry if symmetry else None


def is_symmetric_object(config: Dict, obj_id: int) -> bool:
    """Check if object has symmetry (discrete or continuous)."""
    sym = get_bop_symmetry(config, obj_id)
    return sym is not None and len(sym) > 0


def get_bop_diameters_dict(bop_root: str) -> Dict[str, float]:
    """
    Get diameters dict in ContourPose format (obj1, obj2, ...).

    Args:
        bop_root: BOP dataset root

    Returns:
        Dict mapping "obj{id}" to diameter in mm
    """
    config = load_bop_config(bop_root)
    diameters = {}

    for obj_id_str, info in config.items():
        obj_id = int(obj_id_str)
        # Use ContourPose naming convention
        key = f"obj{obj_id}"
        diameters[key] = info.get("diameter", 0.0)

    return diameters


def get_bop_symmetry_transforms(bop_root: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Get symmetry transforms dict in ContourPose format.

    Returns transforms for pose_reverse() in eval.py.
    For BOP, this extracts the 180-degree rotation symmetry if present.

    Args:
        bop_root: BOP dataset root

    Returns:
        Dict mapping "obj{id}" to 4x4 symmetry transform or None
    """
    config = load_bop_config(bop_root)
    rtDic = {}

    for obj_id_str, info in config.items():
        obj_id = int(obj_id_str)
        key = f"obj{obj_id}"

        if "symmetries_discrete" in info and len(info["symmetries_discrete"]) > 0:
            # Use first discrete symmetry (typically 180-degree rotation)
            sym = np.array(info["symmetries_discrete"][0], dtype=np.float32).reshape(4, 4)
            rtDic[key] = sym
        else:
            rtDic[key] = None

    return rtDic


def get_all_symmetry_transforms(bop_root: str) -> Dict[str, List[np.ndarray]]:
    """
    Get all symmetry transforms for each object as a list of 4x4 matrices.

    For discrete symmetries: returns identity + all discrete transforms.
    Continuous symmetries are NOT included here — they are handled analytically
    by min_symmetry_rotation_error() via get_continuous_symmetry_axes().

    Args:
        bop_root: BOP dataset root

    Returns:
        Dict mapping "obj{id}" to list of 4x4 numpy arrays (identity is always first)
    """
    config = load_bop_config(bop_root)
    result = {}

    for obj_id_str, info in config.items():
        obj_id = int(obj_id_str)
        key = f"obj{obj_id}"
        transforms = [np.eye(4, dtype=np.float64)]  # Identity always first

        # Discrete symmetries only
        if "symmetries_discrete" in info:
            for sym in info["symmetries_discrete"]:
                mat = np.array(sym, dtype=np.float64).reshape(4, 4)
                transforms.append(mat)

        result[key] = transforms

    return result


def get_continuous_symmetry_axes(bop_root: str) -> Dict[str, List[np.ndarray]]:
    """
    Get continuous symmetry rotation axes for each object.

    These are handled analytically by min_symmetry_rotation_error() using
    a closed-form solution instead of brute-force discretization.

    Args:
        bop_root: BOP dataset root

    Returns:
        Dict mapping "obj{id}" to list of unit axis vectors [3].
        Empty list if object has no continuous symmetry.
    """
    config = load_bop_config(bop_root)
    result = {}

    for obj_id_str, info in config.items():
        obj_id = int(obj_id_str)
        key = f"obj{obj_id}"
        axes = []

        if "symmetries_continuous" in info:
            for sym_cont in info["symmetries_continuous"]:
                axis = np.array(sym_cont["axis"], dtype=np.float64)
                axis = axis / np.linalg.norm(axis)
                axes.append(axis)

        result[key] = axes

    return result


def list_bop_objects(bop_root: str) -> List[int]:
    """List all object IDs in BOP dataset."""
    config = load_bop_config(bop_root)
    return sorted([int(k) for k in config.keys()])


# Convenience function to get all config needed for ContourPose
def get_contourpose_config_from_bop(bop_root: str) -> Dict:
    """
    Get all configuration needed for ContourPose from BOP dataset.

    Returns:
        Dict with keys:
        - "diameters": Dict[str, float] mapping obj names to diameters
        - "rtDic": Dict[str, np.ndarray] symmetry transforms
        - "symmetric_objects": List[str] names of symmetric objects
        - "models_info": Raw BOP models_info dict
    """
    config = load_bop_config(bop_root)

    diameters = {}
    rtDic = {}
    symmetric_objects = []

    for obj_id_str, info in config.items():
        obj_id = int(obj_id_str)
        key = f"obj{obj_id}"

        # Diameter
        diameters[key] = info.get("diameter", 0.0)

        # Symmetry
        if "symmetries_discrete" in info and len(info["symmetries_discrete"]) > 0:
            sym = np.array(info["symmetries_discrete"][0], dtype=np.float32).reshape(4, 4)
            rtDic[key] = sym
            symmetric_objects.append(key)
        else:
            rtDic[key] = None

    # All symmetry transforms (identity + discrete + discretized continuous)
    all_sym = get_all_symmetry_transforms(bop_root)

    return {
        "diameters": diameters,
        "rtDic": rtDic,
        "symmetric_objects": symmetric_objects,
        "models_info": config,
        "all_symmetry_transforms": all_sym,
    }
