#!/usr/bin/env python3
"""
Visualize BOP annotations for a scene to verify GT poses, keypoints, and masks.

Works with two layouts:
  Full-scene  — rgb/ (or rgb_{sensor}/), mask_visib/, scene_gt.json, scene_camera.json
  Patches     — patches/ (or patches_{sensor}/), patch_meta.json

For each sampled frame the script renders a single image that shows:
  - Full scene RGB (or patch, whichever is loaded)
  - Colored mask overlay per object instance
  - Projected 3D keypoints (green dots) from GT pose
  - Projected Valid3D contour samples (cyan dots)
  - Per-object label box with obj_id and visibility fraction

Usage:
    # Full-scene BOP (standard single-camera, no sensor suffix)
    python scripts/visualize_annotations.py \\
        --bop_root /home/ham/GithubWorkspace/ContourPose/data/RTLESS_BOP \\
        --obj_id 1 --scene_ids 000013 --num_frames 5

    # Multi-sensor ROBI BOP
    python scripts/visualize_annotations.py \\
        --bop_root data/ROBI_BOP/Gear_BOP \\
        --sensor ensenso_left --obj_id 1 --num_frames 10

    # Pre-extracted patches
    python scripts/visualize_annotations.py \\
        --bop_root data/ROBI_BOP/Gear_BOP \\
        --sensor ensenso_left --obj_id 1 --patches --num_frames 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

def load_ply(path):
    """Load a PLY file, return dict with 'pts' [N,3] (and optionally normals/colors/faces)."""
    import struct as _struct
    f = open(path, 'r')
    n_pts = n_faces = 0
    face_n_corners = 3
    pt_props, face_props = [], []
    is_binary = False
    in_vertex = in_face = False
    while True:
        line = f.readline().rstrip('\n').rstrip('\r')
        if line.startswith('element vertex'):
            n_pts = int(line.split()[-1]); in_vertex = True; in_face = False
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1]); in_vertex = False; in_face = True
        elif line.startswith('element'):
            in_vertex = in_face = False
        elif line.startswith('property') and in_vertex:
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and in_face:
            elems = line.split()
            if elems[-1] == 'vertex_indices':
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
        elif line.startswith('format') and 'binary' in line:
            is_binary = True
        elif line.startswith('end_header'):
            break
    model = {'pts': np.zeros((n_pts, 3), np.float64)}
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float64)
    pt_names = [p[0] for p in pt_props]
    fmts = {'float': ('f', 4), 'double': ('d', 8), 'int': ('i', 4), 'uchar': ('B', 1)}
    load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
    for pt_id in range(n_pts):
        if is_binary:
            pv = {}
            for prop in pt_props:
                fmt = fmts[prop[1]]; val = _struct.unpack(fmt[0], f.read(fmt[1]))[0]
                if prop[0] in load_props: pv[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            pv = {prop[0]: elems[i] for i, prop in enumerate(pt_props) if prop[0] in load_props}
        model['pts'][pt_id] = [float(pv['x']), float(pv['y']), float(pv['z'])]
    for face_id in range(n_faces):
        if is_binary:
            pv = {}
            for prop in face_props:
                fmt = fmts[prop[1]]; val = _struct.unpack(fmt[0], f.read(fmt[1]))[0]
                if prop[0] != 'n_corners': pv[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            pv = {prop[0]: elems[i] for i, prop in enumerate(face_props) if prop[0] != 'n_corners'}
        model['faces'][face_id] = [int(pv['ind_0']), int(pv['ind_1']), int(pv['ind_2'])]
    f.close()
    return model


# --- palette: one RGB colour per object instance index ----------------------
_PALETTE = [
    (255,  60,  60),  # red
    ( 60, 200,  60),  # green
    ( 60, 120, 255),  # blue
    (255, 200,   0),  # yellow
    (200,   0, 255),  # violet
    (  0, 220, 220),  # cyan
    (255, 140,   0),  # orange
    (180, 255,   0),  # lime
    (255,   0, 180),  # pink
    (  0, 180, 255),  # sky-blue
]

def _color(i):
    return _PALETTE[i % len(_PALETTE)]


# --- geometry helpers --------------------------------------------------------

def project_pts(pts_3d, K, R, t):
    """Project [N,3] 3D points (mm) into image. Returns [N,2] floats, valid mask."""
    cam = (R @ pts_3d.T + t).T        # [N,3] in camera frame
    proj = (K @ cam.T).T              # [N,3]
    valid = proj[:, 2] > 0
    px = np.zeros((len(pts_3d), 2))
    px[valid] = proj[valid, :2] / proj[valid, 2:3]
    return px, valid


def find_keypoints_file(data_root, obj_id):
    kp_dir = Path(data_root) / "keypoints"
    candidates = [
        kp_dir / f"obj_{obj_id:06d}.txt",
        kp_dir / f"obj{obj_id}.txt",
        kp_dir / f"{obj_id}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_valid3d_file(data_root, obj_id):
    v3d_dir = Path(data_root) / "Valid3D"
    candidates = [
        v3d_dir / f"obj_{obj_id:06d}.txt",
        v3d_dir / f"obj{obj_id}.txt",
        v3d_dir / f"{obj_id}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_model_file(bop_root, obj_id):
    models_dir = Path(bop_root) / "models"
    candidates = [
        models_dir / f"obj_{obj_id:06d}.ply",
        models_dir / f"obj{obj_id}.ply",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# --- image drawing helpers ---------------------------------------------------

def draw_mask_overlay(img, mask_path, color, alpha=0.45):
    if mask_path is None or not Path(mask_path).exists():
        return img
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return img
    # Resize mask to match image if needed
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask > 0] = (
        (1 - alpha) * img[mask > 0].astype(np.float32) +
        alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def draw_points(img, pts_2d, valid, color, radius=4):
    out = img.copy()
    for i, (pt, v) in enumerate(zip(pts_2d, valid)):
        if not v:
            continue
        x, y = int(round(pt[0])), int(round(pt[1]))
        h, w = out.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_label(img, text, pt, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = int(pt[0]), int(pt[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# --- sfx helper (mirrors BOPTestDALIPipeline) --------------------------------

def _sfx(base, sensor):
    return f"{base}_{sensor}" if sensor else base


# --- sample builders ---------------------------------------------------------

def build_fullres_samples(scene_dir, scene_gt, scene_camera, scene_gt_info, sensor, obj_ids_filter):
    """
    Enumerate (frame_id, inst_idx) pairs from scene_gt, return list of sample dicts.
    obj_ids_filter: set of obj_ids to include, or None for all.
    """
    samples = []
    for frame_str, gt_list in scene_gt.items():
        frame_id = int(frame_str)
        cam_data = scene_camera[frame_str]
        K = np.array(cam_data['cam_K'], dtype=np.float64).reshape(3, 3)

        rgb_dir = scene_dir / _sfx("rgb", sensor)
        rgb_path = rgb_dir / f"{frame_id:06d}.png"
        if not rgb_path.exists():
            rgb_path = rgb_dir / f"{frame_id:06d}.jpg"
        if not rgb_path.exists():
            continue

        instances = []
        for inst_idx, gt in enumerate(gt_list):
            oid = gt['obj_id']
            if obj_ids_filter and oid not in obj_ids_filter:
                continue

            R = np.array(gt['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
            t = np.array(gt['cam_t_m2c'], dtype=np.float64).reshape(3, 1)

            # Mask
            mask_path = None
            for mdir in [_sfx("mask_visib", sensor), _sfx("mask", sensor)]:
                for fmt in [f"{frame_id:06d}_{inst_idx:06d}.png",
                             f"{frame_id:06d}_{inst_idx:02d}.png"]:
                    c = scene_dir / mdir / fmt
                    if c.exists():
                        mask_path = c
                        break
                if mask_path:
                    break

            visibility = -1.0
            if scene_gt_info and frame_str in scene_gt_info:
                info_list = scene_gt_info[frame_str]
                if inst_idx < len(info_list):
                    visibility = info_list[inst_idx].get('visib_fract', -1.0)

            instances.append({
                'inst_idx': inst_idx,
                'obj_id': oid,
                'R': R,
                't': t,
                'mask_path': mask_path,
                'visibility': visibility,
            })

        if instances:
            samples.append({
                'frame_id': frame_id,
                'K': K,
                'rgb_path': rgb_path,
                'instances': instances,
                'is_patch': False,
            })

    samples.sort(key=lambda s: s['frame_id'])
    return samples


def build_patch_samples(scene_dir, patch_meta, sensor, obj_ids_filter):
    """Build sample list from patch_meta.json (pre-extracted patches)."""
    patches_dir = scene_dir / _sfx("patches", sensor)
    samples = []
    for patch_key, meta in patch_meta.items():
        oid = meta['obj_id']
        if obj_ids_filter and oid not in obj_ids_filter:
            continue

        rgb_path = patches_dir / "rgb_masked" / f"{patch_key}.png"
        if not rgb_path.exists():
            rgb_path = patches_dir / "rgb" / f"{patch_key}.png"
        if not rgb_path.exists():
            continue

        mask_path = patches_dir / "mask_visib" / f"{patch_key}.png"
        if not mask_path.exists():
            mask_path = None

        K = np.array(meta['cam_K'], dtype=np.float64).reshape(3, 3)
        R = np.array(meta['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
        t = np.array(meta['cam_t_m2c'], dtype=np.float64).reshape(3, 1)

        samples.append({
            'frame_id': meta['frame_id'],
            'K': K,
            'rgb_path': rgb_path,
            'instances': [{
                'inst_idx': meta['instance_idx'],
                'obj_id': oid,
                'R': R,
                't': t,
                'mask_path': mask_path,
                'visibility': meta.get('visibility', -1.0),
            }],
            'is_patch': True,
            'patch_key': patch_key,
        })

    samples.sort(key=lambda s: (s['frame_id'], s['instances'][0]['inst_idx']))
    return samples


# --- per-frame rendering -----------------------------------------------------

def render_frame(sample, model_cache, keypoints_cache, valid3d_cache, bop_root, args):
    """Render one annotated frame. Returns BGR image."""
    img = cv2.imread(str(sample['rgb_path']))
    if img is None:
        print(f"  [skip] cannot read {sample['rgb_path']}")
        return None

    K = sample['K']
    is_patch = sample.get('is_patch', False)

    for inst in sample['instances']:
        oid = inst['obj_id']
        R, t = inst['R'], inst['t']
        color = _color(inst['inst_idx'])

        # Mask overlay
        img = draw_mask_overlay(img, inst['mask_path'], color)

        # Projected keypoints
        kp = keypoints_cache.get(oid)
        if kp is not None:
            pts_2d, valid = project_pts(kp, K, R, t)
            img = draw_points(img, pts_2d, valid, (0, 255, 0), radius=5)

        # Projected Valid3D (subsample to keep it readable)
        v3d = valid3d_cache.get(oid)
        if v3d is not None:
            stride = max(1, len(v3d) // 300)
            pts_2d, valid = project_pts(v3d[::stride], K, R, t)
            img = draw_points(img, pts_2d, valid, (255, 200, 0), radius=2)

        # Model point cloud outline (very sparse, just for context)
        mesh = model_cache.get(oid)
        if mesh is not None and args.show_model:
            pts = mesh['pts']
            stride = max(1, len(pts) // 500)
            pts_2d, valid = project_pts(pts[::stride], K, R, t)
            img = draw_points(img, pts_2d, valid, color, radius=1)

        # Label at centroid of projected keypoints (or image center for patches)
        label_pt = (10 + inst['inst_idx'] * 130, 20)
        vis_str = f"{inst['visibility']:.2f}" if inst['visibility'] >= 0 else "?"
        draw_label(img, f"obj{oid} vis={vis_str}", label_pt, tuple(reversed(color)))

    # Frame header
    prefix = "patch" if is_patch else f"frame {sample['frame_id']:06d}"
    cv2.putText(img, prefix, (5, img.shape[0] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# --- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bop_root", required=True, help="BOP dataset root")
    parser.add_argument("--obj_id", type=int, default=None,
                        help="Filter to a single obj_id (default: show all objects with models)")
    parser.add_argument("--sensor", default=None,
                        help="Sensor name suffix (None = standard BOP without suffix)")
    parser.add_argument("--scene_ids", nargs="+", default=None,
                        help="Scene IDs to visualize (default: first scene found)")
    parser.add_argument("--split", default="test",
                        help="Dataset split directory (default: test)")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Max frames to visualize per scene (default: 10)")
    parser.add_argument("--patches", action="store_true",
                        help="Load pre-extracted patches instead of full-res images")
    parser.add_argument("--show_model", action="store_true",
                        help="Overlay sparse model point cloud (slow)")
    parser.add_argument("--output_dir", default="results/annotation_viz",
                        help="Output directory (default: results/annotation_viz)")
    args = parser.parse_args()

    bop_root = Path(args.bop_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover which obj_ids have models
    models_dir = bop_root / "models"
    available_obj_ids = set()
    if models_dir.exists():
        for f in models_dir.glob("obj_*.ply"):
            try:
                oid = int(f.stem.split("_")[-1])
                available_obj_ids.add(oid)
            except ValueError:
                pass
        for f in models_dir.glob("obj*.ply"):
            try:
                oid = int(f.stem.replace("obj", ""))
                available_obj_ids.add(oid)
            except ValueError:
                pass

    obj_ids_filter = {args.obj_id} if args.obj_id else None

    # Pre-load models, keypoints, Valid3D for all relevant obj_ids
    relevant_ids = obj_ids_filter if obj_ids_filter else available_obj_ids
    model_cache, keypoints_cache, valid3d_cache = {}, {}, {}

    for oid in sorted(relevant_ids):
        model_file = find_model_file(bop_root, oid)
        if model_file:
            model_cache[oid] = load_ply(str(model_file))
            print(f"  [model] obj{oid}: {model_file.name} ({len(model_cache[oid]['pts'])} pts)")

        kp_file = find_keypoints_file(bop_root, oid)
        if kp_file:
            keypoints_cache[oid] = np.loadtxt(kp_file)
            print(f"  [kp]    obj{oid}: {kp_file.name} ({len(keypoints_cache[oid])} kpts)")

        v3d_file = find_valid3d_file(bop_root, oid)
        if v3d_file:
            valid3d_cache[oid] = np.loadtxt(v3d_file)
            print(f"  [v3d]   obj{oid}: {v3d_file.name} ({len(valid3d_cache[oid])} pts)")

    # Discover scene dirs
    split_dir = bop_root / args.split
    if not split_dir.exists():
        split_dir = bop_root

    if args.scene_ids:
        scene_dirs = [split_dir / sid for sid in args.scene_ids]
    else:
        scene_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if not scene_dirs:
            print(f"No numeric scene directories found under {split_dir}")
            return
        scene_dirs = scene_dirs[:1]  # default: first scene only
        print(f"[viz] No --scene_ids given, visualizing scene: {scene_dirs[0].name}")

    total_saved = 0

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_out = output_dir / scene_id
        scene_out.mkdir(exist_ok=True)

        print(f"\n[scene {scene_id}] {scene_dir}")

        # ---- patch mode ----
        if args.patches:
            patch_meta_path = scene_dir / f"{_sfx('patch_meta', args.sensor)}.json"
            if not patch_meta_path.exists():
                print(f"  [skip] patch_meta not found: {patch_meta_path}")
                continue
            with open(patch_meta_path) as f:
                patch_meta = json.load(f)
            samples = build_patch_samples(scene_dir, patch_meta, args.sensor, obj_ids_filter)
            print(f"  Patch samples: {len(samples)}")

        # ---- full-res mode ----
        else:
            gt_path = scene_dir / f"{_sfx('scene_gt', args.sensor)}.json"
            cam_path = scene_dir / f"{_sfx('scene_camera', args.sensor)}.json"
            if not gt_path.exists() or not cam_path.exists():
                print(f"  [skip] {gt_path.name} or {cam_path.name} not found")
                continue
            with open(gt_path) as f:
                scene_gt = json.load(f)
            with open(cam_path) as f:
                scene_camera = json.load(f)

            gt_info_path = scene_dir / f"{_sfx('scene_gt_info', args.sensor)}.json"
            scene_gt_info = None
            if gt_info_path.exists():
                with open(gt_info_path) as f:
                    scene_gt_info = json.load(f)

            samples = build_fullres_samples(
                scene_dir, scene_gt, scene_camera, scene_gt_info,
                args.sensor, obj_ids_filter)
            print(f"  Full-res frames: {len(samples)} (with matching obj_id)")

        if not samples:
            print("  [skip] no matching samples")
            continue

        # Subsample evenly
        stride = max(1, len(samples) // args.num_frames)
        selected = samples[::stride][:args.num_frames]
        print(f"  Rendering {len(selected)} frame(s)...")

        for sample in selected:
            img = render_frame(sample, model_cache, keypoints_cache, valid3d_cache, bop_root, args)
            if img is None:
                continue

            if args.patches:
                fname = f"patch_{sample['patch_key']}.jpg"
            else:
                fname = f"frame_{sample['frame_id']:06d}.jpg"

            out_path = scene_out / fname
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"    saved {out_path}")
            total_saved += 1

    print(f"\n[done] {total_saved} image(s) saved to {output_dir}")


if __name__ == "__main__":
    main()
