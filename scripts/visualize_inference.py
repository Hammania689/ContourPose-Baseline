"""
Visualize ContourPose inference results on test frames.

Produces per-frame 3-panel figures:
  [Input + GT pose (green)]  [Pred heatmap + keypoints]  [Pred contour + pred pose (red)]

And per-object scene-grid figures that stack all test scenes for that object
into a single image (one row per scene, N frames wide) — useful for a systematic
sanity check across the full dataset.

Units note: mesh_model["pts"] is in meters; pose translation from gt.yml is in
meters; project() receives both in meters → correct pixel projections.

Usage
-----
All objects, all scenes — saves alongside an existing evaluation run:
  python scripts/visualize_inference.py \\
      --all_objects \\
      --results_dir results/rtless/authors_checkpoints/260609_0543_pecp \\
      --model_dir_root model/paper_checkpoints \\
      --data_path /data/Datasets/ContourPose \\
      --n_frames 4

  → results/rtless/authors_checkpoints/260609_0543_pecp/visualizations/{obj}/

Single object, all its scenes:
  python scripts/visualize_inference.py \\
      --class_type obj21 \\
      --results_dir results/rtless/authors_checkpoints/260609_0543_pecp \\
      --model_dir model/paper_checkpoints/obj21 \\
      --data_path /data/Datasets/ContourPose

Two-scene comparison for a specific object (manuscript figure):
  python scripts/visualize_inference.py \\
      --class_type obj21 \\
      --compare_scenes 5:2 29:0 \\
      --results_dir results/rtless/authors_checkpoints/260609_0543_pecp \\
      --model_dir model/paper_checkpoints/obj21 \\
      --data_path /data/Datasets/ContourPose \\
      --n_frames 3

Single scene:
  python scripts/visualize_inference.py \\
      --class_type obj21 --scene 5 --index 2 \\
      --out_dir /some/custom/path \\
      --frame_ids 50 150 300
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.Dataset import MyDataset
from eval_legacy_all_scenes import OBJECT_SCENES, ALL_OBJECTS
from network import ContourPose
from utils.utils import load_ply, project

# Objects that use ITERATIVE RANSAC in eval.py:pnp()
_ITERATIVE_OBJECTS = {"obj2", "obj32"}

MEAN = np.array([0.419, 0.427, 0.424], dtype=np.float32)
STD  = np.array([0.184, 0.206, 0.197], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model + object data
# ---------------------------------------------------------------------------

def load_model(class_type, model_dir, device):
    keypoints = np.loadtxt(os.path.join("keypoints", f"{class_type}.txt"))
    model = ContourPose(heatmap_dim=keypoints.shape[0])
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)

    pkls = [int(f.split(".")[0]) for f in os.listdir(model_dir) if f.endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError(f"No .pkl files in {model_dir}")
    epoch = max(pkls)
    ckpt_path = os.path.join(model_dir, f"{epoch}.pkl")
    print(f"  Checkpoint: {ckpt_path}  (epoch {epoch})")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
    model.eval()
    return model


def load_object_data(class_type):
    """Load keypoints and mesh once per object; reuse across scenes."""
    keypoints_3d = np.loadtxt(os.path.join("keypoints", f"{class_type}.txt"))
    mesh = load_ply(os.path.join("cad", f"{class_type}.ply"))
    pts_3d = mesh["pts"]  # meters — consistent with pose translation units
    return keypoints_3d, pts_3d


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def extract_keypoints_2d(heatmap_tensor):
    """Match eval.py:map_2_points() exactly. Returns [K, 2] numpy (x, y)."""
    flat = heatmap_tensor.view(heatmap_tensor.shape[0], -1)
    max_idx = torch.argmax(flat, dim=1)
    width = heatmap_tensor.shape[2]
    row = (max_idx / width).int().cpu().numpy()
    col = (max_idx % width).cpu().numpy()
    return np.stack([col, row], axis=1).astype(np.float64)  # [K, 2] as (x, y)


def pnp(points_3d, points_2d, K, class_type):
    dist = np.zeros((5, 1), dtype=np.float64)
    p3d = np.ascontiguousarray(points_3d.astype(np.float64))
    p2d = np.ascontiguousarray(points_2d.astype(np.float64))
    Kd  = K.astype(np.float64)
    if p3d.shape[0] < 5:
        _, rvec, tvec = cv2.solvePnP(p3d, p2d, Kd, dist, flags=cv2.SOLVEPNP_EPNP)
    elif class_type in _ITERATIVE_OBJECTS:
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            p3d, p2d, Kd, dist, iterationsCount=1000,
            reprojectionError=5, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            p3d, p2d, Kd, dist, iterationsCount=1000,
            reprojectionError=5, flags=cv2.SOLVEPNP_EPNP)
    R, _ = cv2.Rodrigues(rvec)
    return np.concatenate([R, tvec], axis=-1)  # [3, 4]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denorm(img_tensor):
    """[C, H, W] tensor → uint8 HWC RGB (C-contiguous, safe for OpenCV)."""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * STD + MEAN
    return np.ascontiguousarray(np.clip(img * 255, 0, 255).astype(np.uint8))


def draw_keypoints(img_rgb, kpts_2d, color=(255, 255, 0), radius=4):
    out = img_rgb.copy()
    for x, y in kpts_2d:
        x, y = int(round(x)), int(round(y))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def overlay_pts_3d(img_rgb, pts_3d, K, pose, color, point_size=1, alpha=0.65):
    """Project 3D pts onto img_rgb and alpha-blend. pts_3d and pose in same units."""
    pts_2d = project(pts_3d, K, pose)
    H, W = img_rgb.shape[:2]
    canvas = np.zeros_like(img_rgb)
    for x, y in pts_2d:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(canvas, (xi, yi), point_size, color, -1, cv2.LINE_AA)
    mask = (canvas.sum(axis=2) > 0)[..., None].astype(np.float32)
    return (alpha * canvas * mask + (1 - alpha * mask) * img_rgb).astype(np.uint8)


def make_heatmap_vis(heatmap_tensor):
    """Sum keypoint heatmaps → JET-coloured HWC uint8 image."""
    hm = heatmap_tensor.sum(dim=0).cpu().numpy()
    hm = (hm / (hm.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def make_contour_overlay(img_rgb, contour_logits, threshold=0.0):
    """Burn predicted contour pixels orange onto the image."""
    binary = (contour_logits.cpu().numpy() >= threshold)
    overlay = img_rgb.copy()
    overlay[binary] = [255, 80, 0]
    return cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)


def make_frame_panel(img_rgb, gt_pose, pred_pose, heatmap, contour, pts_3d, K):
    """
    3-column panel for one frame:
      col 0 — input image + GT CAD overlay (green)
      col 1 — predicted heatmap (JET) + predicted keypoints (yellow dots)
      col 2 — predicted contour overlay (orange) + predicted CAD (red)
    """
    col0 = overlay_pts_3d(img_rgb, pts_3d, K, gt_pose, color=(0, 220, 0), point_size=1)

    hm_vis = make_heatmap_vis(heatmap)
    pred_kpts2d = extract_keypoints_2d(heatmap)
    col1 = draw_keypoints(hm_vis, pred_kpts2d, color=(255, 255, 0), radius=5)

    col2 = make_contour_overlay(img_rgb, contour)
    if pred_pose is not None:
        col2 = overlay_pts_3d(col2, pts_3d, K, pred_pose, color=(255, 50, 50), point_size=1)

    return np.concatenate([col0, col1, col2], axis=1)


def add_row_header(img, text, bar_height=36, font_scale=0.65, thickness=2):
    H, W = img.shape[:2]
    bar = np.full((bar_height, W, 3), 40, dtype=np.uint8)
    cv2.putText(bar, text, (8, bar_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return np.vstack([bar, img])


# ---------------------------------------------------------------------------
# Run inference on one scene → list of (frame_idx, panel_img)
# ---------------------------------------------------------------------------

def run_scene(class_type, scene, index, model, data_path, device,
              keypoints_3d, pts_3d, frame_ids=None, n_frames=6):
    dataset = MyDataset(data_path, class_type, is_train=False, scene=scene, index=index)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    total = len(dataset)
    if frame_ids is None:
        step = max(1, total // n_frames)
        frame_ids = list(range(0, total, step))[:n_frames]

    frame_id_set = set(frame_ids)
    panels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i not in frame_id_set:
                continue
            img_t, _, K_np, pose_np = batch
            img_t   = img_t.to(device)
            K       = K_np[0].numpy().astype(np.float64)
            gt_pose = pose_np[0].numpy()

            pred_heatmap, pred_contour = model(img_t)

            pred_kpts2d = extract_keypoints_2d(pred_heatmap[0])
            try:
                pred_pose = pnp(keypoints_3d, pred_kpts2d, K, class_type)
            except Exception as e:
                print(f"    [frame {i}] PnP failed: {e}")
                pred_pose = None

            panel = make_frame_panel(
                denorm(img_t[0]), gt_pose, pred_pose,
                pred_heatmap[0], pred_contour[0, 0],
                pts_3d, K,
            )
            panels.append((i, panel))

            if len(panels) >= n_frames:
                break

    return panels


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_panels(panels, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    for frame_idx, panel in panels:
        path = os.path.join(out_dir, f"{prefix}_frame{frame_idx:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"    saved: {path}")


def make_scene_grid(scene_rows, out_path):
    """
    Stack N scene rows into one figure.

    Args:
        scene_rows: list of (label_str, [(frame_idx, panel_img), ...])
        out_path: where to write the PNG
    """
    if not scene_rows:
        return

    # Determine uniform panel width across all rows
    all_panels = [p for _, panels in scene_rows for _, p in panels]
    if not all_panels:
        return
    panel_W = max(p.shape[1] for p in all_panels)

    def pad_panel(p):
        if p.shape[1] < panel_W:
            pad = np.zeros((p.shape[0], panel_W - p.shape[1], 3), dtype=np.uint8)
            p = np.hstack([p, pad])
        return p

    rows = []
    for label, panels in scene_rows:
        if not panels:
            continue
        row_img = np.hstack([pad_panel(p) for _, p in panels])
        row_img = add_row_header(row_img, label)
        rows.append(row_img)

    if not rows:
        return

    # Pad rows to uniform width
    max_W = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_W:
            pad = np.zeros((r.shape[0], max_W - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    figure = np.vstack(padded)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    print(f"  Grid figure: {out_path}  ({figure.shape[1]}×{figure.shape[0]} px)")


# ---------------------------------------------------------------------------
# Per-object evaluation
# ---------------------------------------------------------------------------

def visualize_object(class_type, model_dir, data_path, device,
                     out_dir, scenes=None, frame_ids=None, n_frames=6):
    """
    Run visualization for one object across all (or specified) scenes.

    Saves:
      - Individual frame panels per scene
      - A scene-grid figure stacking all scenes (if ≥1 scene present)
    """
    print(f"\n{'='*60}")
    print(f"  {class_type}")
    print(f"{'='*60}")

    obj_scenes = scenes or OBJECT_SCENES[class_type]
    obj_out    = os.path.join(out_dir, class_type)
    os.makedirs(obj_out, exist_ok=True)

    model        = load_model(class_type, model_dir, device)
    kpts3d, pts3d = load_object_data(class_type)

    scene_rows = []
    for scene, index in obj_scenes:
        scene_dir = os.path.join(data_path, "test", f"scene{scene}")
        if not os.path.isdir(scene_dir):
            print(f"  [SKIP] scene {scene} not on disk")
            continue

        print(f"\n  scene={scene}  slot={index}")
        panels = run_scene(class_type, scene, index, model, data_path, device,
                           kpts3d, pts3d, frame_ids=frame_ids, n_frames=n_frames)

        save_panels(panels, obj_out, f"{class_type}_scene{scene}")

        label = (f"{class_type}  scene {scene}  [slot {index}]"
                 f"  |  GT (green) · heatmap+kpts · contour+pred (red)")
        scene_rows.append((label, panels))

    if scene_rows:
        grid_path = os.path.join(obj_out, f"{class_type}_all_scenes.png")
        make_scene_grid(scene_rows, grid_path)


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def resolve_out_dir(args, parser):
    if args.out_dir:
        return args.out_dir
    if args.results_dir:
        if not os.path.isdir(args.results_dir):
            parser.error(f"--results_dir does not exist: {args.results_dir}")
        out = os.path.join(args.results_dir, "visualizations")
        print(f"  Output root: {out}")
        return out
    return os.path.join("results", "viz")


def parse_scene_index(s):
    parts = s.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected 'scene:index', got '{s}'")
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Object selection (mutually exclusive group)
    obj_group = p.add_mutually_exclusive_group(required=True)
    obj_group.add_argument("--class_type",  help="Single object, e.g. obj21")
    obj_group.add_argument("--all_objects", action="store_true",
                           help="Run all 10 objects")

    # Scene selection (ignored when --all_objects)
    p.add_argument("--scene",  type=int, help="Single test scene number")
    p.add_argument("--index",  type=int, help="Slot index in scene (sceneObjs.yml)")
    p.add_argument("--compare_scenes", nargs=2, metavar="SCENE:INDEX",
                   help="Two-scene comparison, e.g. --compare_scenes 5:2 29:0")

    # Checkpoints
    p.add_argument("--model_dir",      default=None,
                   help="Checkpoint dir for a single object "
                        "(default: model/paper_checkpoints/{class_type})")
    p.add_argument("--model_dir_root", default="model/paper_checkpoints",
                   help="Root containing per-object checkpoint dirs "
                        "(used by --all_objects; default: model/paper_checkpoints)")

    # Data + output
    p.add_argument("--data_path",   default="/data/Datasets/ContourPose")
    p.add_argument("--results_dir", default=None,
                   help="Existing evaluation run dir; visualizations go to "
                        "{results_dir}/visualizations/")
    p.add_argument("--out_dir",     default=None,
                   help="Explicit output root (overrides --results_dir)")

    # Frame sampling
    p.add_argument("--frame_ids", type=int, nargs="+",
                   help="Specific 0-based frame indices to visualize")
    p.add_argument("--n_frames",  type=int, default=4,
                   help="Frames to sample per scene when --frame_ids not given (default: 4)")
    p.add_argument("--gpu",       type=int, default=0)

    args = p.parse_args()
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = resolve_out_dir(args, p)

    # ------------------------------------------------------------------
    # --all_objects: loop over every object, save into per-object subdirs
    # ------------------------------------------------------------------
    if args.all_objects:
        for class_type in ALL_OBJECTS:
            model_dir = os.path.join(args.model_dir_root, class_type)
            visualize_object(class_type, model_dir, args.data_path, device,
                             out_dir, frame_ids=args.frame_ids, n_frames=args.n_frames)
        return

    # ------------------------------------------------------------------
    # Single-object modes
    # ------------------------------------------------------------------
    class_type = args.class_type
    model_dir  = args.model_dir or os.path.join("model", "paper_checkpoints", class_type)
    obj_out    = os.path.join(out_dir, class_type)

    # Two-scene comparison (--compare_scenes)
    if args.compare_scenes:
        scene_a, index_a = parse_scene_index(args.compare_scenes[0])
        scene_b, index_b = parse_scene_index(args.compare_scenes[1])

        model         = load_model(class_type, model_dir, device)
        kpts3d, pts3d = load_object_data(class_type)

        print(f"\n[{class_type}] scene {scene_a} (slot {index_a})")
        panels_a = run_scene(class_type, scene_a, index_a, model, args.data_path, device,
                             kpts3d, pts3d, frame_ids=args.frame_ids, n_frames=args.n_frames)

        print(f"\n[{class_type}] scene {scene_b} (slot {index_b})")
        panels_b = run_scene(class_type, scene_b, index_b, model, args.data_path, device,
                             kpts3d, pts3d, frame_ids=args.frame_ids, n_frames=args.n_frames)

        label_a = (f"{class_type}  scene {scene_a}  [slot {index_a}]"
                   f"  |  GT (green) · heatmap+kpts · contour+pred (red)")
        label_b = (f"{class_type}  scene {scene_b}  [slot {index_b}]"
                   f"  |  GT (green) · heatmap+kpts · contour+pred (red)")

        fig_path = os.path.join(obj_out,
                                f"{class_type}_scene{scene_a}_vs_scene{scene_b}.png")
        make_scene_grid([(label_a, panels_a), (label_b, panels_b)], fig_path)
        save_panels(panels_a, obj_out, f"{class_type}_scene{scene_a}")
        save_panels(panels_b, obj_out, f"{class_type}_scene{scene_b}")
        return

    # All scenes for one object (default single-object behaviour)
    if args.scene is None:
        visualize_object(class_type, model_dir, args.data_path, device,
                         out_dir, frame_ids=args.frame_ids, n_frames=args.n_frames)
        return

    # Single scene
    if args.index is None:
        p.error("--index is required when --scene is given")
    model         = load_model(class_type, model_dir, device)
    kpts3d, pts3d = load_object_data(class_type)
    print(f"\n[{class_type}] scene {args.scene} (slot {args.index})")
    panels = run_scene(class_type, args.scene, args.index, model, args.data_path, device,
                       kpts3d, pts3d, frame_ids=args.frame_ids, n_frames=args.n_frames)
    label  = (f"{class_type}  scene {args.scene}  [slot {args.index}]"
              f"  |  GT (green) · heatmap+kpts · contour+pred (red)")
    make_scene_grid([(label, panels)], os.path.join(obj_out, f"{class_type}_scene{args.scene}.png"))
    save_panels(panels, obj_out, f"{class_type}_scene{args.scene}")


if __name__ == "__main__":
    main()
