import torch
import tqdm
import numpy as np
import os
import os.path as osp
from utils.utils import load_ply, project, load_camera_intrinsics, get_K_override, min_symmetry_rotation_error
import cv2
from scipy import spatial
from scipy.spatial.transform import Rotation as ScipyRot
from itertools import combinations
from random import choice
import math
import time

# Import default config (used for ContourPose format)
from config import rtDic, diameters

cuda = torch.cuda.is_available()


def _load_bop_eval_config(bop_root, obj_id):
    """
    Load evaluation configuration from BOP dataset.

    Returns:
        dict with keys: diameter, rtDic (symmetry transform), symmetric_objects
    """
    from dataset.bop_config import get_contourpose_config_from_bop

    bop_config = get_contourpose_config_from_bop(bop_root)
    obj_key = f"obj{obj_id}"

    return {
        "diameter": bop_config["diameters"].get(obj_key, 0.0),
        "rtDic": bop_config["rtDic"],
        "symmetric_objects": bop_config["symmetric_objects"],
    }


class evaluator:
    def __init__(self, args, model, test_loader, device, samples_metadata=None):
        self.args = args
        self.model = model
        self.device = device
        self.data_loader = test_loader
        self.samples_metadata = samples_metadata  # Per-instance metadata for BOP CSV

        # Determine format (BOP or ContourPose)
        # legacy_test forces ContourPose format even when bop_root is set
        legacy_test = getattr(args, 'legacy_test', False)
        self.use_bop = getattr(args, 'bop_root', None) is not None and not legacy_test

        if self.use_bop:
            self._init_bop_format(args)
        else:
            self._init_contourpose_format(args)

        # Load K override if specified
        self.K_override = None
        camera_intrinsics_path = getattr(args, 'camera_intrinsics_path', None)
        if camera_intrinsics_path is not None:
            self._load_camera_intrinsics(camera_intrinsics_path)

        # Bbox-ratio rescaling: match test object-to-image ratio to training
        self.fixed_scale = getattr(args, 'fixed_scale', None)
        self.train_bbox_ratio = None
        if self.fixed_scale is not None:
            print(f"[Evaluator] Using fixed scale: {self.fixed_scale:.3f}x")
        else:
            train_mask_dir = getattr(args, 'train_mask_dir', None)
            if train_mask_dir is not None:
                self.train_bbox_ratio = self._compute_mean_bbox_ratio(train_mask_dir)
                print(f"[Evaluator] Training mean bbox ratio: {self.train_bbox_ratio:.4f}")

        # Initialize metric storage
        self.index = 1
        self.threshold = args.threshold
        self.proj_2d = []
        self.proj_2d_mean = []
        self.add = []
        self.x_error_all = []
        self.y_error_all = []
        self.z_error_all = []
        self.alpha_error_all = []
        self.beta_error_all = []
        self.gama_error_all = []

        # Visualization data collection
        self._vis_data = []
        self._num_vis = getattr(args, 'num_vis', 20)
        self._save_vis = getattr(args, 'save_vis', False)

        # PECP (Pose Evaluation using Contour as a Priori) setup
        self.use_pecp = getattr(args, 'use_pecp', True)
        if self.use_pecp and self.valid_3d.shape[0] == 0:
            print("[Evaluator] WARNING: PECP enabled but Valid3D is empty — disabling PECP")
            self.use_pecp = False
        print(f"[Evaluator] PECP: {'enabled' if self.use_pecp else 'disabled'}")

        # Build all C(K,4) keypoint index combinations for PECP Phase 1
        example = ""
        for i in range(0, self.keyponits.shape[0]):
            if i < 10:
                example = example + str(i)
            if i >= 10:
                example = example + chr(97 + i - 10)
        self.list_all = list(combinations(example, 4))

    def _init_contourpose_format(self, args):
        """Initialize for ContourPose dataset format.

        Uses args.data_root (if set) for keypoints/ and Valid3D/,
        falling back to os.getcwd() for legacy compatibility.

        CAD model is loaded from data_path (test data root) first,
        then data_root, then cwd.

        Automatically detects unit mismatches between keypoints and CAD
        model (e.g. keypoints in mm but CAD/poses in meters) and converts
        keypoints to match.
        """
        data_root = getattr(args, 'data_root', None) or os.getcwd()
        data_path = getattr(args, 'data_path', os.getcwd())

        # CAD model: try data_path first (original data), then data_root, then cwd
        cad_file = None
        for cad_base in [data_path, data_root, os.getcwd()]:
            candidate = osp.join(cad_base, "cad", "{}.ply".format(args.class_type))
            if osp.exists(candidate):
                cad_file = candidate
                self.cad_path = osp.join(cad_base, "cad")
                break
        if cad_file is None:
            raise FileNotFoundError(
                f"CAD model not found: cad/{args.class_type}.ply "
                f"(searched: {data_path}, {data_root}, {os.getcwd()})"
            )
        self.mesh_model = load_ply(cad_file)

        # Keypoints and Valid3D from data_root
        self.keyponits = np.loadtxt(
            os.path.join(data_root, "keypoints/{}.txt".format(args.class_type))
        )
        self.pts_3d = self.mesh_model["pts"]
        self.valid_3d = np.loadtxt(
            osp.join(data_root, "Valid3D/{}.txt".format(args.class_type))
        )

        # Auto-detect unit mismatch: if keypoints are ~1000x larger than CAD,
        # they're in mm while CAD/poses are in meters. Convert keypoints to match.
        cad_extent = self.pts_3d.max() - self.pts_3d.min()
        kp_extent = self.keyponits.max() - self.keyponits.min()
        if cad_extent > 0 and kp_extent / cad_extent > 100:
            scale = cad_extent / kp_extent
            # Round to nearest power of 10 (expect 0.001 for mm→m)
            import math
            scale = 10 ** round(math.log10(scale))
            print(f"[Evaluator] Unit mismatch detected: keypoints {scale:.0e}x larger than CAD model")
            print(f"[Evaluator] Scaling keypoints/Valid3D by {scale} to match test data units")
            self.keyponits = self.keyponits * scale
            self.valid_3d = self.valid_3d * scale

        self.diameter = diameters[args.class_type] / 1000.0
        self.rtDic = rtDic
        self.symmetric_objects = [
            "obj1", "obj2", "obj5", "obj14", "obj17", "obj18",
            "obj24", "obj26", "obj29", "obj33"
        ]
        self.obj_key = args.class_type
        # Default: identity-only (no symmetry info for legacy format)
        self.all_symmetry_transforms = [np.eye(4)]
        self.continuous_symmetry_axes = []

    def _init_bop_format(self, args):
        """Initialize for BOP dataset format."""
        # Load BOP config
        bop_config = _load_bop_eval_config(args.bop_root, args.obj_id)
        self.obj_key = f"obj{args.obj_id}"

        # Load symmetry info for symmetry-aware rotation error
        from dataset.bop_config import get_all_symmetry_transforms, get_continuous_symmetry_axes
        all_sym = get_all_symmetry_transforms(args.bop_root)
        self.all_symmetry_transforms = all_sym.get(self.obj_key, [np.eye(4)])
        cont_axes = get_continuous_symmetry_axes(args.bop_root)
        self.continuous_symmetry_axes = cont_axes.get(self.obj_key, [])

        # CAD model — search BOP models/ first, then fall back to repo-root cad/.
        # PLYs in this repo are metre-scale; scale to mm to match BOP cam_t_m2c
        # and the mm-scale keypoints/Valid3D loaded above.
        self.cad_path = osp.join(args.bop_root, "models")
        class_type = getattr(args, 'class_type', f"obj{args.obj_id}")
        cad_candidates = [
            osp.join(self.cad_path, f"obj_{args.obj_id:06d}.ply"),
            osp.join(self.cad_path, f"obj{args.obj_id}.ply"),
            osp.join("cad", f"{class_type}.ply"),
            osp.join("cad", f"obj{args.obj_id}.ply"),
        ]
        cad_file = next((p for p in cad_candidates if osp.exists(p)), None)
        if cad_file is None:
            raise FileNotFoundError(
                f"CAD model not found for obj_id={args.obj_id}. Searched: {cad_candidates}"
            )

        self.mesh_model = load_ply(cad_file)
        pts = self.mesh_model["pts"]
        if np.max(np.abs(pts)) <= 1.0:
            print(f"[Evaluator] CAD {cad_file} appears metre-scale; scaling to mm")
            pts = pts * 1000.0
            self.mesh_model["pts"] = pts
        assert np.max(np.abs(pts)) > 1.0, (
            f"[Evaluator] CAD points in {cad_file} still look like metres after scaling "
            f"(max |xyz| = {np.max(np.abs(pts)):.4f}); expected mm."
        )
        self.pts_3d = pts

        # Keypoints (from keypoints_dir argument, resolve under bop_root).
        # BOP path is mm-native (matches cam_t_m2c); prefer *_mm.txt files and
        # assert on load so silent unit mismatches can't reach pnp/PECP.
        keypoints_dir = getattr(args, 'keypoints_dir', 'keypoints')
        class_type = getattr(args, 'class_type', '')
        kp_bases = [osp.join(args.bop_root, keypoints_dir), keypoints_dir]
        kp_names = [f"obj_{args.obj_id:06d}_mm.txt", f"obj{args.obj_id}_mm.txt",
                    f"{args.obj_id}_mm.txt", f"{class_type}_mm.txt",
                    f"obj_{args.obj_id:06d}.txt", f"obj{args.obj_id}.txt",
                    f"{args.obj_id}.txt", f"{class_type}.txt"]
        keypoint_candidates = [osp.join(b, n) for b in kp_bases for n in kp_names]
        for kp_path in keypoint_candidates:
            if osp.exists(kp_path):
                self.keyponits = np.loadtxt(kp_path)
                assert np.max(np.abs(self.keyponits)) > 1.0, (
                    f"[Evaluator] Keypoints in {kp_path} look like metres "
                    f"(max |xyz| = {np.max(np.abs(self.keyponits)):.4f}); expected mm. "
                    f"Regenerate via scripts/convert_keypoints_to_mm.py."
                )
                break
        else:
            raise FileNotFoundError(
                f"Keypoints not found for obj_id={args.obj_id}. "
                f"Searched: {keypoint_candidates}"
            )

        # Valid3D (from valid3d_dir argument, resolve under bop_root). Same
        # mm-native convention — used in project() against BOP mm-scale poses.
        valid3d_dir = getattr(args, 'valid3d_dir', 'Valid3D')
        v3d_bases = [osp.join(args.bop_root, valid3d_dir), valid3d_dir]
        v3d_names = [f"obj_{args.obj_id:06d}_mm.txt", f"obj{args.obj_id}_mm.txt",
                     f"{args.obj_id}_mm.txt", f"{class_type}_mm.txt",
                     f"obj_{args.obj_id:06d}.txt", f"obj{args.obj_id}.txt",
                     f"{args.obj_id}.txt", f"{class_type}.txt"]
        valid3d_candidates = [osp.join(b, n) for b in v3d_bases for n in v3d_names]
        for v3d_path in valid3d_candidates:
            if osp.exists(v3d_path):
                self.valid_3d = np.loadtxt(v3d_path)
                assert np.max(np.abs(self.valid_3d)) > 1.0, (
                    f"[Evaluator] Valid3D in {v3d_path} look like metres "
                    f"(max |xyz| = {np.max(np.abs(self.valid_3d)):.4f}); expected mm. "
                    f"Regenerate via scripts/convert_keypoints_to_mm.py."
                )
                break
        else:
            # Valid3D is optional - use empty array if not found
            print(f"[Warning] Valid3D not found for obj_id={args.obj_id}, PECP disabled")
            self.valid_3d = np.array([]).reshape(0, 3)

        # Diameter in mm (consistent with BOP pts_3d/translation units)
        self.diameter = bop_config["diameter"]

        # Symmetry handling
        self.rtDic = bop_config["rtDic"]
        self.symmetric_objects = bop_config["symmetric_objects"]

        # BOP RTLESS metadata lacks symmetry annotations for some objects.
        # Supplement with the per-object 180° transforms from config.py (originally in meters).
        from config import rtDic as contourpose_rtDic
        if self.obj_key not in self.symmetric_objects and self.obj_key in contourpose_rtDic:
            rt_m = contourpose_rtDic[self.obj_key]
            if rt_m is not None:
                rt_mm = rt_m.astype(np.float64).copy()
                rt_mm[:3, 3] *= 1000.0  # m → mm to match BOP pose units
                self.all_symmetry_transforms.append(rt_mm)
                self.symmetric_objects = list(self.symmetric_objects) + [self.obj_key]
                print(f"[Evaluator] Supplemented symmetry for {self.obj_key} from ContourPose config")

    def _load_camera_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from file using shared utility."""
        print(f"[Evaluator] Loading camera intrinsics from: {intrinsics_path}")
        self.K_override = load_camera_intrinsics(intrinsics_path)
        print(f"[Evaluator] Loaded K_override: fx={self.K_override[0,0]:.1f}, fy={self.K_override[1,1]:.1f}, "
              f"cx={self.K_override[0,2]:.1f}, cy={self.K_override[1,2]:.1f}")

    @staticmethod
    def _compute_mean_bbox_ratio(mask_dir, max_samples=1000):
        """Compute mean bbox-area-to-image-area ratio from training masks.

        Args:
            mask_dir: directory containing training mask PNGs
            max_samples: number of masks to sample for estimation
        Returns:
            mean bbox ratio (float)
        """
        from pathlib import Path
        mask_dir = Path(mask_dir)
        files = sorted(mask_dir.glob("*.png"))[:max_samples]
        ratios = []
        for f in files:
            mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            bbox_w = xs.max() - xs.min() + 1
            bbox_h = ys.max() - ys.min() + 1
            h, w = mask.shape
            ratios.append((bbox_w * bbox_h) / (h * w))
        if not ratios:
            print(f"[Evaluator] WARNING: No valid masks found in {mask_dir}")
            return None
        mean_ratio = float(np.mean(ratios))
        print(f"[Evaluator] Computed training bbox ratio from {len(ratios)} masks")
        return mean_ratio

    @staticmethod
    def _compute_bbox_ratio_from_mask(mask_path):
        """Compute bbox-area-to-image-area ratio for a single test mask."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        h, w = mask.shape
        return (bbox_w * bbox_h) / (h * w)

    def _rescale_image_batch(self, img, sample_indices):
        """Rescale images so object bbox ratio matches training data.

        For each sample, loads the test mask, computes its bbox ratio, and
        scales the image so the object occupies the same fraction as in
        training. Scale = sqrt(train_bbox_ratio / test_bbox_ratio) applied
        as a zoom around the image center.

        Returns:
            img: rescaled image tensor (same shape, in-place)
            scales: per-sample scale factors [B] numpy array
        """
        import torch.nn.functional as F

        B = img.shape[0]
        scales = np.ones(B, dtype=np.float64)

        for i in range(B):
            idx = sample_indices[i]
            if idx >= len(self.samples_metadata):
                continue

            mask_path = self.samples_metadata[idx].get('mask_path')
            if mask_path is None:
                continue

            test_ratio = self._compute_bbox_ratio_from_mask(mask_path)
            if test_ratio is None or test_ratio < 1e-6:
                continue

            # Scale so that bbox occupies train_bbox_ratio of the patch
            # Linear dimension scale = sqrt(ratio_ratio)
            scale = np.sqrt(self.train_bbox_ratio / test_ratio)
            if abs(scale - 1.0) < 0.02:
                continue

            scales[i] = scale

            # grid_sample uses inverse mapping: to zoom IN by `scale`,
            # we sample from a smaller region (divide by scale)
            theta = torch.zeros(1, 2, 3, device=img.device, dtype=img.dtype)
            theta[0, 0, 0] = 1.0 / scale
            theta[0, 1, 1] = 1.0 / scale

            grid = F.affine_grid(theta, img[i:i+1].shape, align_corners=False)
            img[i:i+1] = F.grid_sample(img[i:i+1], grid, mode='bilinear',
                                        padding_mode='zeros', align_corners=False)

        return img, scales

    @staticmethod
    def _rescale_image_batch_fixed(img, scale):
        """Apply a fixed center-zoom scale to all images in the batch.

        Args:
            img: [B, C, H, W] tensor
            scale: fixed scale factor (> 1 = zoom in)
        Returns:
            img: rescaled tensor (same shape)
            scales: [B] numpy array filled with the fixed scale
        """
        import torch.nn.functional as F

        B = img.shape[0]
        scales = np.full(B, scale, dtype=np.float64)

        if abs(scale - 1.0) < 0.02:
            return img, scales

        theta = torch.zeros(1, 2, 3, device=img.device, dtype=img.dtype)
        theta[0, 0, 0] = 1.0 / scale
        theta[0, 1, 1] = 1.0 / scale

        # Same affine for all images — expand to batch
        theta = theta.expand(B, -1, -1)
        grid = F.affine_grid(theta, img.shape, align_corners=False)
        img = F.grid_sample(img, grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)

        return img, scales

    @staticmethod
    def _inverse_rescale_keypoints(kp, scale, img_size=256):
        """Map predicted keypoints from rescaled image space back to original.

        Args:
            kp: keypoints [N, 2] in rescaled image coordinates
            scale: scale factor that was applied (> 1 = zoomed in)
            img_size: patch size (assumes square)
        Returns:
            kp in original test image coordinates
        """
        center = img_size / 2.0
        return (kp - center) / scale + center

    def get_K(self, batch_K):
        """Get camera intrinsics matrix. Uses K_override if set, otherwise returns batch_K."""
        return get_K_override(self.K_override, batch_K)

    def _unpack_batch(self, data):
        """
        Unpack a batch from either DALI (dict) or PyTorch (tuple) loader.

        Returns:
            img, heatmap, K_batch, pose, edge_input — all as tensors on self.device
            edge_input is None when not available.
        """
        edge_input = None
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            # DALI format: list of [dict]
            d = data[0]
            img = d["images"]
            heatmap = d["heatmaps"]
            K_batch = d["K"]
            pose = d["pose"]
            if "edges_input" in d:
                edge_input = d["edges_input"]
        elif isinstance(data, dict):
            # DALI format: dict directly
            img = data["images"]
            heatmap = data["heatmaps"]
            K_batch = data["K"]
            pose = data["pose"]
            if "edges_input" in data:
                edge_input = data["edges_input"]
        else:
            # PyTorch DataLoader format: tuple
            img, heatmap, K_batch, pose = data

        if cuda:
            img = img.to(self.device)
            heatmap = heatmap.to(self.device)
            K_batch = K_batch.to(self.device)
            pose = pose.to(self.device)
            if edge_input is not None:
                edge_input = edge_input.to(self.device)

        return img, heatmap, K_batch, pose, edge_input

    def evaluate(self):
        self.model.eval()
        per_instance_results = []
        sample_idx = 0

        with torch.no_grad():
            for data in tqdm.tqdm(self.data_loader, leave=False, desc="val"):
                img, heatmap, K_batch, pose, edge_input = self._unpack_batch(data)

                # Use K_override if set, otherwise use batch K
                K = self.get_K(K_batch)

                # Stash unscaled images for visualization before rescaling
                img_unscaled = None
                if self._save_vis and len(self._vis_data) < self._num_vis:
                    img_unscaled = img.clone().cpu()

                # Rescale images
                intrinsic_scales = None
                if self.fixed_scale is not None:
                    # Apply uniform fixed scale to all images
                    img, intrinsic_scales = self._rescale_image_batch_fixed(img, self.fixed_scale)
                elif self.train_bbox_ratio is not None and self.samples_metadata is not None:
                    batch_size = img.shape[0]
                    batch_indices = list(range(sample_idx, sample_idx + batch_size))
                    img, intrinsic_scales = self._rescale_image_batch(img, batch_indices)

                t_start = time.time()
                if edge_input is not None:
                    pred_heatmap, pred_contour = self.model(img, edge=edge_input)
                else:
                    pred_heatmap, pred_contour = self.model(img)
                t_infer = time.time() - t_start

                keypoints_2d, predict_2d = self.map_2_points(heatmap, pred_heatmap)

                # Inverse-map predicted keypoints back to test camera space
                if intrinsic_scales is not None:
                    img_size = img.shape[-1]  # assumes square patches
                    for i in range(predict_2d.shape[0]):
                        if abs(intrinsic_scales[i] - 1.0) > 0.01:
                            pts = predict_2d[i].detach().cpu().numpy().reshape(-1, 2)
                            pts = self._inverse_rescale_keypoints(
                                pts, intrinsic_scales[i], img_size)
                            predict_2d[i] = torch.from_numpy(pts.reshape(
                                predict_2d[i].shape)).to(predict_2d.device)

                # Collect visualization data for first N samples
                if self._save_vis and len(self._vis_data) < self._num_vis:
                    remaining = self._num_vis - len(self._vis_data)
                    for i in range(min(img.shape[0], remaining)):
                        scale_i = intrinsic_scales[i] if intrinsic_scales is not None else 1.0
                        abs_idx = sample_idx + i
                        meta_i = {}
                        if self.samples_metadata and abs_idx < len(self.samples_metadata):
                            meta_i = self.samples_metadata[abs_idx]
                        self._vis_data.append({
                            'img_unscaled': img_unscaled[i] if img_unscaled is not None else img[i].cpu(),
                            'img_scaled': img[i].cpu(),
                            'pred_contour': pred_contour[i].cpu() if pred_contour is not None else None,
                            'K': (K[i] if K.dim() > 2 else K).cpu().numpy(),
                            'gt_pose': pose[i].cpu().numpy(),
                            'gt_kp': keypoints_2d[i].cpu().numpy() if i < keypoints_2d.shape[0] else None,
                            'pred_kp': predict_2d[i].cpu().numpy() if i < predict_2d.shape[0] else None,
                            'scale': scale_i,
                            'sample_idx': abs_idx,
                            'scene_id': str(meta_i.get('scene_id', '000000')),
                        })

                # Per-instance detailed metrics
                batch_size = keypoints_2d.shape[0]
                for i in range(batch_size):
                    scale_i = intrinsic_scales[i] if intrinsic_scales is not None else 1.0
                    result = self._evaluate_single_instance(
                        keypoints_2d[i], predict_2d[i], K[i] if K.dim() > 2 else K,
                        pose[i], heatmap[i], pred_heatmap[i],
                        pred_contour[i] if pred_contour is not None else None,
                        t_infer / batch_size,
                        sample_idx,
                        scale=scale_i,
                    )
                    per_instance_results.append(result)
                    sample_idx += 1

        # Aggregate metrics
        metrics = self._aggregate_metrics(per_instance_results)

        # Print summary
        n_total = metrics["num_instances"]
        n_pnp_failed = metrics["num_pnp_failed"]
        n_pnp_valid = n_total - n_pnp_failed
        pnp_rate = n_pnp_valid / n_total * 100 if n_total > 0 else 0

        print(f"\n{'─'*60}")
        print(f"  {self.obj_key} — All instances ({n_total})")
        print(f"{'─'*60}")
        print(f"  2D Projection: {metrics['proj_2d']:.4f}")
        print(f"  ADD:           {metrics['add']:.4f}")
        print(f"  PCK@1px:       {metrics['pck_1px']:.4f}")
        print(f"  PCK@2px:       {metrics['pck_2px']:.4f}")

        print(f"\n{'─'*60}")
        print(f"  Pose errors (PnP valid: {n_pnp_valid}/{n_total}, "
              f"success rate: {pnp_rate:.1f}%)")
        print(f"{'─'*60}")
        print(f"  x: {metrics['x_error']:.2f} mm  "
              f"y: {metrics['y_error']:.2f} mm  "
              f"z: {metrics['z_error']:.2f} mm")
        print(f"  translation: {metrics['translation_error']:.2f} mm")
        print(f"  alpha: {metrics['alpha_error']:.2f}°  "
              f"beta: {metrics['beta_error']:.2f}°  "
              f"gamma: {metrics['gamma_error']:.2f}°")
        print(f"  rotation: {metrics['rotation_error']:.2f}° (geodesic, symmetry-aware)")
        print(f"  5mm/10deg pass rate: {metrics['pass_5mm_10deg']*100:.1f}%")
        print(f"{'─'*60}")

        # Scale statistics
        scales = np.array([r.get("scale", 1.0) for r in per_instance_results])
        if self.fixed_scale is not None:
            print(f"\n{'─'*60}")
            print(f"  Scale: fixed {self.fixed_scale:.3f}x (all {n_total} images)")
            print(f"{'─'*60}")
        elif self.train_bbox_ratio is not None:
            # test_bbox_ratio = train_bbox_ratio / scale^2
            test_ratios = self.train_bbox_ratio / (scales ** 2)
            n_scaled = int(np.sum(np.abs(scales - 1.0) > 0.02))
            print(f"\n{'─'*60}")
            print(f"  Scale statistics ({n_scaled}/{n_total} images rescaled)")
            print(f"{'─'*60}")
            print(f"  Train obj ratio:  {self.train_bbox_ratio:.4f}")
            print(f"  Test obj ratio:   mean={np.mean(test_ratios):.4f}  "
                  f"min={np.min(test_ratios):.4f}  max={np.max(test_ratios):.4f}")
            print(f"  Applied scale:    mean={np.mean(scales):.3f}  "
                  f"min={np.min(scales):.3f}  max={np.max(scales):.3f}  "
                  f"std={np.std(scales):.3f}")
            print(f"{'─'*60}")

        # Write BOP CSV if output_csv is specified
        output_csv = getattr(self.args, 'output_csv', None)
        if output_csv:
            self._write_bop_csv(per_instance_results, output_csv)
            self._write_detailed_csv(per_instance_results, output_csv)
            self._log_csv_dataframe_to_wandb(output_csv)

        # Save visualizations if requested
        if getattr(self.args, 'save_vis', False) and self._vis_data:
            output_dir = getattr(self.args, 'output_dir', None)
            if output_dir is None and output_csv:
                output_dir = os.path.dirname(output_csv)
            if output_dir:
                sensor = getattr(self.args, 'sensor', None) or "default"
                self._save_visualizations(output_dir, per_instance_results, sensor)

        metrics["per_instance"] = per_instance_results
        return metrics

    def _evaluate_single_instance(self, gt_2d, pred_2d, K, pose, gt_heatmap,
                                  pred_heatmap, pred_contour, infer_time, sample_idx,
                                  scale=1.0):
        """Evaluate a single instance and return detailed metrics."""
        gt_pts = gt_2d.detach().cpu().numpy().reshape(-1, 2)
        pred_pts = pred_2d.detach().cpu().numpy().reshape(-1, 2)
        k = K.detach().cpu().numpy()
        if k.ndim == 1:
            k = k.reshape(3, 3)
        gt_pose = pose.detach().cpu().numpy()

        # Pose solve: use PECP when enabled and contour is available, else plain PnP.
        # PECP includes PnP internally (400 EPnP calls in Phase 1 + progressive in Phase 2).
        # pose_method tracks which solver actually ran:
        #   "pecp"          — PECP ran (sufficient foreground contour)
        #   "pnp_fallback"  — PECP enabled but foreground < 1000px, fell back to plain PnP
        #   "pnp"           — PECP disabled (--no_pecp), plain PnP only
        if self.use_pecp and pred_contour is not None:
            contour_np = pred_contour[0].detach().cpu().numpy()
            # Binarize contour logits: >= 0 → foreground (1), < 0 → background (0)
            contour_bin = contour_np.copy()
            contour_bin[contour_bin >= 0] = 1
            contour_bin[contour_bin < 0] = 0
            contour_bin = contour_bin.astype(np.uint8)
            foreground = np.sum(contour_bin)
            if foreground > 1000:
                t0 = time.time()
                pred_pose = self.PECP(self.keyponits, pred_pts, k, contour_bin, self.list_all)
                t_pose_solve = time.time() - t0
                pose_method = "pecp"
            else:
                t0 = time.time()
                pred_pose = self.pnp(self.keyponits, pred_pts, k)
                t_pose_solve = time.time() - t0
                pose_method = "pnp_fallback"
        else:
            t0 = time.time()
            pred_pose = self.pnp(self.keyponits, pred_pts, k)
            t_pose_solve = time.time() - t0
            pose_method = "pnp"

        # GT pose always uses plain PnP (GT keypoints don't need PECP refinement)
        gt_pose_pnp = self.pnp(self.keyponits, gt_pts, k)

        pnp_failed = pred_pose is None or gt_pose_pnp is None

        if pnp_failed:
            # PnP failed — mark as failed with large but finite sentinel values
            proj_2d_error = float('nan')
            proj_2d_pass = False
            add_value = float('nan')
            add_pass = False
            x_err = y_err = z_err = trans_err = float('nan')
            alpha_err = beta_err = gamma_err = rot_err = float('nan')
            pred_pose = np.zeros((3, 4))  # Placeholder for CSV output
        else:
            # --- Pose metrics ---
            # Projection 2D
            pred_pose_rev = self.pose_reverse(pred_pose, gt_pose_pnp)
            model_2d_pred = project(self.mesh_model["pts"], k, pred_pose_rev)
            model_2d_gt = project(self.mesh_model["pts"], k, gt_pose_pnp)
            proj_2d_error = np.mean(np.linalg.norm(model_2d_pred - model_2d_gt, axis=-1))
            proj_2d_pass = bool(proj_2d_error < self.threshold)

            # ADD / ADD-S
            model_pred = np.dot(self.mesh_model["pts"], pred_pose_rev[:, :3].T) + pred_pose_rev[:, 3]
            model_gt = np.dot(self.mesh_model["pts"], gt_pose_pnp[:, :3].T) + gt_pose_pnp[:, 3]
            if self.obj_key in self.symmetric_objects:
                tree = spatial.cKDTree(model_pred)
                mean_dist, _ = tree.query(model_gt, k=1)
                add_value = np.mean(mean_dist)
            else:
                add_value = np.mean(np.linalg.norm(model_pred - model_gt, axis=-1))
            diameter = self.diameter * 0.1
            add_pass = bool(add_value < diameter)

            # Translation error
            gt_t = gt_pose_pnp[:, 3].flatten()
            pred_t = pred_pose_rev[:, 3].flatten()
            x_err = abs(gt_t[0] - pred_t[0])
            y_err = abs(gt_t[1] - pred_t[1])
            z_err = abs(gt_t[2] - pred_t[2])
            trans_err = np.sqrt(x_err**2 + y_err**2 + z_err**2)

            # Rotation error — symmetry-aware geodesic distance on SO(3)
            # IMPORTANT: Use original pred_pose (not pred_pose_rev) here.
            # pose_reverse already applies symmetry to the prediction for
            # proj_2d/ADD/translation; applying symmetry again to GT would
            # double-transform the residual rotation and corrupt Euler angles.
            rot_err, best_sym_R = min_symmetry_rotation_error(
                pred_pose[:, :3], gt_pose_pnp[:, :3],
                self.all_symmetry_transforms,
                continuous_axes=self.continuous_symmetry_axes)

            # Per-axis Euler angle differences (XYZ intrinsic / "Roll-Pitch-Yaw").
            # Uses the same symmetry-equivalent GT pose that minimized geodesic
            # error, decomposed against the original (un-reversed) prediction.
            R_gt_best = gt_pose_pnp[:, :3] @ best_sym_R
            R_diff = R_gt_best.T @ pred_pose[:, :3]
            diff_euler = ScipyRot.from_matrix(R_diff).as_euler('xyz', degrees=True)
            alpha_err = abs(diff_euler[0])
            beta_err = abs(diff_euler[1])
            gamma_err = abs(diff_euler[2])

        # 5mm/10deg composite metric
        if pnp_failed:
            passes_5mm_10deg = False
        else:
            passes_5mm_10deg = bool((trans_err < 5.0) and (rot_err < 10.0))

        # --- Keypoint metrics ---
        kp_dists = np.linalg.norm(gt_pts - pred_pts, axis=1)
        kp_rmse = np.sqrt(np.mean(kp_dists**2))
        pck_1px = np.mean(kp_dists < 1.0)
        pck_2px = np.mean(kp_dists < 2.0)

        # --- Edge metrics ---
        edge_iou = -1.0
        edge_f1 = -1.0
        if pred_contour is not None:
            contour_np = pred_contour[0].detach().cpu().numpy()
            # Sigmoid to get probabilities, then threshold
            pred_edge = (contour_np > 0).astype(np.uint8)

            # Generate GT edge from heatmap (use gt_heatmap sum as proxy, or compute from Valid3D)
            # For now, use a simple threshold on predicted contour confidence
            gt_edge_from_hm = gt_heatmap.sum(0).detach().cpu().numpy()
            gt_edge_binary = (gt_edge_from_hm > 0.1).astype(np.uint8)

            # Only compute if there are positive pixels in GT
            gt_pos = gt_edge_binary.sum()
            pred_pos = pred_edge.sum()
            if gt_pos > 0 and pred_pos > 0:
                intersection = (gt_edge_binary & pred_edge).sum()
                union = (gt_edge_binary | pred_edge).sum()
                edge_iou = intersection / union if union > 0 else 0.0
                precision = intersection / pred_pos if pred_pos > 0 else 0.0
                recall = intersection / gt_pos if gt_pos > 0 else 0.0
                edge_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Also update legacy metric lists for backward compat
        self.proj_2d.append(proj_2d_pass)
        if proj_2d_pass:
            self.proj_2d_mean.append(proj_2d_error)
        self.add.append(add_pass)
        if add_pass:
            self.x_error_all.append(x_err)
            self.y_error_all.append(y_err)
            self.z_error_all.append(z_err)
            self.alpha_error_all.append(alpha_err)
            self.beta_error_all.append(beta_err)
            self.gama_error_all.append(gamma_err)

        # Metadata
        meta = {}
        if self.samples_metadata and sample_idx < len(self.samples_metadata):
            meta = self.samples_metadata[sample_idx]

        return {
            # BOP CSV fields
            "scene_id": meta.get("scene_id", "0"),
            "im_id": meta.get("frame_id", sample_idx),
            "obj_id": meta.get("obj_id", getattr(self.args, 'obj_id', 1)),
            "instance_idx": meta.get("instance_idx", 0),
            "score": self._compute_detection_score(kp_rmse, edge_iou),
            "pnp_failed": pnp_failed,
            "R": pred_pose[:, :3],
            "t": pred_pose[:, 3],
            "time_network": infer_time,
            "time_pose_solve": t_pose_solve,
            "pose_method": pose_method,
            # Extended metrics
            "visibility": meta.get("visibility", -1.0),
            "proj_2d_error": proj_2d_error,
            "proj_2d_pass": proj_2d_pass,
            "add_value": add_value,
            "add_pass": add_pass,
            "x_error_mm": x_err,
            "y_error_mm": y_err,
            "z_error_mm": z_err,
            "translation_error_mm": trans_err,
            "alpha_error_deg": alpha_err,
            "beta_error_deg": beta_err,
            "gamma_error_deg": gamma_err,
            "rotation_error_deg": rot_err,
            "passes_5mm_10deg": passes_5mm_10deg,
            "keypoint_rmse": kp_rmse,
            "pck_1px": pck_1px,
            "pck_2px": pck_2px,
            "edge_iou": edge_iou,
            "edge_f1": edge_f1,
            # Raw predictions (for further analysis)
            "gt_keypoints_2d": gt_pts,
            "pred_keypoints_2d": pred_pts,
            "scale": float(scale),
        }

    def _aggregate_metrics(self, results):
        """Aggregate per-instance results into summary metrics."""
        proj_2d_mean = np.mean([r["proj_2d_pass"] for r in results])
        add_mean = np.mean([r["add_pass"] for r in results])

        # Count PnP failures
        pnp_valid = [r for r in results if not np.isnan(r["x_error_mm"])]
        num_pnp_failed = len(results) - len(pnp_valid)
        if num_pnp_failed > 0:
            print(f"[Eval] WARNING: PnP failed on {num_pnp_failed}/{len(results)} instances")

        # Average translation/rotation over all PnP-valid instances (exclude NaN)
        if pnp_valid:
            x = np.mean([r["x_error_mm"] for r in pnp_valid])
            y = np.mean([r["y_error_mm"] for r in pnp_valid])
            z = np.mean([r["z_error_mm"] for r in pnp_valid])
            translation_error = np.mean([r["translation_error_mm"] for r in pnp_valid])
            alpha = np.mean([r["alpha_error_deg"] for r in pnp_valid])
            beta = np.mean([r["beta_error_deg"] for r in pnp_valid])
            gamma = np.mean([r["gamma_error_deg"] for r in pnp_valid])
            rotation_error = np.mean([r["rotation_error_deg"] for r in pnp_valid])
        else:
            x = y = z = alpha = beta = gamma = float('nan')
            translation_error = rotation_error = float('nan')

        # 5mm/10deg pass rate over ALL instances (PnP failures count as failures)
        pass_5mm_10deg = np.mean([r["passes_5mm_10deg"] for r in results])

        return {
            "proj_2d": proj_2d_mean,
            "add": add_mean,
            "x_error": x,
            "y_error": y,
            "z_error": z,
            "translation_error": translation_error,
            "alpha_error": alpha,
            "beta_error": beta,
            "gamma_error": gamma,
            "rotation_error": rotation_error,
            "pass_5mm_10deg": pass_5mm_10deg,
            "keypoint_rmse": np.mean([r["keypoint_rmse"] for r in results]),
            "pck_1px": np.mean([r["pck_1px"] for r in results]),
            "pck_2px": np.mean([r["pck_2px"] for r in results]),
            "edge_iou": np.mean([r["edge_iou"] for r in results if r["edge_iou"] >= 0]),
            "edge_f1": np.mean([r["edge_f1"] for r in results if r["edge_f1"] >= 0]),
            "num_instances": len(results),
            "num_pnp_failed": num_pnp_failed,
        }

    def _write_bop_csv(self, results, output_csv):
        """Write BOP-format CSV: scene_id,im_id,obj_id,score,R,t,time"""
        csv_path = output_csv
        with open(csv_path, 'w') as f:
            f.write("scene_id,im_id,obj_id,score,R,t,time\n")
            for r in results:
                R_flat = " ".join(f"{v:.8f}" for v in r["R"].flatten())
                t_flat = " ".join(f"{v:.4f}" for v in r["t"].flatten())
                scene_id = int(r["scene_id"]) if isinstance(r["scene_id"], str) and r["scene_id"].isdigit() else r["scene_id"]
                t_total = r['time_network'] + r['time_pose_solve']
                f.write(f"{scene_id},{r['im_id']},{r['obj_id']},{r['score']:.6f},{R_flat},{t_flat},{t_total:.4f}\n")
        print(f"[Eval] BOP CSV written to: {csv_path}")

    def _write_detailed_csv(self, results, output_csv):
        """Write detailed per-instance CSV with all metrics."""
        # Derive detailed CSV path from BOP CSV path
        # Handles both old format (foo.csv -> foo_detailed.csv) and
        # new format (sensor_bop.csv -> sensor_detailed.csv)
        base = output_csv.rsplit('.', 1)[0] if '.' in output_csv else output_csv
        if base.endswith('_bop'):
            detailed_path = f"{base[:-4]}_detailed.csv"
        else:
            detailed_path = f"{base}_detailed.csv"

        columns = [
            "scene_id", "im_id", "obj_id", "instance_idx", "visibility",
            "proj_2d_error", "proj_2d_pass", "add_value", "add_pass",
            "x_error_mm", "y_error_mm", "z_error_mm", "translation_error_mm",
            "alpha_error_deg", "beta_error_deg", "gamma_error_deg", "rotation_error_deg",
            "passes_5mm_10deg",
            "keypoint_rmse", "pck_1px", "pck_2px",
            "edge_iou", "edge_f1",
            "scale",
            "score", "time_network", "time_pose_solve", "pose_method",
            "pred_keypoints_2d", "gt_keypoints_2d",
        ]

        with open(detailed_path, 'w') as f:
            f.write(",".join(columns) + "\n")
            for r in results:
                vals = []
                for col in columns:
                    v = r.get(col, "")
                    if isinstance(v, np.ndarray):
                        v = " ".join(f"{x:.4f}" for x in v.flatten())
                    elif isinstance(v, (bool, np.bool_)):
                        v = "1" if v else "0"
                    elif isinstance(v, (float, np.floating)):
                        v = f"{v:.6f}"
                    else:
                        v = str(v)
                    vals.append(v)
                f.write(",".join(vals) + "\n")

        print(f"[Eval] Detailed CSV written to: {detailed_path}")

    def _log_csv_dataframe_to_wandb(self, output_csv):
        """Load detailed CSV as DataFrame and log to wandb as a Table."""
        try:
            import pandas as pd
            import wandb as _wandb
            if _wandb.run is None:
                return

            base = output_csv.rsplit('.', 1)[0] if '.' in output_csv else output_csv
            if base.endswith('_bop'):
                detailed_path = f"{base[:-4]}_detailed.csv"
            else:
                detailed_path = f"{base}_detailed.csv"

            if not osp.exists(detailed_path):
                return

            df = pd.read_csv(detailed_path)
            # Drop array columns that don't display well in wandb tables
            drop_cols = [c for c in df.columns if 'keypoints' in c.lower()]
            df_display = df.drop(columns=drop_cols, errors='ignore')

            _wandb.log({"eval/detailed_results": _wandb.Table(dataframe=df_display)})
            print(f"[Eval] Logged DataFrame ({len(df)} rows) to wandb")

            # Print short summary
            print(f"\n[Eval] DataFrame summary ({len(df)} instances):")
            numeric_cols = ['proj_2d_error', 'add_value', 'translation_error_mm',
                            'rotation_error_deg', 'keypoint_rmse']
            summary_cols = [c for c in numeric_cols if c in df.columns]
            if summary_cols:
                print(df[summary_cols].describe().to_string())
        except ImportError:
            print("[Eval] pandas not available, skipping DataFrame logging")
        except Exception as e:
            print(f"[Eval] Failed to log DataFrame to wandb: {e}")

    def _save_visualizations(self, output_dir, per_instance_results, subfolder="default"):
        """
        Save 2-row x 7-column visualization grids for collected samples.

        Row 1 ("Original"): unscaled patch
        Row 2 ("Scaled"):   scaled patch (with GT keypoints transformed to match)

        Columns:
          1. Input (no overlay)
          2. GT Keypoints overlay
          3. Pred Keypoints overlay
          4. GT Edge overlay
          5. Pred Edge overlay
          6. GT Edge + Keypoints combined overlay
          7. Pred Edge + Keypoints combined overlay
        """
        from utils.visualization import denormalize_image

        vis_base = osp.join(str(output_dir), "visualizations")

        COL_LABELS = [
            "Input", "GT Keypoints", "Pred Keypoints",
            "GT Edge", "Pred Edge",
            "GT Edge + KP", "Pred Edge + KP",
        ]
        LABEL_H = 30  # pixels for column label bar
        ROW_LABEL_W = 80  # pixels for row label bar
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.4
        FONT_THICK = 1

        def _to_rgb(tensor):
            """Convert [C,H,W] normalized tensor to [H,W,3] uint8 RGB."""
            img = denormalize_image(tensor).numpy().transpose(1, 2, 0)
            return (img * 255).clip(0, 255).astype(np.uint8)

        def _draw_keypoints(img, kp, color, radius=3):
            out = img.copy()
            if kp is None:
                return out
            for pt in kp.reshape(-1, 2):
                cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))),
                           radius, color, -1, cv2.LINE_AA)
            return out

        def _contour_to_edge_map(contour_logits):
            """Convert network contour logits [1,H,W] tensor to [H,W] float edge map."""
            if contour_logits is None:
                return None
            logits = contour_logits.numpy()[0]  # [H, W]
            return 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> [0,1]

        def _zoom_edge_map(edge_map, scale):
            """Apply center-zoom to an edge map (same transform as _rescale_image_batch).

            scale > 1 means zoom in (object gets bigger).
            Uses grid_sample with inverse mapping: sample from 1/scale region.
            """
            if edge_map is None or abs(scale - 1.0) < 0.02:
                return edge_map
            import torch.nn.functional as F
            t = torch.from_numpy(edge_map).float().unsqueeze(0).unsqueeze(0)
            theta = torch.zeros(1, 2, 3)
            theta[0, 0, 0] = 1.0 / scale
            theta[0, 1, 1] = 1.0 / scale
            grid = F.affine_grid(theta, t.shape, align_corners=False)
            out = F.grid_sample(t, grid, mode='nearest',
                                padding_mode='zeros', align_corners=False)
            return out.squeeze().numpy()

        def _unzoom_edge_map(edge_map, scale):
            """Inverse of _zoom_edge_map: undo center-zoom.

            Maps from scaled space back to original space.
            If zoom used 1/scale, inverse uses scale.
            """
            if edge_map is None or abs(scale - 1.0) < 0.02:
                return edge_map
            import torch.nn.functional as F
            t = torch.from_numpy(edge_map).float().unsqueeze(0).unsqueeze(0)
            theta = torch.zeros(1, 2, 3)
            theta[0, 0, 0] = scale  # inverse: multiply by scale
            theta[0, 1, 1] = scale
            grid = F.affine_grid(theta, t.shape, align_corners=False)
            out = F.grid_sample(t, grid, mode='nearest',
                                padding_mode='zeros', align_corners=False)
            return out.squeeze().numpy()

        def _load_gt_edge(sample_idx, img_h, img_w):
            """
            Load pre-computed GT edge image from disk (patch-level).

            Searches (patch format first, then full-res edges):
              {scene_dir}/patches_{sensor}/edges/{frame}_{inst}.png
              {scene_dir}/edges_{sensor}/{frame}_{inst}.png
              {scene_dir}/edges/{frame}.png

            Returns [H,W] float in [0,1], or None if not found.
            """
            meta = {}
            if self.samples_metadata and sample_idx < len(self.samples_metadata):
                meta = self.samples_metadata[sample_idx]

            bop_root = getattr(self.args, 'bop_root', None)
            sensor = getattr(self.args, 'sensor', None)
            scene_id = meta.get('scene_id', None)
            frame_id = meta.get('frame_id', None)
            inst_idx = meta.get('instance_idx', 0)
            is_patch = meta.get('is_patch', False)

            if bop_root and scene_id is not None and frame_id is not None:
                # Try split dir first, then bop_root directly
                split = getattr(self.args, 'split', 'test')
                split_dir = osp.join(bop_root, split)
                if not osp.isdir(split_dir):
                    split_dir = bop_root

                scene_str = scene_id if isinstance(scene_id, str) else f"{int(scene_id):06d}"
                scene_dir = osp.join(split_dir, scene_str)
                if not osp.isdir(scene_dir):
                    scene_dir = osp.join(bop_root, scene_str)

                candidates = []

                # Try multiple instance index formats (02d and 06d)
                for inst_fmt in [f"{int(inst_idx):02d}", f"{int(inst_idx):06d}"]:
                    patch_key = f"{int(frame_id):06d}_{inst_fmt}"
                    # Patch-level edge (same crop as the input patch)
                    if sensor:
                        candidates.append(osp.join(
                            scene_dir, f"patches_{sensor}", "edges", f"{patch_key}.png"))
                    # Full-res per-instance edge
                    if sensor:
                        candidates.append(osp.join(
                            scene_dir, f"edges_{sensor}", f"{patch_key}.png"))
                # Full-res per-frame edge
                candidates.append(osp.join(
                    scene_dir, "edges", f"{int(frame_id):06d}.png"))

                for path in candidates:
                    if osp.exists(path):
                        edge = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        if edge is not None:
                            if edge.shape[0] != img_h or edge.shape[1] != img_w:
                                edge = cv2.resize(edge, (img_w, img_h),
                                                  interpolation=cv2.INTER_NEAREST)
                            return (edge > 127).astype(np.float32)

            return None

        def _canny_edge(img_rgb, low=50, high=150):
            """Compute Canny edge from RGB [H,W,3] uint8. Returns [H,W] float."""
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low, high)
            return (edges > 0).astype(np.float32)

        def _draw_edge_overlay(img, edge_map, color, alpha=0.6, threshold=0.3):
            """Overlay edge map [H,W] float on image with colored blending."""
            out = img.copy().astype(np.float32)
            if edge_map is None:
                return out.astype(np.uint8)
            color_arr = np.array(color, dtype=np.float32)
            mask = edge_map > threshold
            # Intensity-weighted blending for soft edges
            weight = np.clip(edge_map, 0, 1) * alpha
            for ch in range(3):
                out[:, :, ch] = np.where(
                    mask,
                    weight * color_arr[ch] + (1.0 - weight) * out[:, :, ch],
                    out[:, :, ch],
                )
            return out.clip(0, 255).astype(np.uint8)

        def _scale_keypoints(kp, scale, img_size):
            """Transform keypoints from original space to scaled space."""
            if kp is None:
                return None
            center = img_size / 2.0
            return (kp.reshape(-1, 2) - center) * scale + center

        def _add_col_labels(row_img, labels):
            """Add column labels above a row image."""
            H, W = row_img.shape[:2]
            n_cols = len(labels)
            col_w = W // n_cols
            bar = np.ones((LABEL_H, W, 3), dtype=np.uint8) * 255
            for i, label in enumerate(labels):
                text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
                x = i * col_w + (col_w - text_size[0]) // 2
                y = LABEL_H - (LABEL_H - text_size[1]) // 2
                cv2.putText(bar, label, (x, y), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)
            return np.vstack([bar, row_img])

        def _add_row_label(row_img, label):
            """Add row label to the left of a row image."""
            H, W = row_img.shape[:2]
            bar = np.ones((H, ROW_LABEL_W, 3), dtype=np.uint8) * 255
            text_size = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
            x = (ROW_LABEL_W - text_size[0]) // 2
            y = H // 2 + text_size[1] // 2
            cv2.putText(bar, label, (x, y), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)
            return np.hstack([bar, row_img])

        for idx, vd in enumerate(self._vis_data):
            try:
                img_unscaled_t = vd['img_unscaled']    # [C, H, W] tensor (before rescale)
                img_scaled_t = vd['img_scaled']          # [C, H, W] tensor (after rescale)
                pred_contour = vd['pred_contour']        # [1, H, W] or None
                K = vd['K']                              # [3, 3]
                gt_pose = vd['gt_pose']                  # [3, 4]
                gt_kp = vd['gt_kp']                      # [K, 2] in original space
                pred_kp = vd['pred_kp']                  # [K, 2] in original space (inverse-mapped)
                scale = vd['scale']
                sample_idx = vd['sample_idx']
                scene_id = vd.get('scene_id', '000000')

                img_size = img_unscaled_t.shape[-1]  # assumes square
                img_h, img_w = img_unscaled_t.shape[1], img_unscaled_t.shape[2]

                # Convert to RGB uint8
                img_orig = _to_rgb(img_unscaled_t)
                img_scl = _to_rgb(img_scaled_t)

                # --- Edge maps ---
                # pred_contour is in SCALED space (network input).
                # GT edge is in ORIGINAL patch space.
                # We need both in both spaces.

                # Pred edge: sigmoid of network logits (in scaled space)
                pred_edge_scaled = _contour_to_edge_map(pred_contour)

                # Pred edge in original space: undo the zoom
                pred_edge_orig = _unzoom_edge_map(pred_edge_scaled, scale)

                # GT edge: load from disk or compute Canny on original patch
                gt_edge_orig = _load_gt_edge(sample_idx, img_h, img_w)
                if gt_edge_orig is None:
                    gt_edge_orig = _canny_edge(img_orig)

                # GT edge in scaled space: apply same zoom as image
                gt_edge_scaled = _zoom_edge_map(gt_edge_orig, scale)

                # --- Keypoints ---
                # gt_kp and pred_kp are both in ORIGINAL patch space
                # (pred_kp was inverse-mapped in the eval loop already)
                # For scaled row, transform to scaled space
                gt_kp_scaled = _scale_keypoints(gt_kp, scale, img_size) if gt_kp is not None else None
                pred_kp_scaled = _scale_keypoints(pred_kp, scale, img_size) if pred_kp is not None else None

                GREEN = (0, 255, 0)
                RED = (255, 0, 0)

                # --- Row 1: Original (unscaled) ---
                # All overlays in original patch space
                r1 = []
                r1.append(img_orig.copy())                                             # 1. Input
                r1.append(_draw_keypoints(img_orig, gt_kp, GREEN))                     # 2. GT KP
                r1.append(_draw_keypoints(img_orig, pred_kp, RED))                     # 3. Pred KP
                r1.append(_draw_edge_overlay(img_orig, gt_edge_orig, GREEN))           # 4. GT Edge
                r1.append(_draw_edge_overlay(img_orig, pred_edge_orig, RED))           # 5. Pred Edge
                # 6. GT Edge + KP combined
                combined_gt = _draw_edge_overlay(img_orig, gt_edge_orig, GREEN)
                combined_gt = _draw_keypoints(combined_gt, gt_kp, GREEN)
                r1.append(combined_gt)
                # 7. Pred Edge + KP combined
                combined_pred = _draw_edge_overlay(img_orig, pred_edge_orig, RED)
                combined_pred = _draw_keypoints(combined_pred, pred_kp, RED)
                r1.append(combined_pred)

                row1 = np.concatenate(r1, axis=1)

                # --- Row 2: Scaled ---
                # All overlays in scaled space
                r2 = []
                r2.append(img_scl.copy())                                              # 1. Input
                r2.append(_draw_keypoints(img_scl, gt_kp_scaled, GREEN))               # 2. GT KP
                r2.append(_draw_keypoints(img_scl, pred_kp_scaled, RED))               # 3. Pred KP
                r2.append(_draw_edge_overlay(img_scl, gt_edge_scaled, GREEN))          # 4. GT Edge
                r2.append(_draw_edge_overlay(img_scl, pred_edge_scaled, RED))          # 5. Pred Edge
                # 6. GT Edge + KP combined
                combined_gt_s = _draw_edge_overlay(img_scl, gt_edge_scaled, GREEN)
                combined_gt_s = _draw_keypoints(combined_gt_s, gt_kp_scaled, GREEN)
                r2.append(combined_gt_s)
                # 7. Pred Edge + KP combined
                combined_pred_s = _draw_edge_overlay(img_scl, pred_edge_scaled, RED)
                combined_pred_s = _draw_keypoints(combined_pred_s, pred_kp_scaled, RED)
                r2.append(combined_pred_s)

                row2 = np.concatenate(r2, axis=1)

                # Add column labels (only on top row)
                row1_labeled = _add_col_labels(row1, COL_LABELS)
                # No column labels on row 2 (same columns)
                # Add row labels
                row1_labeled = _add_row_label(row1_labeled, "Original")
                row2_labeled = _add_row_label(row2, f"Scaled ({scale:.2f})")

                # Ensure same width (row1 has label bar height added)
                # Pad row2 to match (col label adds LABEL_H to row1)
                pad = np.ones((LABEL_H, row2_labeled.shape[1], 3), dtype=np.uint8) * 255
                row2_padded = row2_labeled  # no extra top pad needed, row label already matched

                # Stack rows vertically — but row1_labeled is taller by LABEL_H
                # So pad row2_labeled on top to align
                row2_final = np.vstack([
                    np.ones((LABEL_H, row2_labeled.shape[1], 3), dtype=np.uint8) * 255,
                    row2_labeled,
                ])

                # Ensure widths match
                w1 = row1_labeled.shape[1]
                w2 = row2_final.shape[1]
                if w1 != w2:
                    # Pad narrower to match wider
                    max_w = max(w1, w2)
                    if w1 < max_w:
                        row1_labeled = np.hstack([row1_labeled,
                            np.ones((row1_labeled.shape[0], max_w - w1, 3), dtype=np.uint8) * 255])
                    if w2 < max_w:
                        row2_final = np.hstack([row2_final,
                            np.ones((row2_final.shape[0], max_w - w2, 3), dtype=np.uint8) * 255])

                combined = np.vstack([row1_labeled, row2_final])

                # Save to visualizations/{scene_id}/{sensor}/
                vis_dir = osp.join(vis_base, scene_id, subfolder)
                os.makedirs(vis_dir, exist_ok=True)
                save_path = osp.join(vis_dir, f"sample_{sample_idx:04d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            except Exception as e:
                import traceback
                print(f"[Eval] Visualization failed for sample {idx}: {e}")
                traceback.print_exc()

        print(f"[Eval] Saved {len(self._vis_data)} visualizations to: {vis_base}")

    def map_2_points(self, heatmap, pred_heatmap):

        def extract_coords(input_map):
            flat_map = input_map.view(input_map.shape[0], input_map.shape[1], -1)
            max_idx = torch.argmax(flat_map, dim=2)
            width = input_map.shape[3]
            x = (max_idx / width).int().unsqueeze(dim=2)
            y = (max_idx % width).unsqueeze(dim=2)
            return torch.cat((y, x), dim=2)

        gt_points = extract_coords(heatmap)
        pred_points = extract_coords(pred_heatmap)
        return gt_points, pred_points

    def calculate_tra_and_rot(self, pose, pred_pose):
        if self.add[-1] == False:
            return 0
        pred_pose = self.pose_reverse(pred_pose, pose)
        rot = pose[:, :3]
        tra = pose[:, 3:].reshape(1, 3)
        pred_rot = pred_pose[:, :3]
        pred_tra = pred_pose[:, 3:].reshape(1, 3)
        tra_error = tra - pred_tra  # Already in mm

        x_error = math.fabs(tra_error[:, 0])
        y_error = math.fabs(tra_error[:, 1])
        z_error = math.fabs(tra_error[:, 2])

        self.x_error_all.append(x_error)
        self.y_error_all.append(y_error)
        self.z_error_all.append(z_error)

        sy = math.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
        alpha = math.atan2(rot[2, 1], rot[2, 2])
        beta = math.atan2(-rot[2, 0], sy)
        gamma = math.atan2(rot[1, 0], rot[0, 0])

        pred_sy = math.sqrt(pred_rot[2, 1] * pred_rot[2, 1] + pred_rot[2, 2] * pred_rot[2, 2])
        pred_alpha = math.atan2(pred_rot[2, 1], pred_rot[2, 2])
        pred_beta = math.atan2(-pred_rot[2, 0], pred_sy)
        pred_gamma = math.atan2(pred_rot[1, 0], pred_rot[0, 0])

        alpha_error = math.fabs((math.fabs(alpha) - math.fabs(pred_alpha)) * 180 / math.pi)
        beta_error = math.fabs((math.fabs(beta) - math.fabs(pred_beta)) * 180 / math.pi)
        gamma_error = math.fabs((math.fabs(gamma) - math.fabs(pred_gamma)) * 180 / math.pi)

        self.alpha_error_all.append(alpha_error)
        self.beta_error_all.append(beta_error)
        self.gama_error_all.append(gamma_error)

    def calculate_metric(self, keypoints2d, predict2d, K):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            keypoints = keypoints2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            #self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = self.pnp(self.keyponits, keypoints, k)
            pred_pose = self.pnp(self.keyponits, predict, k)
            self.projection_2d(pred_pose, gt_pose, k)
            if self.obj_key in self.symmetric_objects:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)

    def set_error_points(self, predict, img_width=256, img_height=256):
        error_list = np.random.choice(10, self.error_num, replace=False)
        for num in error_list:
            w = np.random.randint(0, img_width)
            h = np.random.randint(0, img_height)
            predict[num] = [w, h]

    def calculate_metric_PECP(self, keypoints2d, predict2d, K, pose, pred_contour):
        """Legacy batch metric calculation using PECP pose refinement.

        Binarizes the predicted contour (logits → {0,1}), then calls PECP
        when sufficient foreground exists (>1000 pixels), otherwise falls
        back to standard RANSAC PnP. Updates legacy metric accumulators
        (proj_2d, add, translation/rotation errors).

        Note: Superseded by the per-instance path in _evaluate_single_instance
        which integrates PECP via the use_pecp flag.
        """
        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            k = K[i].detach().cpu().numpy()
            gt_pose = pose[i].detach().cpu().numpy()

            # Binarize contour logits: >= 0 → foreground (1), < 0 → background (0)
            contour = pred_contour[i][0].detach().cpu().numpy()
            contour[contour >= 0] = 1
            contour[contour < 0] = 0
            contour = np.asarray(contour).astype(np.uint8)

            foreground = np.sum(contour)
            if foreground > 1000:
                # Sufficient contour pixels detected — use PECP refinement
                edge_heatmap = contour
                pred_pose = self.PECP(self.keyponits, predict, k, edge_heatmap, self.list_all)
            else:
                # Too few contour pixels — fall back to standard PnP
                pred_pose = self.pnp(self.keyponits, predict, k)

            self.projection_2d(pred_pose, gt_pose, k)
            if self.obj_key in self.symmetric_objects:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)

    def PECP(self, points_3d, points_2d, K, target_contour, list_all):
        """Pose Evaluation using Contour as a Priori (PECP) — Stage II solver.

        Refines the 6D pose by scoring candidate poses against the predicted
        contour map, selecting the keypoint subset that maximizes contour
        alignment. See paper Section III-B, Fig. 4, Equations 6-12.

        Args:
            points_3d: [K, 3] 3D keypoint coordinates (mm)
            points_2d: [K, 2] predicted 2D keypoint locations (pixels)
            K: [3, 3] camera intrinsic matrix
            target_contour: [H, W] binarized contour map (0/1)
            list_all: all C(K, 4) combinations of keypoint indices

        Returns:
            [3, 4] pose matrix [R|t] with highest contour confidence
        """
        # Build lookup: map each keypoint index to its (2D, 3D) pair.
        # Indices 0-9 → '0'-'9', indices 10+ → 'a','b','c',...
        match_dict = {}
        for i in range(points_3d.shape[0]):
            match_keypoints = np.concatenate((points_2d[i], points_3d[i]), axis=0)
            if i >= 10:
                match_dict[chr(97 + i - 10)] = match_keypoints
            else:
                match_dict[str(i)] = match_keypoints

        # Per-keypoint accumulated confidence score λ (Eq. 11-12)
        list_score = np.zeros(points_2d.shape[0])

        # ── Phase 1: Random 4-point sampling (Eq. 6-11) ──
        # For I iterations, randomly sample 4 keypoints from C(K,4),
        # solve EPnP for a temporary pose T, project Valid3D contour
        # points S onto 2D: P = K[R|t]S (Eq. 6), then score the pose
        # by summing contour values at projected locations (Eq. 10).
        iteration_time = 400  # I = 400 random samples

        for i in range(iteration_time):
            temp_list = choice(list_all)
            keypoints_2d = np.zeros((len(temp_list), 2))
            keypoints_3d = np.zeros((len(temp_list), 3))
            for j in range(len(temp_list)):
                keypoints_2d[j] = match_dict[temp_list[j]][:2]
                keypoints_3d[j] = match_dict[temp_list[j]][2:]

            # Solve EPnP with 4 points → temporary pose T
            _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K,
                                       distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                       flags=cv2.SOLVEPNP_EPNP)
            R, _ = cv2.Rodrigues(R_exp)
            pose = np.concatenate([R, t], axis=-1)

            # Project Valid3D contour points S to image plane (Eq. 6)
            valid_2d = project(self.valid_3d, K, pose).astype(int)

            # Compute contour confidence μ_T = Σ H(u,v) (Eq. 10)
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)

            # Score = μ_T − θ, where θ = m/3 (m = num contour points) (Eq. 11)
            score = sum - 0.33 * valid_2d.shape[0]

            # If score > 0: accumulate to the 4 sampled keypoints' λ (Eq. 12)
            if score > 0:
                for t in temp_list:
                    if t >= 'a':
                        index = ord(t) - 97 + 10
                    else:
                        index = int(t)
                    list_score[index] = list_score[index] + score

        # ── Phase 2: Progressive keypoint selection ──
        # Start with top-4 keypoints by accumulated confidence λ.
        # Iteratively add the next-highest-confidence keypoint, solve PnP
        # with N points, and compute contour confidence Q.
        # If Q > M (best so far): keep the point, update M = Q.
        # If Q ≤ M: discard (set λ = -1), shrink candidate pool.
        max = 0
        total = points_2d.shape[0] + 1
        error_num = 0
        pose_final = None

        for k in range(4, total):
            k = k - error_num
            top_index = self.top_K_idx(list_score, k)
            keypoints_2d = np.zeros((top_index.shape[0], 2))
            keypoints_3d = np.zeros((top_index.shape[0], 3))
            j = 0
            for idx in top_index:
                if idx >= 10:
                    temp_idx = chr(97 + idx - 10)
                else:
                    temp_idx = str(idx)
                keypoints_2d[j] = match_dict[temp_idx][:2]
                keypoints_3d[j] = match_dict[temp_idx][2:]
                j = j + 1

            # Solve PnP: EPnP for exactly 4 points, RANSAC for 5+
            if top_index.shape[0] == 4:
                _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K,
                                           distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                           flags=cv2.SOLVEPNP_EPNP)
                R, _ = cv2.Rodrigues(R_exp)
                pose = np.concatenate([R, t], axis=-1)
            else:
                pose = self.pnp(keypoints_3d, keypoints_2d, K)

            if pose is None:
                # PnP failed for this keypoint subset — discard and continue
                list_score[top_index[-1]] = -1
                total = total - 1
                error_num = error_num + 1
                continue

            # Compute contour confidence Q for this candidate pose
            valid_2d = project(self.valid_3d, K, pose).astype(int)
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)

            if sum >= max:
                # Adding this keypoint improved confidence → keep it
                max = sum
                pose_final = pose
            else:
                # Adding this keypoint decreased confidence → discard it
                list_score[top_index[-1]] = -1
                total = total - 1
                error_num = error_num + 1
                continue

        # ── Final check: compare PECP result against standard RANSAC PnP ──
        # Return whichever pose has higher contour confidence.
        ransac_pose = self.pnp(points_3d, points_2d, K)
        if ransac_pose is not None:
            valid_2d = project(self.valid_3d, K, ransac_pose).astype(int)
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)
            if sum >= max:
                pose_final = ransac_pose

        return pose_final

    def get_confidence(self, target_contour, valid_2d):
        """Compute contour confidence μ_T for a set of projected 2D points (Eq. 10).

        Clamps out-of-bounds projections to (0,0), deduplicates, then sums
        the contour map values H(u,v) at each projected location.

        Args:
            target_contour: [H, W] binarized contour map (0/1)
            valid_2d: [M, 2] integer pixel coordinates (x, y)

        Returns:
            (confidence, valid_2d): total contour sum μ_T and cleaned points
        """
        H, W = target_contour.shape[:2]
        valid_2d[valid_2d[:, 0] >= W] = 0
        valid_2d[valid_2d[:, 0] < 0] = 0
        valid_2d[valid_2d[:, 1] >= H] = 0
        valid_2d[valid_2d[:, 1] < 0] = 0
        valid_2d = np.unique(valid_2d, axis=0)
        sum = np.sum(target_contour[valid_2d[:, 1], valid_2d[:, 0]])
        return sum, valid_2d

    def _compute_detection_score(self, kp_rmse, edge_iou, edge_weight=1.0):
        """
        Compute detection confidence score (higher is better).

        Uses the same formula as cross-validation selection score
        (keypoint_rmse + edge_weight * (1 - edge_iou)) but inverted
        so higher values indicate better detections.

        Maps selection_score through exp(-score) to get a [0, 1] confidence.
        """
        edge_iou_val = max(edge_iou, 0.0)  # Ignore sentinel -1.0
        selection_score = kp_rmse + edge_weight * (1.0 - edge_iou_val)
        return np.exp(-selection_score)

    def pnp(self, points_3d, points_2d, camera_matrix):

        try:
            dist_coeffs = self.dist_coeffs
        except:
            dist_coeffs = np.zeros(shape=[5, 1], dtype="float64")

        assert (
                points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        if points_2d.shape[0] < 5:
            success, R_exp, t = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
        else:
            # Use ITERATIVE for specific objects that benefit from it
            iterative_objects = ["obj2", "obj32"]
            if self.obj_key in iterative_objects:
                success, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                success, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_EPNP)

        if not success:
            return None

        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)

    def projection_2d(self, pose_pred, pose_targets, K):
        # rot pos in z
        pose_pred = self.pose_reverse(pose_pred, pose_targets)

        model_2d_pred = project(self.mesh_model["pts"], K, pose_pred)
        model_2d_targets = project(self.mesh_model["pts"], K, pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
        )

        if proj_mean_diff < self.threshold:
            self.proj_2d_mean.append(proj_mean_diff)
        self.proj_2d.append(proj_mean_diff < self.threshold)

    def pose_reverse(self, pose_pred, pose_targets):
        if self.obj_key in self.symmetric_objects:
            best_pred = pose_pred
            best_dist = np.linalg.norm(pose_targets - pose_pred)
            for S in self.all_symmetry_transforms:
                pose_pred_sym = np.dot(pose_pred, S)
                dist = np.linalg.norm(pose_targets - pose_pred_sym)
                if dist < best_dist:
                    best_dist = dist
                    best_pred = pose_pred_sym
            pose_pred = best_pred
        return pose_pred

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = (
                np.dot(self.mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3]
        )
        model_targets = (
                np.dot(self.mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3]
        )

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add.append(mean_dist < diameter)

    def top_K_idx(self, data, k):
        data = np.array(data)
        idx = data.argsort()[-k:][::-1]
        return idx
