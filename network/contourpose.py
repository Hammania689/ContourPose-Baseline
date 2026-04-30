import gin
import torch
from torch import nn, optim
from .resnet import resnet18
import torch.nn.functional as F
import time
import numpy as np
import wandb
import os
from utils.utils import load_ply, load_camera_intrinsics, get_K_override
from utils.visualization import visualize_batch_with_pose
from dataset.DALIDataset import generate_heatmaps_gpu


@gin.configurable
class ContourPose(torch.nn.Module):
    def __init__(self,
                 fcdim=256, s16dim=256, s8dim=128, s4dim=64, s2dim=64, raw_dim=64,
                 seg_dim=2, feature_dim=64, heatmap_dim=8, edge_dim=1,
                 cat=True, dropout=0.1, sigma=100, init_lr=1e-3, dataloader_size=None, lr_step_size=20, lr_gamma=0.5, weight_decay=0.1, data_root=None, class_type=None,
                 camera_intrinsics_path=None):
        super(ContourPose, self).__init__()

        # self.sigma = sigma

        self.alpha = 0.99
        self.gamma = 2
        # self.cat = cat  # Ture
        # self.dropout = dropout  # 0.1
        # self.seg_dim = seg_dim  # 2
        # self.feature_dim = feature_dim  # 64
        # self.heatmap_dim = heatmap_dim
        # self.edge_dim = edge_dim
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.global_step = 0

        # Cross-validation best model tracking
        # Named 'best_pose_error' for checkpoint compatibility, but actually stores
        # selection_score = keypoint_rmse + (1 - edge_iou), lower is better
        self.best_pose_error = 1e9  # Initialize to very high value

        # Save all __init__ parameters as instance attributes (excluding self to avoid circular reference)
        local_vars = locals()
        local_vars.pop('self', None)
        self.__dict__.update(local_vars)

        # Load camera intrinsics from file if provided (overrides batch K in validation/visualization)
        self.K_override = None
        if camera_intrinsics_path is not None:
            self._load_camera_intrinsics(camera_intrinsics_path)



        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=16,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # The second encoder
        # x16s -> 256
        self.conv16s = nn.Sequential(
            nn.Conv2d(256 + fcdim, s16dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s16dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up16sto8s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + s16dim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_raw = nn.Sequential(
            # input channel
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
        )
        self.conv_heatmap = nn.Sequential(
            nn.Conv2d(raw_dim, heatmap_dim, 1, 1)

        )
        self.conv_edge = nn.Sequential(
            nn.Conv2d(raw_dim, edge_dim, 1, 1)
        )

        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        # Setup Optimizer and LR scheduler
        self.opt = optim.AdamW(self.parameters(), weight_decay=self.weight_decay, lr=init_lr)
        self.lr_sched = optim.lr_scheduler.StepLR(self.opt, step_size=lr_step_size, gamma=lr_gamma)


        # Initialize keypoints, contours, and object diameter
        self._init_geo_info()


    def _init_geo_info(self):
        # Load 3D keypoints (cached)
        if not hasattr(self, 'keypoints_3d'):
            self.keypoints_3d = np.loadtxt(os.path.join(self.data_root, f"keypoints/{self.class_type}.txt")) * 1000  # m → mm (consistent with pts_3d and BOP cam_t_m2c)

        # Load 3D contour points (cached in memory after first call)
        if not hasattr(self, 'pts_3d'):
            self.pts_3d = np.loadtxt(os.path.join(self.data_root, f"Valid3D/{self.class_type}.txt"))  # Already in mm

        self.object_diameter = np.linalg.norm(self.keypoints_3d.max(axis=0) - self.keypoints_3d.min(axis=0))

    def _load_camera_intrinsics(self, intrinsics_path):
        """Load camera intrinsics from file using shared utility."""
        print(f"[ContourPose] Loading camera intrinsics from: {intrinsics_path}")
        self.K_override = load_camera_intrinsics(intrinsics_path)
        print(f"[ContourPose] Loaded K_override: fx={self.K_override[0,0]:.1f}, fy={self.K_override[1,1]:.1f}, "
              f"cx={self.K_override[0,2]:.1f}, cy={self.K_override[1,2]:.1f}")

    def get_K(self, batch_K=None, device=None):
        """Get camera intrinsics matrix. Uses K_override if set, otherwise returns batch_K."""
        return get_K_override(self.K_override, batch_K, device)


    def heatmap_loss(self, pred_heatmap, gt_heatmap):
        """
        NOTE: The scale used below is a loss scale NOT a unit scale for the 3D model.
        Heatmaps are 2D Gaussian blobs with values 0-1 — their magnitude doesn't change based on whether your 3D model is in meters or millimeters.
        The MSE on these small values produces tiny gradients compared to BCEWithLogitsLoss on the contour branch.
        """
        loss_scale = 1000
        return F.mse_loss(pred_heatmap, gt_heatmap) * loss_scale


    def weighted_cross_entropy_loss(self, pred_contour, target):
        """ Calculate sum of weighted cross entropy loss. """
        mask = (target > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight.masked_scatter_(target > 0.5,
                               torch.ones_like(target) * num_neg / (num_pos + num_neg))
        weight.masked_scatter_(target <= 0.5,
                               torch.ones_like(target) * num_pos / (num_pos + num_neg))
        losses = F.binary_cross_entropy_with_logits(
            pred_contour.float(), target.float(), weight=weight, reduction='none')
        loss = torch.sum(losses) / b
        return loss

    def forward(self, x, heatmap = None, target_contour = None):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = self.up16sto8s(fm1)
        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = self.up8sto4s(fm1)
        if fm1.shape[2] == 136:
            fm1 = nn.functional.interpolate(fm1, (135, 180), mode='bilinear', align_corners=False)

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = self.up4sto2s(fm1)

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = self.up2storaw(fm1)
        fm1 = self.conv_raw(torch.cat([fm1, x], 1))
        fm1 = self.conv_heatmap(fm1)

        pred_heatmap = fm1
        fm2 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm2 = self.up16sto8s(fm2)
        fm2 = self.conv8s(torch.cat([fm2, x8s], 1))
        fm2 = self.up8sto4s(fm2)
        if fm2.shape[2] == 136:
            fm2 = nn.functional.interpolate(fm2, (135, 180), mode='bilinear', align_corners=False)

        fm2 = self.conv4s(torch.cat([fm2, x4s], 1))
        fm2 = self.up4sto2s(fm2)

        fm2 = self.conv2s(torch.cat([fm2, x2s], 1))
        fm2 = self.up2storaw(fm2)
        fm2 = self.conv_raw(torch.cat([fm2, x], 1))
        fm2 = self.conv_edge(fm2)
        pred_contour = fm2

        return pred_heatmap, pred_contour
    
    def reset_logger(self, epoch, data_size):
        """Reset logging metrics at the start of each epoch"""
        self.total_loss = []
        self.total_heatmap_loss = []
        self.total_contour_loss = []
        self.start_time = time.time()
        self.epoch = epoch
        self.data_size = data_size

    def log_epoch(self):
        """Log epoch-level metrics to wandb"""

        # End of epoch metrics
        avg_loss         = np.mean(self.total_loss)
        avg_heatmap_loss = np.mean(self.total_heatmap_loss)
        avg_contour_loss = np.mean(self.total_contour_loss)

        epoch_duration   = time.time() - self.start_time
        minutes, seconds = divmod(epoch_duration, 60)
        # Use f-string for formatting with leading zeros for seconds
        formatted_time = f"{int(minutes):02d}:{int(seconds):02d}"

        # Log epoch metrics
        if wandb.run is not None:
            wandb.log({
                "train/loss_epoch": avg_loss,
                "train/heatmap_loss_epoch": avg_heatmap_loss,
                "train/contour_loss_epoch": avg_contour_loss,
                "train/epoch_time_seconds": epoch_duration,
                "train/epoch_time_formatted": formatted_time,
                "epoch": self.epoch,
                })



    def optimize_params(self, data, batch_idx, model_wrapper=None):
        """Perform one optimization step with the given batch

        Args:
            data: Batch data from dataloader
            batch_idx: Current batch index
            model_wrapper: DataParallel wrapper (if using multi-GPU), None for single GPU
        """
        self.opt.zero_grad()

        # Extract data from DALI loader
        img = data[0]["images"]
        gt_heatmap = data[0]["heatmaps"]
        gt_contour = data[0]["edges"]

        # Forward pass - use wrapper for DataParallel, otherwise use self
        # This ensures DataParallel properly distributes computation across GPUs
        model_forward = model_wrapper if model_wrapper is not None else self
        pred_heatmap, pred_contour = model_forward(img)

        # Compute losses (loss functions already return scalars with reduction='mean')
        # DataParallel automatically averages across GPUs
        heatmap_loss = self.heatmap_loss(pred_heatmap, gt_heatmap)
        contour_loss = self.seg_loss(pred_contour.float(), gt_contour.float())

        loss = heatmap_loss + contour_loss

        # Track losses for epoch-level logging
        self.total_loss.append(loss.detach().cpu().item())
        self.total_heatmap_loss.append(heatmap_loss.detach().cpu().item())
        self.total_contour_loss.append(contour_loss.detach().cpu().item())

        # Backward pass
        loss.backward()

        # Optimizer step
        self.opt.step()

        self.global_step += 1

    def visualize_batches(self, fixed_batch, random_batch, class_type, epoch, device):
        """
        Visualize fixed and random validation batches with pose estimation.

        Args:
            fixed_batch: Dict with "images", "keypoints_2d", "edges", "K", "pose"
            random_batch: Dict with "images", "keypoints_2d", "edges", "K", "pose"
            class_type: Object class (e.g., "obj1")
            epoch: Current epoch number
            device: Device (cuda/cpu)
        """


        self.eval()

        with torch.no_grad():
            # Fixed batch visualization
            images_fixed = fixed_batch["images"].to(device)
            heatmaps_fixed = fixed_batch["heatmaps"].to(device)
            edges_fixed = fixed_batch["edges"].to(device)
            K_fixed_batch = fixed_batch["K"].to(device)
            pose_fixed = fixed_batch["pose"].to(device)

            # Use K_override if set, otherwise use batch K
            K_fixed = self.get_K(K_fixed_batch, device)

            # Build batch dict with potentially overridden K
            batch_fixed = {
                "images": images_fixed,
                "heatmaps": heatmaps_fixed,
                "edges": edges_fixed,
                "K": K_fixed,
                "pose": pose_fixed
            }

            pred_heatmaps_fixed, pred_edges_fixed = self.forward(images_fixed)
            grid_fixed, metrics_fixed = visualize_batch_with_pose(
                batch_fixed, pred_heatmaps_fixed, pred_edges_fixed,
                self.keypoints_3d, self.pts_3d, num_samples=4, debug=False
            )

            # Random batch visualization
            images_random = random_batch["images"].to(device)
            heatmaps_random = random_batch["heatmaps"].to(device)
            edges_random = random_batch["edges"].to(device)
            K_random_batch = random_batch["K"].to(device)
            pose_random = random_batch["pose"].to(device)

            # Use K_override if set, otherwise use batch K
            K_random = self.get_K(K_random_batch, device)

            batch_random = {
                "images": images_random,
                "heatmaps": heatmaps_random,
                "edges": edges_random,
                "K": K_random,
                "pose": pose_random
            }

            pred_heatmaps_random, pred_edges_random = self.forward(images_random)
            grid_random, metrics_random = visualize_batch_with_pose(
                batch_random, pred_heatmaps_random, pred_edges_random,
                self.keypoints_3d, self.pts_3d, num_samples=4
            )

            # Log to wandb
            wandb.log({
                "train/fixed_batch": wandb.Image(grid_fixed.cpu().numpy().transpose(1, 2, 0)),
                "train/random_batch": wandb.Image(grid_random.cpu().numpy().transpose(1, 2, 0)),
                "train/fixed_trans_error_mm": metrics_fixed["avg_translation_error_mm"],
                "train/fixed_rot_error_deg": metrics_fixed["avg_rotation_error_deg"],
                "train/random_trans_error_mm": metrics_random["avg_translation_error_mm"],
                "train/random_rot_error_deg": metrics_random["avg_rotation_error_deg"],
            "epoch": self.epoch,
            })

        self.train()

    def validate_with_pose_metrics(self, val_loader, class_type, epoch, device, val_name="val"):
        """
        Run validation with detection and pose metrics.

        Model selection is based on detection metrics (keypoint RMSE + edge IoU)
        which directly measure network prediction quality, rather than derived
        pose metrics which are sensitive to PnP solver noise.

        Computes:
        - Detection metrics (keypoint RMSE, PCK@1px, PCK@2px, edge IoU/F1) - for model selection
        - Standard losses (heatmap, contour, total) - for monitoring
        - Pose estimation metrics (translation, rotation error) - for monitoring only

        Args:
            val_loader: Validation data loader
            class_type: Object class (e.g., "obj1")
            epoch: Current epoch number
            device: Device (cuda/cpu)
            val_name: Name prefix for wandb logging (e.g., "val_fixed", "val_random")

        Returns:
            Dictionary of validation metrics
        """
        from utils.visualization import (
            extract_keypoints_from_heatmap,
            pnp_pose_estimation,
            compute_pose_metrics,
            compute_detection_metrics,
            compute_selection_score,
        )
        from dataset.DALIDataset import generate_heatmaps_gpu

        self.eval()

        # Accumulators for losses
        val_losses = []
        val_heatmap_losses = []
        val_contour_losses = []

        # Accumulators for detection metrics (aggregated per batch)
        all_keypoint_rmse = []
        all_pck_1px = []
        all_pck_2px = []
        all_edge_iou = []
        all_edge_f1 = []

        # Accumulators for pose metrics (monitoring only)
        val_trans_errors = []
        val_rot_errors = []
        pnp_failures = 0
        total_samples = 0

        with torch.no_grad():
            for data in val_loader:
                # Extract data
                img = data[0]["images"].to(device)
                gt_contour = data[0]["edges"].to(device)
                K_batch = data[0]["K"].to(device)
                gt_pose = data[0]["pose"].to(device)

                batch_size = img.shape[0]
                total_samples += batch_size

                # Get GT keypoints - either directly or by extracting from heatmaps
                if "keypoints_2d" in data[0]:
                    # DALI loader provides keypoints directly
                    gt_keypoints_2d = data[0]["keypoints_2d"].to(device)  # [B, N, 2]
                    img_h, img_w = img.shape[2], img.shape[3]
                    gt_heatmap = generate_heatmaps_gpu(gt_keypoints_2d, height=img_h, width=img_w)
                else:
                    # BOP loader provides heatmaps - extract keypoints from them
                    gt_heatmap = data[0]["heatmaps"].to(device)
                    gt_keypoints_2d = extract_keypoints_from_heatmap(gt_heatmap)

                # Use K_override if set, otherwise use batch K
                K = self.get_K(K_batch, device)

                # Forward pass
                pred_heatmap, pred_contour = self.forward(img)

                # === Compute losses (for monitoring) ===
                heatmap_loss = self.heatmap_loss(pred_heatmap, gt_heatmap)
                contour_loss = self.seg_loss(pred_contour.float(), gt_contour.float())
                loss = heatmap_loss + contour_loss

                val_losses.append(loss.cpu().item())
                val_heatmap_losses.append(heatmap_loss.cpu().item())
                val_contour_losses.append(contour_loss.cpu().item())

                # === Compute detection metrics (for model selection) ===
                detection_metrics = compute_detection_metrics(
                    pred_heatmap, pred_contour, gt_keypoints_2d, gt_contour,
                    pck_thresholds=[1.0, 2.0]
                )

                all_keypoint_rmse.append(detection_metrics["keypoint_rmse"])
                all_pck_1px.append(detection_metrics["pck@1.0px"])
                all_pck_2px.append(detection_metrics["pck@2.0px"])
                all_edge_iou.append(detection_metrics["edge_iou"])
                all_edge_f1.append(detection_metrics["edge_f1"])

                # === Compute pose metrics (for monitoring only) ===
                pred_keypoints_2d = extract_keypoints_from_heatmap(pred_heatmap)

                for i in range(batch_size):
                    try:
                        K_np = K[i].cpu().numpy()
                        gt_pose_np = gt_pose[i].cpu().numpy()
                        pred_kpts_2d = pred_keypoints_2d[i].cpu().numpy()

                        # Estimate pose via PnP
                        pred_pose_np = pnp_pose_estimation(self.keypoints_3d, pred_kpts_2d, K_np)

                        # Compute pose errors
                        pose_metrics = compute_pose_metrics(pred_pose_np, gt_pose_np)
                        val_trans_errors.append(pose_metrics["translation_error_mm"])
                        val_rot_errors.append(pose_metrics["rotation_error_deg"])

                    except Exception as e:
                        pnp_failures += 1
                        # Penalty: worst-case errors
                        val_trans_errors.append(self.object_diameter)
                        val_rot_errors.append(180)

        # === Aggregate metrics ===
        avg_keypoint_rmse = np.mean(all_keypoint_rmse)
        avg_pck_1px = np.mean(all_pck_1px)
        avg_pck_2px = np.mean(all_pck_2px)
        avg_edge_iou = np.mean(all_edge_iou)
        avg_edge_f1 = np.mean(all_edge_f1)

        avg_trans_error = np.mean(val_trans_errors)
        avg_rot_error = np.mean(val_rot_errors)
        pnp_success_rate = 1 - (pnp_failures / total_samples) if total_samples > 0 else 0

        # Compute selection score (lower is better)
        selection_metrics = {
            "keypoint_rmse": avg_keypoint_rmse,
            "edge_iou": avg_edge_iou,
        }
        selection_score = compute_selection_score(selection_metrics, edge_weight=1.0)

        # Build metrics dictionary
        metrics = {
            # Detection metrics (for model selection)
            f"{val_name}/keypoint_rmse": avg_keypoint_rmse,
            f"{val_name}/pck@1px": avg_pck_1px,
            f"{val_name}/pck@2px": avg_pck_2px,
            f"{val_name}/edge_iou": avg_edge_iou,
            f"{val_name}/edge_f1": avg_edge_f1,
            f"{val_name}/selection_score": selection_score,
            # Loss metrics (for monitoring)
            f"{val_name}/loss": np.mean(val_losses),
            f"{val_name}/heatmap_loss": np.mean(val_heatmap_losses),
            f"{val_name}/contour_loss": np.mean(val_contour_losses),
            # Pose metrics (for monitoring only)
            f"{val_name}/trans_error_mm": avg_trans_error,
            f"{val_name}/rot_error_deg": avg_rot_error,
            f"{val_name}/pnp_success_rate": pnp_success_rate,
            "epoch": self.epoch
        }

        # Log to wandb
        wandb.log(metrics)

        print(f"  {val_name} - RMSE: {avg_keypoint_rmse:.2f}px, PCK@1px: {avg_pck_1px:.1f}%, "
              f"PCK@2px: {avg_pck_2px:.1f}%, IoU: {avg_edge_iou:.3f}")
        print(f"  {val_name} - Trans: {avg_trans_error:.2f}mm, Rot: {avg_rot_error:.2f}° "
              f"(selection_score: {selection_score:.4f})")

        # === Check and Save Best Model ===
        # Use detection-based selection score instead of pose error
        run_name = wandb.run.name if wandb.run is not None else None

        if selection_score < self.best_pose_error:  # Note: best_pose_error now stores selection_score
            self.best_pose_error = selection_score

            self.update_best_checkpoint(
                pose_metrics=metrics,
                cur_pose_error=selection_score,  # Actually selection_score now
                epoch=epoch,
                run_name=run_name
            )

        self.train()
        return metrics

    def save_checkpoint(self, epoch,  model_wrapper=None, save_best=False, run_name=None):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            model_wrapper: DataParallel wrapper (if using multi-GPU), otherwise uses self
            save_best: If True, saves as 'best_model.pkl', otherwise 'model.pkl'
            run_name: WandB run name for organizing checkpoints (optional)

        Returns:
            Path to saved checkpoint
        """
        from pathlib import Path

        # Determine which model to save (unwrap DataParallel if needed)
        model_to_save = model_wrapper if model_wrapper is not None else self

        # Build checkpoint directory
        if run_name:
            ckpt_dir = Path.cwd() / 'model' / self.class_type / run_name
        else:
            ckpt_dir = Path.cwd() / 'model' / self.class_type

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build state dict
        state = {
            'net': model_to_save.state_dict(),
            'optimizer': self.opt.state_dict(),
            'lr_scheduler': self.lr_sched.state_dict(),
            'epoch': epoch,
            'best_pose_error': self.best_pose_error,
            'global_step': self.global_step,
        }

        # Choose filename
        if save_best:
            ckpt_path = ckpt_dir / "best_model.pkl"
            print(f"  [Checkpoint] Saving BEST model (selection_score={self.best_pose_error:.4f}) to {ckpt_path}")
        else:
            ckpt_path = ckpt_dir / "model.pkl"

        # Save checkpoint
        torch.save(state, str(ckpt_path))

        # Log to wandb if available
        if wandb.run is not None:
            wandb.save(str(ckpt_path))

        return ckpt_path

    def load_checkpoint(self, checkpoint_path=None, run_name=None, load_best=False, strict=True,
                        estimated_steps=None):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Direct path to checkpoint file (overrides run_name/load_best)
            run_name: WandB run name subdirectory (used if checkpoint_path is None)
            load_best: If True, loads 'best_model.pkl', else 'model.pkl'
            strict: Whether to strictly enforce state_dict keys match
            estimated_steps: Manually specify global_step for legacy checkpoints without LR scheduler state.
                             If provided, fast-forwards LR scheduler to this step.
                             Formula: estimated_steps = epoch * batches_per_epoch

        Returns:
            Dictionary with checkpoint info: {'epoch': int, 'best_pose_error': float, 'global_step': int}
            Returns None if checkpoint not found
        """
        from pathlib import Path

        # Determine checkpoint path
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
        else:
            # Build path from class_type and run_name
            if run_name:
                ckpt_dir = Path.cwd() / 'model' / self.class_type / run_name
            else:
                ckpt_dir = Path.cwd() / 'model' / self.class_type

            filename = "best_model.pkl" if load_best else "model.pkl"
            ckpt_path = ckpt_dir / filename

        # Check if file exists
        if not ckpt_path.exists():
            print(f"  [Checkpoint] Not found: {ckpt_path}")
            return None

        print(f"  [Checkpoint] Loading: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(str(ckpt_path), map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Load model state dict
        is_old_format = False
        try:
            state_dict = checkpoint['net']
        except KeyError:
            # Handle old checkpoint format (just state dict, no wrapper)
            state_dict = checkpoint
            is_old_format = True

        # Strip "module." prefix from DataParallel-saved checkpoints
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}

        self.load_state_dict(state_dict, strict=strict)

        if is_old_format:
            return {'epoch': 0, 'best_pose_error': float('inf')}

        # Load optimizer state dict
        if 'optimizer' in checkpoint:
            try:
                self.opt.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"  [Checkpoint] Warning: Could not load optimizer state: {e}")

        # Load LR scheduler state dict
        if 'lr_scheduler' in checkpoint:
            try:
                self.lr_sched.load_state_dict(checkpoint['lr_scheduler'])
                print(f"  [Checkpoint] Restored LR scheduler (LR={self.lr_sched.get_last_lr()[0]:.6f})")
            except Exception as e:
                print(f"  [Checkpoint] Warning: Could not load LR scheduler state: {e}")
        else:
            # Legacy checkpoint - need to fast-forward LR scheduler
            epoch = checkpoint.get('epoch', 0)

            # Determine steps to fast-forward
            steps_to_forward = None
            if estimated_steps is not None:
                # User provided explicit step count
                steps_to_forward = estimated_steps
                print(f"  [Checkpoint] No LR scheduler state - using provided estimated_steps={estimated_steps}")
            elif epoch > 0:
                # StepLR steps once per epoch
                steps_to_forward = epoch
                print(f"  [Checkpoint] No LR scheduler state - fast-forwarding {steps_to_forward} epochs")

            if steps_to_forward and steps_to_forward > 0:
                print(f"  [Checkpoint] Fast-forwarding LR scheduler {steps_to_forward} epochs...")
                for _ in range(steps_to_forward):
                    self.lr_sched.step()
                print(f"  [Checkpoint] LR scheduler fast-forwarded (LR={self.lr_sched.get_last_lr()[0]:.6f})")
            else:
                print(f"  [Checkpoint] Warning: No LR scheduler state in checkpoint (using fresh scheduler)")
                print(f"  [Checkpoint] Hint: Use estimated_steps parameter or pass dataloader_size to ContourPose()")

        # Restore global_step (for LR scheduler stepping)
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        elif estimated_steps is not None:
            self.global_step = estimated_steps
            print(f"  [Checkpoint] Set global_step={self.global_step} from estimated_steps")
        elif 'epoch' in checkpoint and hasattr(self, 'dataloader_size') and self.dataloader_size:
            # Estimate global_step from epoch
            self.global_step = checkpoint['epoch'] * self.dataloader_size
            print(f"  [Checkpoint] Estimated global_step={self.global_step} from epoch")
        else:
            self.global_step = 0

        # Restore best_pose_error (handle multiple legacy field names)
        if 'best_pose_error' in checkpoint:
            self.best_pose_error = checkpoint['best_pose_error']
        elif 'best_val_loss' in checkpoint:
            # Handle legacy checkpoint format
            self.best_pose_error = checkpoint['best_val_loss']
            print(f"  [Checkpoint] Migrated 'best_val_loss' -> 'best_pose_error'")
        elif 'cur_pose_error' in checkpoint:
            # Legacy field - reset to high value since metrics may not be comparable
            self.best_pose_error = 1e9
            print(f"  [Checkpoint] Found legacy 'cur_pose_error' ({checkpoint['cur_pose_error']:.4f}) - reset to {self.best_pose_error}")

        # Get epoch
        epoch = checkpoint.get('epoch', 0)

        print(f"  [Checkpoint] Loaded epoch {epoch}, best_pose_error={self.best_pose_error:.4f}, global_step={self.global_step}")

        return {
            'epoch': epoch,
            'best_pose_error': self.best_pose_error,
            'global_step': self.global_step
        }

    def update_best_checkpoint(self, pose_metrics, cur_pose_error, epoch, run_name=None):
        """
        Update and save best checkpoint when validation selection score improves.

        The selection score is based on detection metrics (keypoint RMSE + edge IoU)
        which directly measure network prediction quality.

        Args:
            pose_metrics: dictionary of all validation metrics
            cur_pose_error: current selection score (RMSE + (1-IoU)), lower is better
            epoch: Current epoch number
            run_name: WandB run name for organizing checkpoints
        """
        print(f"  [Best Model] New best selection score: {cur_pose_error:.4f} "
              f"(RMSE: {pose_metrics.get('val_fixed/keypoint_rmse', 0):.2f}px, "
              f"IoU: {pose_metrics.get('val_fixed/edge_iou', 0):.3f})")

        # Log to wandb - use summary for best model stats (persists and updates)
        if wandb.run is not None:
            wandb.log({"best_model/pose_error": cur_pose_error, "epoch": epoch})

            # Update summary with best model metrics (cleaner than Table)
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_pose_error"] = cur_pose_error
            for key, value in pose_metrics.items():
                # Clean up key names (remove val_fixed/ prefix if present)
                clean_key = key.replace("val_fixed/", "best_")
                wandb.run.summary[clean_key] = value

        # Save best checkpoint
        self.save_checkpoint(
            epoch=epoch,
            model_wrapper=self,
            save_best=True,
            run_name=run_name
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test — run with: python -m network.contourpose
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ContourPose — Standalone Architecture Test")
    print("=" * 70)

    # ── Test configuration ──
    B = 8
    H, W = 256, 256
    HEATMAP_DIM = 8
    EDGE_DIM = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Create dummy input ──
    rgb = torch.randn(B, 3, H, W, device=device)
    print(f"\nInput shape:")
    print(f"  RGB: {rgb.shape}")

    # ── Build model (skip geo_info / file loading) ──
    print(f"\n{'─' * 70}")
    print("Test 1: Model construction & forward pass")
    print(f"{'─' * 70}")

    original_init_geo = ContourPose._init_geo_info
    ContourPose._init_geo_info = lambda self: None

    model = ContourPose(
        heatmap_dim=HEATMAP_DIM,
        edge_dim=EDGE_DIM,
        data_root=".",
        class_type="test_obj",
    ).to(device)

    ContourPose._init_geo_info = original_init_geo

    # ── Parameter count ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Break down by component
    encoder_params = sum(p.numel() for p in model.resnet18_8s.parameters())
    decoder_params = (
        sum(p.numel() for p in model.conv16s.parameters())
        + sum(p.numel() for p in model.conv8s.parameters())
        + sum(p.numel() for p in model.conv4s.parameters())
        + sum(p.numel() for p in model.conv2s.parameters())
        + sum(p.numel() for p in model.conv_raw.parameters())
    )
    heatmap_head_params = sum(p.numel() for p in model.conv_heatmap.parameters())
    edge_head_params = sum(p.numel() for p in model.conv_edge.parameters())

    print(f"\n  Parameter breakdown:")
    print(f"    ResNet-18 encoder:    {encoder_params:>12,}")
    print(f"    Shared decoder:       {decoder_params:>12,}")
    print(f"    Heatmap head (1x1):   {heatmap_head_params:>12,}")
    print(f"    Edge head (1x1):      {edge_head_params:>12,}")
    print(f"    {'─' * 45}")
    print(f"    Total:                {total_params:>12,}")
    print(f"    Trainable:            {trainable_params:>12,}")

    # ── Forward pass ──
    pred_heatmap, pred_contour = model(rgb)

    print(f"\n  Forward pass outputs:")
    print(f"    Heatmap: {pred_heatmap.shape}  (expected: ({B}, {HEATMAP_DIM}, {H}, {W}))")
    print(f"    Contour: {pred_contour.shape}  (expected: ({B}, {EDGE_DIM}, {H}, {W}))")

    assert pred_heatmap.shape == (B, HEATMAP_DIM, H, W), \
        f"Heatmap shape mismatch: got {pred_heatmap.shape}"
    assert pred_contour.shape == (B, EDGE_DIM, H, W), \
        f"Contour shape mismatch: got {pred_contour.shape}"
    print(f"  ✓ Both output shapes correct")

    # ── Gradient flow test ──
    print(f"\n{'─' * 70}")
    print("Test 2: Gradient flow")
    print(f"{'─' * 70}")

    loss = pred_heatmap.sum() + pred_contour.sum()
    loss.backward()

    components = {
        "ResNet conv1":     model.resnet18_8s.conv1.weight,
        "conv16s":          list(model.conv16s.parameters())[0],
        "conv8s":           list(model.conv8s.parameters())[0],
        "conv4s":           list(model.conv4s.parameters())[0],
        "conv2s":           list(model.conv2s.parameters())[0],
        "conv_raw":         list(model.conv_raw.parameters())[0],
        "conv_heatmap":     list(model.conv_heatmap.parameters())[0],
        "conv_edge":        list(model.conv_edge.parameters())[0],
    }

    all_grads_ok = True
    for name, param in components.items():
        has_grad = param.grad is not None and param.grad.abs().sum() > 0
        status = "✓" if has_grad else "✗"
        print(f"    {status} {name}: grad={'yes' if has_grad else 'NONE'}")
        all_grads_ok = all_grads_ok and has_grad

    # ── Shared vs separate decoder comparison ──
    print(f"\n{'─' * 70}")
    print("Test 3: Shared decoder vs paper's two-decoder comparison")
    print(f"{'─' * 70}")
    print("  Current impl: SHARED decoder — both branches reuse the same")
    print("  conv16s/conv8s/conv4s/conv2s/conv_raw in two forward passes,")
    print("  diverging only at the final 1x1 output convolutions.")
    print()
    print("  Paper describes: TWO SEPARATE decoders — each branch has its own")
    print("  independent decoder weights (no weight sharing).")

    # Per-layer decoder breakdown
    decoder_layers = {
        "conv16s": model.conv16s,
        "conv8s":  model.conv8s,
        "conv4s":  model.conv4s,
        "conv2s":  model.conv2s,
        "conv_raw": model.conv_raw,
    }
    print(f"\n  Shared decoder layer breakdown:")
    layer_params = {}
    for name, layer in decoder_layers.items():
        p = sum(p.numel() for p in layer.parameters())
        layer_params[name] = p
        print(f"    {name:<12s}  {p:>10,}")
    print(f"    {'─' * 30}")
    print(f"    {'Decoder total':<12s}  {decoder_params:>10,}")

    # Compute hypothetical two-decoder totals
    two_decoder_params = decoder_params * 2
    paper_total = encoder_params + two_decoder_params + heatmap_head_params + edge_head_params
    param_diff = paper_total - total_params
    pct_increase = (param_diff / total_params) * 100

    print(f"\n  {'':>30s} {'Shared (ours)':>14s}  {'Separate (paper)':>16s}")
    print(f"  {'─' * 64}")
    print(f"  {'Encoder (ResNet-18)':<30s} {encoder_params:>14,}  {encoder_params:>16,}")
    print(f"  {'Decoder(s)':<30s} {decoder_params:>14,}  {two_decoder_params:>16,}")
    print(f"  {'Heatmap head (1x1)':<30s} {heatmap_head_params:>14,}  {heatmap_head_params:>16,}")
    print(f"  {'Edge head (1x1)':<30s} {edge_head_params:>14,}  {edge_head_params:>16,}")
    print(f"  {'─' * 64}")
    print(f"  {'TOTAL':<30s} {total_params:>14,}  {paper_total:>16,}")
    print(f"  {'Trainable':<30s} {trainable_params:>14,}  {paper_total:>16,}")
    print(f"  {'─' * 64}")
    print(f"  {'Difference':<30s} {'':>14s}  {'+' + f'{param_diff:,}':>16s}")
    print(f"  {'% increase':<30s} {'':>14s}  {'+' + f'{pct_increase:.1f}%':>16s}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Architecture:       ResNet-18 encoder + shared U-Net decoder")
    print(f"  Decoder sharing:    conv16s..conv_raw shared, heads independent")
    print(f"  End-to-end forward: ✓ (RGB in → heatmap + contour out)")
    print(f"  Gradient flow:      {'✓ all components' if all_grads_ok else '✗ BROKEN — check above'}")
    print(f"  Total params (shared):   {total_params:>12,}")
    print(f"  Total params (separate): {paper_total:>12,}  (+{pct_increase:.1f}%)")
    print(f"  Input:  ({B}, 3, {H}, {W}) RGB")
    print(f"  Output: ({B}, {HEATMAP_DIM}, {H}, {W}) heatmap + ({B}, {EDGE_DIM}, {H}, {W}) contour")
    print()
