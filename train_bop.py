# train_bop.py
# Derived from the original main.py by the ContourPose authors.
# Adds BOP dataset support via DALI dataloader and wandb logging.
# The original main.py is preserved unchanged for reference.
# See CHANGELOG.md for a full list of modifications and rationale.
# Usage: python train_bop.py \
#   --gin_config configs/contourpose_bop.gin \
#   --bop_root /path/to/RTLESS_BOP \
#   --class_type obj1 \
#   --obj_id 1

import os
import time
from pathlib import Path
import argparse

import gin
import wandb
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from network import ContourPose
from dataset.data_utils import (
    load_keypoints,
    create_bop_validation_setup,
)


def train(model, dataloader, device, epoch):
    """Train for one epoch.

    # optimize_params and log_epoch are clean replacements for the
    # inline forward/backward pass in the original main.py.
    # See network/contourpose.py for implementation.
    """
    model.train()

    model_module = model.module if isinstance(model, nn.DataParallel) else model
    model_module.reset_logger(epoch=epoch, data_size=len(dataloader))

    for batch_idx, data in tqdm(enumerate(dataloader), desc="Batch", leave=False):
        model_module.optimize_params(data, batch_idx=batch_idx, model_wrapper=model)

    model_module.log_epoch()


def load_network(net, model_dir, optimizer, resume=True, epoch=-1, strict=True):
    # Preserved verbatim from original main.py
    if not resume:
        return 0
    if not os.path.exists(model_dir):
        return 0
    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir) if "pkl" in pth]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch

    print("Load model: {}".format(os.path.join(model_dir, "{}.pkl".format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pkl".format(pth)))
    try:
        net.load_state_dict(pretrained_model['net'], strict=strict)
        optimizer.load_state_dict(pretrained_model['optimizer'])
    except KeyError:
        net.load_state_dict(pretrained_model, strict=strict)
    return pth


@gin.configurable
def main(args, model_cls=ContourPose, lr=1e-3, batch_size=8, epochs=150,
         lr_step_size=20, lr_gamma=0.5, val_interval=5, viz_interval=10,
         compute_edge_input=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    # WandB init
    if args.use_wandb:
        wandb.init(
            project="contourpose-bop",
            name=args.run_name,
            tags=[args.class_type, model_cls.__name__],
            config={
                "class_type": args.class_type,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "gin_config": gin.operative_config_str(),
            },
        )

    # --bop_root is the top-level dataset root (e.g. /data/RTLESS_BOP).
    # keypoints/ and Valid3D/ live there, but the DALI loader needs the
    # per-object scene directory one level deeper: train_pbr/{obj_id:06d}/.
    # We keep bop_dataset_root for the model and patch args.bop_root to the
    # scene dir only for the duration of the DALI setup call.
    bop_dataset_root = args.bop_root
    bop_scene_dir = str(Path(args.bop_root) / "train_pbr" / f"{args.obj_id:06d}")
    print(f"[Data] Dataset root: {bop_dataset_root}")
    print(f"[Data] Scene dir:    {bop_scene_dir}")

    # Load keypoints from dataset root (keypoints_dir resolved against bop_dataset_root)
    corners = load_keypoints(args)

    # Expose gin params on args for downstream utilities that read them
    args.batch_size = batch_size
    args.compute_edge_input = compute_edge_input

    # Build validation + train loaders with 80/20 cross-val split.
    # create_bop_validation_setup reads args.bop_root as the DALI data_dir, so
    # temporarily point it at the scene directory, then restore.
    args.bop_root = bop_scene_dir
    val_setup = create_bop_validation_setup(args, num_gpus=2)
    args.bop_root = bop_dataset_root
    train_loader      = val_setup["train_loader"]
    fixed_val_loader  = val_setup["fixed_val_loader"]
    random_val_loader = val_setup["random_val_loader"]
    fixed_batch       = val_setup["fixed_batch"]

    model_kwargs = {
        "heatmap_dim": corners.shape[0],
        # data_root must have keypoints/ and Valid3D/ — that's the dataset root,
        # not the scene dir used for DALI loading.
        "data_root": bop_dataset_root,
        "class_type": args.class_type,
        "init_lr": lr,
        "lr_step_size": lr_step_size,
        "lr_gamma": lr_gamma,
    }

    ContourNet = model_cls(**model_kwargs)
    # .cuda() before DataParallel is intentional: moves all parameters to GPU
    # first so DataParallel can scatter them across devices without a host-device
    # round-trip on every forward pass.
    ContourNet = ContourNet.cuda()
    ContourNet = nn.DataParallel(ContourNet, device_ids=[0, 1])

    model_module = ContourNet.module if isinstance(ContourNet, nn.DataParallel) else ContourNet

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Class:             {args.class_type}")
    print(f"  BOP root:          {bop_dataset_root}")
    print(f"  Scene dir:         {bop_scene_dir}")
    print(f"  Batch size:        {batch_size}")
    print(f"  LR:                {lr}")
    print(f"  LR step size:      {lr_step_size} epochs")
    print(f"  LR gamma:          {lr_gamma}")
    print(f"  Epochs:            {epochs}")
    print(f"  Val interval:      {val_interval}")
    print(f"  Viz interval:      {viz_interval}")
    print("=" * 60 + "\n")

    start_epoch = 1

    # Resume from checkpoint
    if args.resume:
        ckpt_info = model_module.load_checkpoint(
            checkpoint_path=args.checkpoint_path,
            run_name=args.run_name,
            load_best=args.load_best,
        )
        if ckpt_info is not None:
            start_epoch = ckpt_info['epoch'] + 1
            print(f"[Checkpoint] Resuming from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs"):
        train(ContourNet, train_loader, device, epoch)

        model_module = ContourNet.module if isinstance(ContourNet, nn.DataParallel) else ContourNet
        model_module.lr_sched.step()

        if epoch % val_interval == 0 and fixed_val_loader is not None:
            print(f"\n[Validation] Epoch {epoch}: fixed set")
            model_module.validate_with_pose_metrics(
                fixed_val_loader, args.class_type, epoch, device,
                val_name="val_fixed"
            )

        if epoch % viz_interval == 0 and fixed_batch is not None:
            print(f"\n[Visualization] Epoch {epoch}")
            random_batch = next(iter(random_val_loader))[0]
            model_module.visualize_batches(
                fixed_batch, random_batch, args.class_type, epoch, device
            )

        if epoch % 10 == 0:
            run_name = wandb.run.name if args.use_wandb and wandb.run is not None else None
            model_module.save_checkpoint(
                epoch=epoch,
                model_wrapper=ContourNet,
                save_best=False,
                run_name=run_name,
            )

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ContourPose BOP training script")

    parser.add_argument("--class_type",      type=str, default="obj1")
    parser.add_argument("--bop_root",        type=str, default=None)
    parser.add_argument("--obj_id",          type=int, default=1)
    parser.add_argument("--keypoints_dir",   type=str, default="keypoints")
    parser.add_argument("--valid3d_dir",     type=str, default="Valid3D")
    parser.add_argument("--img_size",        type=int, nargs=2, default=[480, 640])
    parser.add_argument("--background_dir",  type=str, default=None)
    parser.add_argument("--use_wandb",       action="store_true", default=True)
    parser.add_argument("--no_wandb",        action="store_false", dest="use_wandb")
    parser.add_argument("--resume",          action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--run_name",        type=str, default=None)
    parser.add_argument("--load_best",       action="store_true", default=False)
    parser.add_argument("--gin_config",      type=str, nargs="+", default=[])
    parser.add_argument("--gin_param",       type=str, nargs="+", default=[])

    args = parser.parse_args()
    args.img_size = tuple(args.img_size)

    gin.parse_config_files_and_bindings(args.gin_config, args.gin_param)

    main(args)
