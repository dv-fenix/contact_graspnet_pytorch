import argparse
import os
import sys
import time
from datetime import datetime
from genericpath import exists

import numpy as np
import torch
import wandb  # Replaced tensorboardX
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

os.environ["PYOPENGL_PLATFORM"] = "egl"  # To get pyrender to work headless

# Import pointnet library
CONTACT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, "Pointnet_Pointnet2_pytorch"))

import config_utils
from acronym_dataloader import AcryonymDataset

from contact_graspnet_pytorch import utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_graspnet import (ContactGraspnet,
                                                       ContactGraspnetLoss)


def train(global_config, log_dir):
    """
    Trains Contact-GraspNet. Configure the training process by modifying the
    config.yaml file.

    Arguments:
        global_config {dict} -- config dict
        log_dir {str} -- Checkpoint directory
    """
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    batch_size = global_config["OPTIMIZER"]["batch_size"]
    num_workers = 12  # Keep unchanged for minimal edits

    train_dataset = AcryonymDataset(
        global_config, train=True, device=device, use_saved_renders=True
    )
    test_dataset = AcryonymDataset(
        global_config, train=False, device=device, use_saved_renders=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    grasp_estimator = ContactGraspnet(global_config, device).to(device)
    loss_fn = ContactGraspnetLoss(global_config, device).to(device)

    if global_config["OPTIMIZER"].get("freeze_backbone_for_confidence_only", False):
        for p in grasp_estimator.parameters():
            p.requires_grad = False

        for p in grasp_estimator.binary_seg_head.parameters():
            p.requires_grad = True

        for name, module in grasp_estimator.named_children():
            if name != "binary_seg_head":
                module.eval()

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, grasp_estimator.parameters()),
        lr=global_config["OPTIMIZER"]["learning_rate"],
    )

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_io = CheckpointIO(checkpoint_dir, model=grasp_estimator, opt=opt)

    try:
        load_dict = checkpoint_io.load("model.pt")
    except FileExistsError:
        load_dict = dict()

    cur_epoch = load_dict.get("epoch_it", 0)
    it = load_dict.get("it", 0)
    metric_val_best = load_dict.get("loss_val_best", np.inf)

    grasp_estimator, opt, train_dataloader, test_dataloader = accelerator.prepare(
        grasp_estimator, opt, train_dataloader, test_dataloader
    )

    print_every = (
        global_config["OPTIMIZER"]["print_every"]
        if "print_every" in global_config["OPTIMIZER"]
        else 0
    )
    checkpoint_every = (
        global_config["OPTIMIZER"]["checkpoint_every"]
        if "checkpoint_every" in global_config["OPTIMIZER"]
        else 0
    )
    backup_every = (
        global_config["OPTIMIZER"]["backup_every"]
        if "backup_every" in global_config["OPTIMIZER"]
        else 0
    )
    val_every = (
        global_config["OPTIMIZER"]["val_every"]
        if "val_every" in global_config["OPTIMIZER"]
        else 0
    )

    log_string(f"Accelerator device: {device}")
    log_string(f"Num processes: {accelerator.num_processes}")
    log_string(f"Distributed type: {accelerator.distributed_type}")

    for epoch_it in range(cur_epoch, global_config["OPTIMIZER"]["max_epoch"]):
        log_string("**** EPOCH %03d ****" % epoch_it)
        grasp_estimator.train()

        if global_config["OPTIMIZER"].get("freeze_backbone_for_confidence_only", False):
            for name, module in grasp_estimator.named_children():
                if name != "binary_seg_head":
                    module.eval()

        pbar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        for i, data in enumerate(pbar):
            utils.send_dict_to_device(data, device)
            # Target contains input and target values
            pc_cam = data["pc_cam"]

            pred = grasp_estimator(pc_cam)
            loss, loss_info = loss_fn(pred, data)

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            if accelerator.is_main_process:
                for k, v in loss_info.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().item()
                    wandb.log({f"train/{k}": v}, step=it)

            if checkpoint_every and it % checkpoint_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_io.save(
                        "model.pt",
                        epoch_it=epoch_it,
                        it=it,
                        loss_val_best=metric_val_best,
                    )

            if backup_every and it % backup_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_io.save(
                        "model_%d.pt" % it,
                        epoch_it=epoch_it,
                        it=it,
                        loss_val_best=metric_val_best,
                    )

            if accelerator.is_main_process:
                wandb.log({"train/loss": loss.item()}, step=it)

            if accelerator.is_local_main_process:
                pbar.set_postfix({"loss": loss.item(), "epoch": epoch_it})

            it += 1

        # -- Run Validation -- #
        if val_every and epoch_it % val_every == 0:
            grasp_estimator.eval()
            with torch.no_grad():
                loss_log = []
                for val_it, data in enumerate(
                    tqdm(test_dataloader, disable=not accelerator.is_local_main_process)
                ):
                    utils.send_dict_to_device(data, device)
                    pc_cam = data["pc_cam"]

                    pred = grasp_estimator(pc_cam)
                    loss, loss_info = loss_fn(pred, data)

                    gathered_loss = accelerator.gather_for_metrics(
                        loss.detach().reshape(1)
                    )
                    loss_log.extend(gathered_loss.cpu().numpy().tolist())

                val_loss = np.mean(loss_log)

                if accelerator.is_main_process:
                    wandb.log({"val/val_loss": val_loss}, step=it)

            if val_loss < metric_val_best:
                metric_val_best = val_loss
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_io.save(
                        "model_best.pt",
                        epoch_it=epoch_it,
                        it=it,
                        loss_val_best=metric_val_best,
                    )


if __name__ == "__main__":
    # Usage:
    # To continue training:
    #   accelerate launch train.py --ckpt_dir {current_ckpt_dir}
    #
    # To start training from scratch:
    #   accelerate launch train.py
    #
    # Example multi-GPU:
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu train.py --ckpt_dir /path/to/ckpt

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint dir")
    parser.add_argument(
        "--data_path", type=str, default=None, help="Grasp data root dir"
    )
    parser.add_argument("--max_epoch", type=int, default=None, help="Epochs to run")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch Size during training"
    )
    parser.add_argument(
        "--arg_configs",
        nargs="*",
        type=str,
        default=[],
        help="overwrite config parameters",
    )
    parser.add_argument(
        "--freeze_backbone_for_confidence_only",
        action="store_true",
        help="Freeze backbone and non-confidence heads; train only binary_seg_head",
    )
    FLAGS = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    ckpt_dir = FLAGS.ckpt_dir
    if ckpt_dir is None:
        # ckpt_dir is contact_graspnet_year_month_day_hour_minute_second
        ckpt_dir = os.path.join(
            CONTACT_DIR,
            "../",
            f'checkpoints/contact_graspnet_{datetime.now().strftime("Y%YM%mD%d_H%HM%M")}',
        )

    data_path = FLAGS.data_path
    if data_path is None:
        data_path = os.path.join(CONTACT_DIR, "../", "acronym/")

    if FLAGS.freeze_backbone_for_confidence_only:
        FLAGS.arg_configs.append("OPTIMIZER.freeze_backbone_for_confidence_only:true")

    if accelerator.is_main_process:
        if not os.path.exists(ckpt_dir):
            if not os.path.exists(os.path.dirname(ckpt_dir)):
                ckpt_dir = os.path.join(BASE_DIR, ckpt_dir)
            os.makedirs(ckpt_dir, exist_ok=True)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.system(
            "cp {} {}".format(
                os.path.join(CONTACT_DIR, "contact_graspnet.py"), ckpt_dir
            )
        )  # bkp of model def
        os.system(
            "cp {} {}".format(os.path.join(CONTACT_DIR, "train.py"), ckpt_dir)
        )  # bkp of train procedure

    accelerator.wait_for_everyone()

    LOG_FOUT = open(os.path.join(ckpt_dir, "log_train.txt"), "a")

    if accelerator.is_main_process:
        LOG_FOUT.write(str(FLAGS) + "\n")
        LOG_FOUT.flush()

    def log_string(out_str):
        if accelerator.is_main_process:
            LOG_FOUT.write(out_str + "\n")
            LOG_FOUT.flush()
            print(out_str)

    global_config = config_utils.load_config(
        ckpt_dir,
        batch_size=FLAGS.batch_size,
        max_epoch=FLAGS.max_epoch,
        data_path=FLAGS.data_path,
        arg_configs=FLAGS.arg_configs,
        save=accelerator.is_main_process,
    )

    log_string(str(global_config))
    log_string("pid: %s" % (str(os.getpid())))

    # Initialize WandB only on main process
    if accelerator.is_main_process:
        wandb.init(
            project="contact-graspnet",
            config=global_config,
            dir=ckpt_dir,
            name=os.path.basename(ckpt_dir),
        )

    train(global_config, ckpt_dir)

    if accelerator.is_main_process:
        wandb.finish()

    LOG_FOUT.close()
