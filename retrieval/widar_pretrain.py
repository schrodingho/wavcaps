#!/usr/bin/env python3
# coding: utf-8
# @Original Author  : Xinhao Mei @CVSSP, University of Surrey
# @Author  : Dinghao Xue @ES, ST, CS, TU Delft
# @E-mail  : d.xue@student.tudelft.nl

import dill
import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
import ruamel.yaml as yaml
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader, pretrain_widar_dataloader
from models.ase_model import ASE, ASE_widar
import torch.distributed as dist
import pandas as pd
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    setup_seed,
    AverageMeter, t2a, a2t, set_logger, log_results,
)


def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()

    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio, text, idx, y_true) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()

        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        loss = model(audio, text, idx)

        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    wandb.log({"loss": epoch_loss.avg,
               "epoch": epoch})

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/pretrain_widar.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="Widar_Exp", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=64, type=int,
                        help="Batch size.")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_ub8k_esc50_vggsound.json', type=str,
                        help="Blacklist file.")
    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        yaml_loader = yaml.YAML(typ="safe")
        config = yaml_loader.load(f)

    config["audio_encoder_args"]["type"] = args.model_type
    config["audio_encoder_args"]["model"] = args.model
    config["audio_args"]["max_length"] = args.max_length
    config["optim_args"]["lr"] = args.lr
    config["blacklist"] = None
    config["data_args"]["batch_size"] = args.batch_size

    # setup distribution mode
    init_distributed_mode(config["dist_args"])
    device = torch.device(config["device"])

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    exp_name = exp_name + f"_lr_{args.lr}_seed_{seed}"

    wandb.init(
        project="AT-retrieval",
        name=exp_name,
        config=config
    )

    # create pretrain dataloader
    # dataloader = pretrain_dataloader(config,
    #                                  bucket=False,
    #                                  bucket_boundaries=(5, 30, 6),
    #                                  is_distributed=is_dist_avail_and_initialized(),
    #                                  num_tasks=get_world_size(),
    #                                  global_rank=get_rank())

    dataloader = pretrain_widar_dataloader(config, widar_file_path="../data/json_files/Widar/train_meta.pkl")

    """
    prepare Widar validation set (zero-shot) seen and unseen classes
    """
    val_seen_loader = pretrain_widar_dataloader(config, widar_file_path="../data/json_files/Widar/val_seen_meta.pkl", val_data=True)
    val_unseen_loader = pretrain_widar_dataloader(config, widar_file_path="../data/json_files/Widar/val_unseen_meta.pkl", val_data=True)
    val_seen_text_label_dict = dill.load(open("../data/json_files/Widar/val_seen_meta.pkl", 'rb'))["label_text_dict"]
    val_unseen_text_label_dict = dill.load(open("../data/json_files/Widar/val_unseen_meta.pkl", 'rb'))["label_text_dict"]
    val_seen_classes = list(val_seen_text_label_dict.keys())
    val_unseen_classes = list(val_unseen_text_label_dict.keys())
    val_unseen_classes.extend(val_seen_classes)

    # setup model
    model = ASE_widar(config)
    model = model.to(device)
    wandb.watch(model)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]
    # max_epoch = 5

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')

    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model
        )
        model_without_ddp = model.module


    loss_stats = []
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]
        loss_stats.append(loss)

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        if loss <= min(loss_stats) and is_main_process():
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }

            torch.save(sav_obj, str(model_output_dir) + "/best_model.pt")

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()
        # # validate on Widar seen classes
        zero_shot(model, val_seen_loader, device, val_seen_classes, epoch=epoch, type="seen")
        # # validate on Widar unseen classes
        zero_shot(model, val_unseen_loader, device, val_unseen_classes, epoch=epoch, type="unseen")


    main_logger.info("Done.")
    wandb.finish()


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        audio_embeds = model.encode_audio(audio)
        text_embeds = model.encode_text(text)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}


def preprocess_esc50(df_path, unseen_classes):
    df = pd.read_csv(df_path)
    class_to_idx = {}
    sorted_df = df[df['category'].isin(unseen_classes)]
    # re index the target by the order of the unseen classes
    sorted_df['target'] = sorted_df['category'].map(lambda x: unseen_classes.index(x))
    classes = unseen_classes
    for i, category in enumerate(classes):
        class_to_idx[category] = i
    return sorted_df, classes


import librosa
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


@torch.no_grad()
def zero_shot(model, dataloader, device, class_text_list, epoch=None, type="seen"):
    model.eval()
    text_embeds = model.encode_text(class_text_list)
    targets = torch.zeros(0).to(device)
    similarity = torch.zeros(0).to(device)
    for batch_idx, (wifi, _, idx, y_true) in tqdm(enumerate(dataloader), total=len(dataloader)):
        wifi = wifi.to(device)
        wifi_embeds = model.encode_wifi(wifi)
        logits = wifi_embeds @ text_embeds.t()
        sim_batch = F.log_softmax(logits, dim=1)
        targets = torch.cat((targets, y_true.to(device)), dim=0)
        similarity = torch.cat((similarity, sim_batch), dim=0)
    values, indices = similarity.topk(5)
    top1 = indices[:, 0] == targets
    top1_acc = top1.sum() / len(top1)
    if not epoch:
        epoch = -1
    print(f'Type: {type}, epoch {epoch}, val/top1 {top1_acc}')


if __name__ == '__main__':
    main()
