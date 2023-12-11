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
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
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

    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):

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
    parser.add_argument("-c", "--config", default="settings/pretrain_esc50.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=32, type=int,
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
    dataloader = pretrain_dataloader(config,
                                     bucket=False,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=is_dist_avail_and_initialized(),
                                     num_tasks=get_world_size(),
                                     global_rank=get_rank())

    # setup model
    model = ASE(config)
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

    # load evaluation datamodule
    # ac_datamodule = AudioCaptionDataModule(config, "AudioCaps")
    # clotho_datamodule = AudioCaptionDataModule(config, "Clotho")
    # ac_val_loader = ac_datamodule.val_dataloader()
    # clotho_val_loader = clotho_datamodule.val_dataloader()

    """
    prepare ESC50 validation set (zero-shot)
    """
    val_meta = dill.load(open("../data/json_files/ESC50/val_meta_label_modified.pkl", "rb"))
    df_path = "/home/dinghao/Dataset/ESC-50/meta/esc50.csv"
    esc_root_dir = val_meta["dataset_path"]
    # df_path should first remove "/audio", then add "/meta/esc50.csv"
    df_path = val_meta["dataset_path"].replace("/audio", "/meta/esc50.csv")
    # esc_root_dir = "/home/dinghao/Dataset/ESC-50/audio/"
    
    unseen_classes = set()
    for data in val_meta["data"]:
        unseen_classes.add(data["text"])
    unseen_classes = list(unseen_classes)

    val_sorted_df, val_classes = preprocess_esc50(df_path, unseen_classes)



    loss_stats = []
    ac_recall_stats = []
    clotho_recall_stats = []
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

        # # validate on ESC50 unseen classes
        zero_shot(model, device, val_sorted_df, val_classes, esc_root_dir)


        # # validate on AC and Clotho
        # ac_metrics = validate(model, ac_val_loader, device)
        # log_results(ac_metrics, 'AudioCaps', main_logger, test=False)
        # ac_recall_stats.append(ac_metrics["t2a"][0] + ac_metrics["a2t"][0])
        # if ac_recall_stats[-1] >= max(ac_recall_stats) and is_main_process():
        #     sav_obj = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "config": config,
        #         "epoch": epoch
        #     }
        #     torch.save(sav_obj, str(model_output_dir) + "/ac_best_model.pt")
        #
        # clotho_metrics = validate(model, clotho_val_loader, device)
        # log_results(clotho_metrics, 'Clotho', main_logger, test=False)
        # clotho_recall_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["a2t"][0])
        # if clotho_recall_stats[-1] >= max(clotho_recall_stats) and is_main_process():
        #     sav_obj = {
        #         "model": model_without_ddp.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "config": config,
        #         "epoch": epoch
        #     }
        #     torch.save(sav_obj, str(model_output_dir) + "/clotho_best_model.pt")

    # main_logger.info('Evaluation start...')
    # ac_test_loader = ac_datamodule.test_dataloader()
    # clotho_test_loader = clotho_datamodule.test_dataloader()
    #
    # model.load_state_dict(torch.load(str(model_output_dir) + "/best_model.pt")["model"])
    # main_logger.info(f"Evaluation model with smallest loss... epoch:{torch.load(str(model_output_dir) + '/best_model.pt')['epoch']}")
    # ac_metrics = validate(model, ac_test_loader, device)
    # log_results(ac_metrics, 'AudioCaps', main_logger, test=True)
    # clotho_metrics = validate(model, clotho_test_loader, device)
    # log_results(clotho_metrics, 'Clotho', main_logger, test=True)
    #
    # model.load_state_dict(torch.load(str(model_output_dir) + "/ac_best_model.pt")["model"])
    # main_logger.info(f"Evaluation best AudioCaps model... epoch:{torch.load(str(model_output_dir) + '/ac_best_model.pt')['epoch']}")
    # ac_metrics = validate(model, ac_test_loader, device)
    # log_results(ac_metrics, 'AudioCaps', main_logger, test=True)
    # clotho_metrics = validate(model, clotho_test_loader, device)
    # log_results(clotho_metrics, 'Clotho', main_logger, test=True)
    #
    # model.load_state_dict(torch.load(str(model_output_dir) + "/clotho_best_model.pt")["model"])
    # main_logger.info(f"Evaluation best Clotho model... epoch:{torch.load(str(model_output_dir) + '/clotho_best_model.pt')['epoch']}")
    # ac_metrics = validate(model, ac_test_loader, device)
    # log_results(ac_metrics, 'AudioCaps', main_logger, test=True)
    # clotho_metrics = validate(model, clotho_test_loader, device)
    # log_results(clotho_metrics, 'Clotho', main_logger, test=True)

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
    sorted_df = df.sort_values(by=['target'])
    # classes = [x.replace('_', ' ') + " can be heard" for x in sorted_df['category'].unique()]
    classes = [x for x in sorted_df['category'].unique()]
    for i, category in enumerate(classes):
        class_to_idx[category] = i

    sorted_df = sorted_df[sorted_df['category'].isin(unseen_classes)]
    sorted_df = df.sort_values(by=['target'])
    return sorted_df, classes




import librosa
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

@torch.no_grad()
def zero_shot(model, device, sorted_df, classes, esc_root_dir):
    model.eval()
    with torch.no_grad():
        text_embeds = model.encode_text(classes)
        fold_acc = []
        for fold in range(1, 6):
            fold_df = sorted_df[sorted_df['fold'] == fold]
            y_preds, y_labels = [], []
            for file_path, target in tqdm(zip(fold_df["filename"], fold_df["target"]), total=len(fold_df)):
                audio_path = esc_root_dir + file_path
                one_hot_target = torch.zeros(len(classes)).scatter_(0, torch.tensor(target), 1).reshape(1, -1)
                audio, _ = librosa.load(audio_path, sr=32000, mono=True)
                audio = torch.tensor(audio).unsqueeze(0).to(device)
                if audio.shape[-1] < 32000 * 10:
                    pad_length = 32000 * 10 - audio.shape[-1]
                    audio = F.pad(audio, [0, pad_length], "constant", 0.0)
                audio_emb = model.encode_audio(audio)
                similarity = audio_emb @ text_embeds.t()
                y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
                y_preds.append(y_pred)
                y_labels.append(one_hot_target.cpu().numpy())

            y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
            acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
            print('Fold {} Accuracy {}'.format(fold, acc))
            fold_acc.append(acc)

    print('ESC50 Accuracy {}'.format(np.mean(np.array(fold_acc))))


if __name__ == '__main__':
    main()
