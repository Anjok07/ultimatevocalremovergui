import argparse
from datetime import datetime as dt
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from lib import dataset
from lib import nets
from lib import spec_utils


def train_val_split(mix_dir, inst_dir, val_rate, val_filelist_json):
    input_exts = ['.wav', '.m4a', '.3gp', '.oma', '.mp3', '.mp4']
    X_list = sorted([
        os.path.join(mix_dir, fname)
        for fname in os.listdir(mix_dir)
        if os.path.splitext(fname)[1] in input_exts])
    y_list = sorted([
        os.path.join(inst_dir, fname)
        for fname in os.listdir(inst_dir)
        if os.path.splitext(fname)[1] in input_exts])

    filelist = list(zip(X_list, y_list))
    random.shuffle(filelist)

    val_filelist = []
    if val_filelist_json is not None:
        with open(val_filelist_json, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    if len(val_filelist) == 0:
        val_size = int(len(filelist) * val_rate)
        train_filelist = filelist[:-val_size]
        val_filelist = filelist[-val_size:]
    else:
        train_filelist = [
            pair for pair in filelist
            if list(pair) not in val_filelist]

    return train_filelist, val_filelist


def train_inner_epoch(X_train, y_train, model, optimizer, batchsize, instance_loss):
    sum_loss = 0
    model.train()
    aux_crit = nn.L1Loss()
    criterion = nn.L1Loss(reduction='none')
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(X_train), batchsize):
        local_perm = perm[i: i + batchsize]
        X_batch = torch.from_numpy(X_train[local_perm]).cpu()
        y_batch = torch.from_numpy(y_train[local_perm]).cpu()

        model.zero_grad()
        mask, aux = model(X_batch)

        aux_loss = aux_crit(X_batch * aux, y_batch)
        X_batch = spec_utils.crop_center(mask, X_batch, False)
        y_batch = spec_utils.crop_center(mask, y_batch, False)
        abs_diff = criterion(X_batch * mask, y_batch)

        loss = abs_diff.mean() * 0.9 + aux_loss * 0.1
        loss.backward()
        optimizer.step()

        abs_diff_np = abs_diff.detach().cpu().numpy()
        instance_loss[local_perm] += abs_diff_np.mean(axis=(1, 2, 3))
        sum_loss += float(loss.detach().cpu().numpy()) * len(X_batch)

    return sum_loss / len(X_train)


def val_inner_epoch(dataloader, model):
    sum_loss = 0
    model.eval()
    criterion = nn.L1Loss()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.cpu()
            y_batch = y_batch.cpu()
            mask = model.predict(X_batch)
            X_batch = spec_utils.crop_center(mask, X_batch, False)
            y_batch = spec_utils.crop_center(mask, y_batch, False)

            loss = criterion(X_batch * mask, y_batch)
            sum_loss += float(loss.detach().cpu().numpy()) * len(X_batch)

    return sum_loss / len(dataloader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=2019)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--mixture_dataset', '-m', required=True)
    p.add_argument('--instrumental_dataset', '-i', required=True)
    p.add_argument('--learning_rate', type=float, default=0.001)
    p.add_argument('--lr_min', type=float, default=0.0001)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--val_rate', '-v', type=float, default=0.1)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=4)
    p.add_argument('--val_cropsize', '-C', type=int, default=512)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--epoch', '-E', type=int, default=100)
    p.add_argument('--inner_epoch', '-e', type=int, default=4)
    p.add_argument('--oracle_rate', '-O', type=float, default=0)
    p.add_argument('--oracle_drop_rate', '-o', type=float, default=0.5)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', '-d', action='store_true')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = dt.now().strftime('%Y%m%d%H%M%S')

    model = nets.CascadedASPPNet()
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model))
    if args.gpu >= 0:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.lr_min,
        verbose=True)

    train_filelist, val_filelist = train_val_split(
        mix_dir=args.mixture_dataset,
        inst_dir=args.instrumental_dataset,
        val_rate=args.val_rate,
        val_filelist_json=args.val_filelist)

    if args.debug:
        print('### DEBUG MODE')
        train_filelist = train_filelist[:1]
        val_filelist = val_filelist[:1]

    with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
        json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        print(i + 1, os.path.basename(X_fname), os.path.basename(y_fname))

    val_dataset = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        offset=model.offset)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=4)

    log = []
    oracle_X = None
    oracle_y = None
    best_loss = np.inf
    for epoch in range(args.epoch):
        X_train, y_train = dataset.make_training_set(
            train_filelist, args.cropsize, args.patches, args.sr, args.hop_length, model.offset)

        X_train, y_train = dataset.mixup_generator(
            X_train, y_train, args.mixup_rate, args.mixup_alpha)

        if oracle_X is not None and oracle_y is not None:
            perm = np.random.permutation(len(oracle_X))
            X_train[perm] = oracle_X
            y_train[perm] = oracle_y

        print('# epoch', epoch)
        instance_loss = np.zeros(len(X_train), dtype=np.float32)
        for inner_epoch in range(args.inner_epoch):
            print('  * inner epoch {}'.format(inner_epoch))
            train_loss = train_inner_epoch(
                X_train, y_train, model, optimizer, args.batchsize, instance_loss)
            val_loss = val_inner_epoch(val_dataloader, model)

            print('    * training loss = {:.6f}, validation loss = {:.6f}'
                  .format(train_loss * 1000, val_loss * 1000))

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                print('    * best validation loss')
                model_path = 'models/model_iter{}.pth'.format(epoch)
                torch.save(model.state_dict(), model_path)

            log.append([train_loss, val_loss])
            with open('log_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
                json.dump(log, f, ensure_ascii=False)

        if args.oracle_rate > 0:
            instance_loss /= args.inner_epoch
            oracle_X, oracle_y, idx = dataset.get_oracle_data(
                X_train, y_train, instance_loss, args.oracle_rate, args.oracle_drop_rate)
            print('  * oracle loss = {:.6f}'.format(instance_loss[idx].mean()))

        del X_train, y_train
        gc.collect()


if __name__ == '__main__':
    main()
