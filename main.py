from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import MultiModalTransformer, MultiModalConv
from data import dataloaders
from utils import seed_everything, metrics, plot_roc_auc, adjust_learning_rate
# seed_everything()

def train_epoch(model, optimizer, criterion, loader, modalities, writer=None, epoch=0):
    y_true, y_logits = [], []
    len_loader = int(np.ceil(len(loader.dataset) / loader.batch_size))
    for i, sample in enumerate(tqdm(loader, leave=False)):
        # if i == 10: break
        x = {mod: sample[mod].cuda() for mod in modalities+['input_ids','attention_mask']}
        y = sample['y'].cuda()
        out = model(x)
        loss = criterion(out, y.float())
        loss.backward()
        if writer is not None:
            writer.add_scalar(f'{loader.dataset.split}/loss', loss.item(), global_step=i+epoch*len_loader)
        # if i % 2 == 0:
        optimizer.step()
        optimizer.zero_grad()
        logits = torch.sigmoid(out)
        y_true.append(y.cpu())
        y_logits.append(logits.detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_logits = torch.cat(y_logits, dim=0)
    return y_true, y_logits

@torch.no_grad()
def test_epoch(model, criterion, loader, modalities, device='cuda', writer=None, epoch=0):
    y_true, y_logits = [], []
    len_loader = int(np.ceil(len(loader.dataset) / loader.batch_size))
    for i, sample in enumerate(tqdm(loader, leave=False)):
        # if i == 10: break
        x = {mod: sample[mod].to(device) for mod in modalities+['input_ids','attention_mask']}
        y = sample['y'].to(device)
        out = model(x)
        loss = criterion(out, y.float())
        if writer is not None:
            writer.add_scalar(f'{loader.dataset.split}/loss', loss.item(), global_step=i+epoch*len_loader)
        logits = torch.sigmoid(out)
        y_true.append(y.cpu())
        y_logits.append(logits.detach().cpu())
    y_true = torch.cat(y_true, dim=0)
    y_logits = torch.cat(y_logits, dim=0)
    return y_true, y_logits

def main(args):
    exp_name = f'{args.model}_' + '_'.join(args.modalities)
    if args.with_text:
        exp_name += '_text'
    if args.with_diagnoses:
        exp_name += '_diagnoses'
    ts = str(datetime.now().timestamp())
    run_name = f'{exp_name}/{ts}'
    writer = SummaryWriter(log_dir=f'./runs/{args.task}/{run_name}')

    if args.model == 'swin':
        # img_model_name = 'microsoft/swin-base-patch4-window7-224-in22k'
        # image_size = 224
        img_model_name = 'microsoft/swin-large-patch4-window12-384-in22k'
        image_size = 384
        model = MultiModalTransformer(
            task=args.task,
            img_model_name=img_model_name,
            img_modalities=args.modalities,
            with_text=args.with_text
        ).cuda()
        model = nn.DataParallel(model)
        
    elif args.model == 'vit':
        img_model_name = 'google/vit-base-patch16-224'
        image_size = 224
        model = MultiModalTransformer(
            task=args.task,
            img_model_name=img_model_name,
            img_modalities=args.modalities,
            with_text=args.with_text
        ).cuda()
    elif args.model == 'conv':
        image_size = 224
        img_model_name = 'google/vit-base-patch16-224'
        model = MultiModalConv(
            task=args.task,
            # img_model_name='convnext_base',
            img_modalities=args.modalities,
            with_text=args.with_text
        ).cuda()
    
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
    
    if args.task == 'phenotyping':
        criterion = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor(4168 / 717).cuda() # n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lrs[0], weight_decay=args.weight_decay)

    trainloader, valloader, testloader = dataloaders(
        task=args.task,
        image_size=image_size,
        modalities=args.modalities,
        batch_size=args.batch_size,
        root=args.root,
        with_diagnoses=args.with_diagnoses,
        img_model_name=img_model_name
    )

    sample = next(iter(testloader))
    images = torch.cat([sample[m] for m in args.modalities], dim=-2)
    writer.add_images('input_images', images)
    if args.with_text:
        text = '\n'.join([testloader.dataset._text(i) for i in range(args.batch_size)])
        writer.add_text('input_text', text)

    pbar = tqdm(range(args.n_epochs))
    for i,e in enumerate(pbar):
        lr = args.lrs[i] if i < len(args.lrs) else args.lrs[-1]
        adjust_learning_rate(optimizer, lr)
        res_train = train_epoch(model, optimizer, criterion, trainloader, args.modalities, writer=writer)
        res_val = test_epoch(model, criterion, valloader, args.modalities, writer=writer)
        res_test = test_epoch(model, criterion, testloader, args.modalities, writer=writer)
        
        train_metrics = np.array([metrics(res_train[0][:,j], res_train[1][:,j]) for j in range(res_train[0].shape[1])]).mean(0)
        val_metrics = np.array([metrics(res_val[0][:,j], res_val[1][:,j]) for j in range(res_val[0].shape[1])]).mean(0)
        test_metrics = np.array([metrics(res_test[0][:,j], res_test[1][:,j]) for j in range(res_test[0].shape[1])]).mean(0)

        for mode, e_metrics in zip(['train','val','test'], [train_metrics,val_metrics,test_metrics]):
            for j, k in enumerate(['rocauc','auprc','balanced_accuracy']):
                writer.add_scalar(f'{mode}/{k}', e_metrics[j], global_step=e)
        if args.task == 'inhospital_mortality':
            fig = plot_roc_auc(['train', *res_train], ['val', *res_val], ['test', *res_test])
            writer.add_figure('rocauc_fig', fig, global_step=e, close=True)
        torch.save(model.state_dict(), Path(f'./runs/{args.task}') / run_name / 'checkpoint.pt')
        pbar.set_description(f'Train: {train_metrics[0]:.4f}, Val: {val_metrics[0]:.4f}, Test: {test_metrics[0]:.4f}')

    hparams = args.__dict__
    hparams['lrs'] = torch.tensor(hparams['lrs'])
    hparams['modalities'] = '_'.join(hparams['modalities'])
    if args.with_text:
        hparams['modalities'] += '_text'
    del hparams['deploy_as_job']
    # del hparams['root']
    writer.add_hparams(
        hparams,
        {'train_rocauc': train_metrics[0], 'val_auroc': val_metrics[0], 'test_auroc': test_metrics[0]},
        # run_name=run_name
    )

if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--task', default='inhospital_mortality')
    parser.add_argument('--model', default='swin')
    parser.add_argument('--modalities', nargs='+', default=['cxr','lab','ecg','med'])
    parser.add_argument('--with_text', action='store_true', default=False)
    parser.add_argument('--with_diagnoses', action='store_true', default=False)
    parser.add_argument('--root', default='/mnt/hdd/data/MMMedViT_data/data')
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=3e-8)
    parser.add_argument('--lrs', nargs='+', type=float, default=[1e-5])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ckpt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deploy_as_job', action='store_true', default=False)
    args = parser.parse_args()

    if args.deploy_as_job:
        pass
    else:
        seed_everything(args.seed)
        main(args)