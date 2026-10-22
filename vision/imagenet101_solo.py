"""SOLO / BP / DSN with ResNet-101 on ImageNet-1k.

Built by composition: MDHeadResNet32 is instantiated with ARCH=r50 (SPLIT is removed from the
environment first, since the base model asserts a 16-block split), to_r101() rebuilds the backbone with
the R101 block table (3,4,23,3 = 33 bottlenecks) and rebuilds groups/heads/module_params/opts from
SPLIT, then imagenet_solo.to_imagenet_r50() swaps in the ImageNet stem. The training loop matches
imagenet_solo.main.

Default K=4 split SPLIT=3,6,11,13 (balanced by activation bytes):
  m1 = stem + layer1 | m2 = layer2 + 2 x layer3 | m3 = 11 x layer3 | m4 = 10 x layer3 + layer4

Usage (data at ./data/imagenet; on 40GB GPUs use BATCH=128 or 192 for all arms):
  BP:   MODE=bp EPOCHS=90 BATCH=256 CKPT=ckpt/in101_bp.pt python -u imagenet101_solo.py
  SOLO: MODE=solo GS_MODE=jacobi SCHED=s3 K_MODULES=4 SPLIT=3,6,11,13 \\
          STRIDED_HEAD=1 HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 \\
          EPOCHS=90 BATCH=256 CKPT=ckpt/in101_solo_k4.pt python -u imagenet101_solo.py
  DSN:  MODE=dsn K_MODULES=4 SPLIT=3,6,11,13 STRIDED_HEAD=1 HEAD_DEPTH=3 \\
          HEAD_WIDEN=2 EPOCHS=90 BATCH=256 CKPT=ckpt/in101_dsn_k4.pt python -u imagenet101_solo.py
  Smoke test: VERIFY=1 python imagenet101_solo.py
Env: as imagenet_solo.py.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from solo_cifar import (MDHeadResNet32, Bottleneck, ReadoutHead, evaluate, DEVICE)
from imagenet_solo import build_loaders, to_imagenet_r50

R101_STAGES = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]   # 33 blocks


def to_r101(model, num_classes, lr, wd, split):
    """Rebuild an ARCH=r50 model's backbone as R101 (33 bottlenecks) with new heads and optimizers.
    The stem stays CIFAR-style; to_imagenet_r50 replaces it afterwards."""
    model.blocks = nn.ModuleList()
    block_ch = []
    in_ch = 64
    for planes, n, first_stride in R101_STAGES:
        for b in range(n):
            stride = first_stride if b == 0 else 1
            model.blocks.append(Bottleneck(in_ch, planes, stride))
            in_ch = planes * Bottleneck.expansion
            block_ch.append(in_ch)
    model.n_blocks = len(model.blocks)
    model.fc = nn.Linear(2048, num_classes)

    if split:
        sizes = [int(s) for s in split.split(',')]
        assert sum(sizes) == model.n_blocks, f'SPLIT {sizes} != {model.n_blocks} blocks'
        idx, model.groups = 0, []
        for s in sizes:
            model.groups.append(list(range(idx, idx + s)))
            idx += s
        K = len(sizes)
    else:
        K = model.K
        model.groups = [list(g) for g in np.array_split(np.arange(model.n_blocks), K)]
    model.K, model.R = K, K - 1

    head_widen = float(os.environ.get('HEAD_WIDEN', '1'))
    head_depth = int(os.environ.get('HEAD_DEPTH', '1'))
    strided = os.environ.get('STRIDED_HEAD', '0') == '1'
    model.heads = nn.ModuleList(
        [ReadoutHead(block_ch[model.groups[k][-1]], num_classes, widen=head_widen,
                     depth=head_depth, strided=strided) for k in range(K - 1)])
    model.ach_ema = [math.log(num_classes)] * (K - 1)
    model._init_weights()

    model.module_params = []
    for k in range(K):
        params = list(model.stem.parameters()) if k == 0 else []
        for bi in model.groups[k]:
            params += list(model.blocks[bi].parameters())
        params += list(model.heads[k].parameters()) if k < K - 1 \
            else list(model.fc.parameters())
        model.module_params.append(params)
    model.opts = [SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
                  for p in model.module_params]
    model.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.opts[0], T_max=model.scheduler.T_max, eta_min=0.0)
    return model


def build_model(n_cls, base_lr, n_epochs, mode, wd, split):
    model = MDHeadResNet32(
        num_classes=n_cls, lr=base_lr, K=max(2, len(split.split(',')) if split else
                                             int(os.environ.get('K_MODULES', '2'))),
        gs_mode=os.environ.get('GS_MODE', 'jacobi'), mode=mode,
        sched=os.environ.get('SCHED', 's3'),
        lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
        a_tgt=float(os.environ.get('A_TGT', '0.15')),
        alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
        mu=float(os.environ.get('MU', '1.0')),
        ratchet=os.environ.get('RATCHET', '1') == '1',
        delta_t=float(os.environ.get('DELTA_T', '0.2')),
        aux_ce=float(os.environ.get('AUX_CE', '0.1')),
        t_max=n_epochs)
    model = to_r101(model, n_cls, base_lr, wd, split)
    model = to_imagenet_r50(model, base_lr, wd)          # ImageNet stem; rebuilds module 0
    return model


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(os.environ.get('SEED', '42'))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    data_root = os.environ.get('DATA_ROOT', './data/imagenet')
    n_epochs = int(os.environ.get('EPOCHS', '90'))
    batch_size = int(os.environ.get('BATCH', '256'))
    warmup = int(os.environ.get('WARMUP', '5'))
    base_lr = float(os.environ.get('LR', '0.1')) * batch_size / 256
    mode = os.environ.get('MODE', 'solo')
    os.environ['ARCH'] = 'r50'                            # base instantiation; rebuilt as R101 afterwards
    os.environ.setdefault('EPS_FLOOR', '1e-5')            # C=1000 feasibility
    wd = float(os.environ.setdefault('WD', '1e-4'))
    split = os.environ.pop('SPLIT', '')                   # hidden from the base model's 16-block assertion

    from tqdm import tqdm
    train_loader, val_loader, n_cls = build_loaders(
        data_root, batch_size, int(os.environ.get('TRAIN_RES', '224')),
        int(os.environ.get('VAL_RES', '224')), int(os.environ.get('WORKERS', '8')))

    model = build_model(n_cls, base_lr, n_epochs, mode, wd, split).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'device={DEVICE} | {mode.upper()} R101 IMAGENET-1K K={model.K} '
          f'split={[len(g) for g in model.groups]} bs={batch_size} lr={base_lr:.3f} '
          f'warmup={warmup} eps={os.environ["EPS_FLOOR"]} params={n_par:.1f}M '
          f'TF32+channels_last', flush=True)

    best, best_ens, start_ep = 0.0, 0.0, 0
    ckpt_path = os.environ.get('CKPT', '')
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck['model'])
        for o, s in zip(model.opts, ck.get('opts', [])):
            o.load_state_dict(s)
        start_ep, best = ck['epoch'], ck['best']
        print(f'  [ckpt] resumed @ epoch {start_ep} best={best:.4f}', flush=True)

    def set_lr(ep):
        if ep < warmup:
            lr = base_lr * (ep + 1) / warmup
        else:
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(n_epochs - warmup, 1)))
        for opt in model.opts:
            for g in opt.param_groups:
                g['lr'] = lr
        return lr

    for epoch in range(start_ep, n_epochs):
        lr_now = set_lr(epoch)
        model.train()
        correct, seen = 0, 0
        c_acc = dict(u0=0.0, u_last=0.0, bound=0.0, n=0)
        pbar = tqdm(train_loader, desc=f'Ep {epoch + 1}/{n_epochs} lr={lr_now:.4f}')
        for data, target in pbar:
            data = data.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            target = target.to(DEVICE, non_blocking=True)
            out, loss, cert = model.forward_with_update(data, target)
            correct += out.argmax(1).eq(target).sum().item(); seen += target.size(0)
            if cert:
                for kk in ('u0', 'u_last', 'bound'):
                    c_acc[kk] += cert[kk]
                c_acc['n'] += 1
            pbar.set_postfix({'loss': f'{loss:.3f}', 'acc': f'{correct / seen:.4f}'})
        acc, acc_ens = evaluate(model, val_loader, DEVICE)
        best = max(best, acc)
        ens_str = ''
        if acc_ens is not None:
            best_ens = max(best_ens, acc_ens)
            ens_str = f' ens={acc_ens:.4f} best_ens={best_ens:.4f}'
        mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9 if torch.cuda.is_available() else 0.0
        cert_str = ''
        if c_acc['n']:
            n = c_acc['n']
            u_l, bd = c_acc['u_last'] / n, c_acc['bound'] / n
            cert_str = (f' | cert: u0={c_acc["u0"] / n:.3f} u_last={u_l:.3f} '
                        f'bound={bd:.3f} tight={u_l / max(bd, 1e-8):.2f}')
        print(f'[epoch {epoch + 1:3d}/{n_epochs}] train={correct / seen:.4f} test={acc:.4f} '
              f'best={best:.4f}{ens_str} peak_mem={mem:.2f}GB{cert_str}', flush=True)
        if ckpt_path:
            torch.save({'model': model.state_dict(),
                        'opts': [o.state_dict() for o in model.opts],
                        'epoch': epoch + 1, 'best': best}, ckpt_path)
    print('=' * 64)
    print(f'FINAL best top-1: {best:.4f}  (mode={mode}, r101in, IMAGENET-1K, '
          f'K={model.K})  best_ens={best_ens:.4f}')
    print('=' * 64)


if __name__ == '__main__':
    if os.environ.get('VERIFY'):
        os.environ['ARCH'] = 'r50'
        os.environ.setdefault('EPS_FLOOR', '1e-5')
        split = os.environ.pop('SPLIT', '3,6,11,13')
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, 1000, (2,), device=DEVICE)
        for mode in ('bp', 'solo', 'dsn'):
            net = build_model(1000, 0.1, 90, mode, 1e-4, split).to(DEVICE)
            net.train()
            out, loss, cert = net.forward_with_update(x, y)
            n = sum(p.numel() for p in net.parameters()) / 1e6
            print(f'[smoke r101in {mode:4s} K={net.K} split={[len(g) for g in net.groups]}] '
                  f'loss={loss:.3f} params={n:.1f}M out={tuple(out.shape)} '
                  f'cert={"y" if cert else "n"} ok', flush=True)
    else:
        main()
