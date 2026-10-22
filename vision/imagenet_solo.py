"""SOLO / BP / DSN on ImageNet-1k with a standard ResNet-50.

Reuses solo_cifar.MDHeadResNet32 (all modes and options): the model is built with ARCH=r50 and its
CIFAR stem is replaced by the ImageNet stem (7x7/s2 conv + maxpool). Training uses 224px
RandomResizedCrop, center-crop eval, TF32 + channels_last, linear warmup + cosine, lr = LR * BATCH / 256.

Data: DATA_ROOT (default ./data/imagenet) must contain train/<wnid>/ and val/<wnid>/ (ImageFolder layout).

Usage:
  BP:   ARCH=r50in MODE=bp   EPOCHS=90 BATCH=256 python imagenet_solo.py
  DSN:  ARCH=r50in MODE=dsn  HEAD_DEPTH=3 HEAD_WIDEN=2 python imagenet_solo.py
  SOLO: ARCH=r50in MODE=solo GS_MODE=jacobi SCHED=s3 STRIDED_HEAD=1 \\
        HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 python imagenet_solo.py
  Smoke test: VERIFY=1 python imagenet_solo.py

Env (in addition to solo_cifar's): TRAIN_RES (224), VAL_RES (224), WARMUP (5 epochs), WORKERS (8),
AUTOAUG (1), WD (1e-4), CKPT (resume/save path), EPS_FLOOR (1e-5; with C=1000 the feasibility
condition (C-1)*eps < a_tgt - lambda rules out 1e-3 and 1e-4).
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from solo_cifar import MDHeadResNet32, evaluate, DEVICE

IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


def to_imagenet_r50(model, lr, wd):
    """Replace the CIFAR stem of an ARCH=r50 model with the ImageNet stem (7x7/s2 conv + BN + ReLU +
    3x3/s2 maxpool) and rebuild module 0's parameter list and optimizer, which reference the old stem."""
    stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                         nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                         nn.MaxPool2d(3, 2, 1))
    for m in stem.modules():                      # same rule as solo_cifar._init_weights
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    model.stem = stem
    params0 = list(model.stem.parameters())
    for bi in model.groups[0]:
        params0 += list(model.blocks[bi].parameters())
    params0 += list(model.heads[0].parameters()) if model.K > 1 else list(model.fc.parameters())
    model.module_params[0] = params0
    model.opts[0] = SGD(params0, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    return model


def build_loaders(data_root, batch_size, train_res=224, val_res=224, workers=8):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    for d in (train_dir, val_dir):
        assert os.path.isdir(d), f'not found: {d} - ImageFolder layout required (train/<wnid>/*.JPEG)'
    extra = []
    if os.environ.get('AUTOAUG', '1') == '1':
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        extra = [AutoAugment(AutoAugmentPolicy.IMAGENET)]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(train_res), transforms.RandomHorizontalFlip(),
        *extra, transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
    val_tf = transforms.Compose([
        transforms.Resize(int(val_res * 256 / 224)), transforms.CenterCrop(val_res),
        transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
    tr = datasets.ImageFolder(train_dir, transform=train_tf)
    va = datasets.ImageFolder(val_dir, transform=val_tf)
    assert len(tr.classes) == len(va.classes), 'train/val class directory counts differ'
    print(f'  data: train={len(tr)} val={len(va)} classes={len(tr.classes)}', flush=True)
    return (DataLoader(tr, batch_size, shuffle=True, num_workers=workers, drop_last=True,
                       pin_memory=True, persistent_workers=True),
            DataLoader(va, batch_size, shuffle=False, num_workers=max(workers // 2, 2),
                       pin_memory=True, persistent_workers=True), len(tr.classes))


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
    base_lr = float(os.environ.get('LR', '0.1')) * batch_size / 256   # linear scaling
    mode = os.environ.get('MODE', 'solo')
    os.environ['ARCH'] = 'r50'                            # build as CIFAR-r50, then swap in the ImageNet stem
    os.environ.setdefault('EPS_FLOOR', '1e-5')            # C=1000 feasibility (see module docstring)
    wd = float(os.environ.setdefault('WD', '1e-4'))

    train_loader, val_loader, n_cls = build_loaders(
        data_root, batch_size, int(os.environ.get('TRAIN_RES', '224')),
        int(os.environ.get('VAL_RES', '224')), int(os.environ.get('WORKERS', '8')))

    model = MDHeadResNet32(
        num_classes=n_cls, lr=base_lr, K=int(os.environ.get('K_MODULES', '2')),
        gs_mode=os.environ.get('GS_MODE', 'jacobi'), mode=mode,
        sched=os.environ.get('SCHED', 's3'),
        lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
        a_tgt=float(os.environ.get('A_TGT', '0.15')),
        alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
        mu=float(os.environ.get('MU', '1.0')),
        ratchet=os.environ.get('RATCHET', '1') == '1',
        delta_t=float(os.environ.get('DELTA_T', '0.2')),
        aux_ce=float(os.environ.get('AUX_CE', '0.1')),
        t_max=n_epochs).to(DEVICE)
    model = to_imagenet_r50(model, base_lr, wd).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'device={DEVICE} | {mode.upper()} {os.environ["ARCH"].upper()} IMAGENET-1K '
          f'K={model.K} bs={batch_size} lr={base_lr:.3f} warmup={warmup} '
          f'eps={os.environ["EPS_FLOOR"]} params={n_par:.1f}M TF32+channels_last', flush=True)

    best, best_ens, start_ep = 0.0, 0.0, 0
    ckpt_path = os.environ.get('CKPT', '')
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck['model'])
        for o, s in zip(model.opts, ck.get('opts', [])):
            o.load_state_dict(s)
        start_ep, best = ck['epoch'], ck['best']
        for _ in range(start_ep):
            model.scheduler.step()
        print(f'  [ckpt] resumed @ epoch {start_ep} best={best:.4f}', flush=True)

    def set_lr(ep):
        # per-epoch linear warmup -> cosine; overrides solo_cifar's internal scheduler
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
    print(f'FINAL best top-1: {best:.4f}  (mode={mode}, r50in, IMAGENET-1K, '
          f'K={model.K})  best_ens={best_ens:.4f}')
    print('=' * 64)


if __name__ == '__main__':
    if os.environ.get('VERIFY'):
        os.environ['ARCH'] = 'r50'
        os.environ.setdefault('EPS_FLOOR', '1e-5')
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, 1000, (2,), device=DEVICE)
        for mode in ('bp', 'solo', 'dsn'):
            net = MDHeadResNet32(num_classes=1000, K=2, mode=mode,
                                 gs_mode='jacobi', t_max=90).to(DEVICE)
            net = to_imagenet_r50(net, 0.1, 1e-4).to(DEVICE)
            net.train()
            out, loss, cert = net.forward_with_update(x, y)
            n = sum(p.numel() for p in net.parameters()) / 1e6
            print(f'[smoke r50in {mode:4s}] loss={loss:.3f} params={n:.1f}M '
                  f'out={tuple(out.shape)} cert={"y" if cert else "n"} ok', flush=True)
    else:
        main()
