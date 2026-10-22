"""DSN control on ImageNet-1k with DDP: ResNet-50 (ImageNet stem) plus the same auxiliary heads as
SOLO, trained end to end with full backprop.

Loss (same as the dsn branch of solo_cifar.forward_with_update, via forward_all with no detach):
  CE(final) + DSN_AUX * sum_k CE(head_k)
Recipe matches imagenet_solo.py: SGD momentum 0.9 nesterov, WD=1e-4, lr = LR * BATCH / 256, linear
warmup 5 epochs + cosine, AutoAugment(IMAGENET), label smoothing 0, DSN_AUX=0.3, TF32 + channels_last.
A single global SGD is used; this is per-parameter equivalent to per-module SGD.

Requires solo_cifar.py and imagenet_solo.py in the same directory; data layout as imagenet_solo.py.

Usage (2 GPUs; BATCH is the global batch, split across ranks):
  HEAD_DEPTH=3 HEAD_WIDEN=2 EPOCHS=90 BATCH=256 CKPT=ckpt/in_dsn.pt \\
    torchrun --nproc_per_node=2 imagenet_dsn_ddp.py
  Smoke test (no GPU, no data):
  BACKEND=gloo FAKE_DATA=1 STEPS=2 BATCH=8 torchrun --nproc_per_node=2 imagenet_dsn_ddp.py

Eval runs on rank 0 only (final head and geometric ensemble); other ranks wait at a barrier.
BN statistics are per GPU; SYNC_BN=1 switches to synchronized BN.
Other env: BACKEND (nccl), FAKE_DATA (0), STEPS (2), NUM_CLASSES (1000), TRAIN_RES (224), WORKERS (8),
AUTOAUG (1), K_MODULES (2), WARMUP (5), LR (0.1), SEED (42), DATA_ROOT (./data/imagenet).
"""

import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

os.environ.setdefault('ARCH', 'r50')
from solo_cifar import MDHeadResNet32                       # noqa: E402
from imagenet_solo import to_imagenet_r50, IN_MEAN, IN_STD  # noqa: E402


class DSNLoss(nn.Module):
    """DDP-wrappable forward returning the joint DSN loss and the detached final logits."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y):
        logits, outs = self.m.forward_all(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.m.ls) \
            + self.m.dsn_aux * sum(F.cross_entropy(z, y, label_smoothing=self.m.ls)
                                   for z in outs)
        return loss, logits.detach()


@torch.no_grad()
def evaluate_full(model, loader, dev):
    """Returns (final-head acc, geometric-ensemble acc), same rule as solo_cifar.evaluate."""
    model.eval()
    correct, correct_ens, n = 0, 0, 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(dev, non_blocking=True)
        fin, outs = model.forward_all(x)
        correct += fin.argmax(1).eq(y).sum().item()
        lp = F.log_softmax(fin, -1)
        for z in outs:
            lp = lp + F.log_softmax(z, -1)
        correct_ens += lp.argmax(1).eq(y).sum().item()
        n += y.size(0)
    model.train()
    return correct / n, correct_ens / n


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    backend = os.environ.get('BACKEND', 'nccl')
    fake = os.environ.get('FAKE_DATA', '0') == '1'
    use_cuda = backend == 'nccl' and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda') if use_cuda else torch.device('cpu')
    dist.init_process_group(backend)

    seed = int(os.environ.get('SEED', '42'))
    torch.manual_seed(seed)                                  # same init on all ranks
    n_epochs = int(os.environ.get('EPOCHS', '90'))
    gbs = int(os.environ.get('BATCH', '256'))                # global batch
    bs = gbs // world
    warmup = int(os.environ.get('WARMUP', '5'))
    base_lr = float(os.environ.get('LR', '0.1')) * gbs / 256
    wd = float(os.environ.setdefault('WD', '1e-4'))
    os.environ.setdefault('EPS_FLOOR', '1e-5')               # unused by dsn
    res = int(os.environ.get('TRAIN_RES', '224'))
    n_cls = int(os.environ.get('NUM_CLASSES', '1000'))

    model = MDHeadResNet32(num_classes=n_cls, lr=base_lr, K=int(os.environ.get('K_MODULES', '2')),
                           gs_mode='jacobi', mode='dsn', t_max=n_epochs)
    model = to_imagenet_r50(model, base_lr, wd)
    model.opts = []                                          # drop per-module optimizers, use a global SGD
    model = model.to(dev)
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)
    if os.environ.get('SYNC_BN', '0') == '1':
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    wrapper = DSNLoss(model).to(dev)
    ddp = DDP(wrapper, device_ids=[dev.index] if use_cuda else None)
    opt = SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=wd)

    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'DSN-DDP | R50-IN K={model.K} world={world} gbs={gbs} (bs/gpu={bs}) '
              f'lr={base_lr:.3f} wd={wd} warmup={warmup} ls={model.ls} '
              f'aux={model.dsn_aux} params={n_par:.1f}M dev={dev}', flush=True)

    # ---- data ----
    if fake:
        steps = int(os.environ.get('STEPS', '2'))
        train_loader = [(torch.randn(bs, 3, res, res), torch.randint(0, n_cls, (bs,)))
                        for _ in range(steps)]
        sampler, val_loader = None, train_loader[:1]
    else:
        root = os.environ.get('DATA_ROOT', './data/imagenet')
        extra = []
        if os.environ.get('AUTOAUG', '1') == '1':
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            extra = [AutoAugment(AutoAugmentPolicy.IMAGENET)]
        tr_tf = transforms.Compose([transforms.RandomResizedCrop(res),
                                    transforms.RandomHorizontalFlip(), *extra,
                                    transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
        va_tf = transforms.Compose([transforms.Resize(int(res * 256 / 224)),
                                    transforms.CenterCrop(res),
                                    transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
        tr = datasets.ImageFolder(os.path.join(root, 'train'), transform=tr_tf)
        sampler = DistributedSampler(tr, shuffle=True, drop_last=True)
        wk = int(os.environ.get('WORKERS', '8'))
        train_loader = DataLoader(tr, bs, sampler=sampler, num_workers=wk,
                                  pin_memory=use_cuda, persistent_workers=wk > 0,
                                  drop_last=True)
        val_loader = None
        if rank == 0:
            va = datasets.ImageFolder(os.path.join(root, 'val'), transform=va_tf)
            val_loader = DataLoader(va, gbs, shuffle=False, num_workers=max(wk // 2, 2),
                                    pin_memory=use_cuda, persistent_workers=True)
            print(f'  data: train={len(tr)} val={len(va)}', flush=True)

    # ---- resume from checkpoint ----
    ck_path = os.environ.get('CKPT', '')
    start_ep, best, best_ens = 0, 0.0, 0.0
    if ck_path and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_ep, best, best_ens = ck['epoch'], ck['best'], ck.get('best_ens', 0.0)
        if rank == 0:
            print(f'  [ckpt] resumed @ {start_ep} best={best:.4f}', flush=True)

    for ep in range(start_ep, n_epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        lr = base_lr * (ep + 1) / warmup if ep < warmup else \
            base_lr * 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(n_epochs - warmup, 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        ddp.train()
        correct, seen, loss_sum, nb = 0, 0, 0.0, 0
        for x, y in train_loader:
            x = x.to(dev, non_blocking=True)
            if use_cuda:
                x = x.to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, logits = ddp(x, y)
            loss.backward()
            opt.step()
            correct += logits.argmax(1).eq(y).sum().item()
            seen += y.size(0)
            loss_sum += loss.item()
            nb += 1
        dist.barrier()
        if rank == 0 and val_loader is not None:
            acc, ens = evaluate_full(model, val_loader, dev)
            best, best_ens = max(best, acc), max(best_ens, ens)
            mem = (f' peak_mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GB'
                   if use_cuda else '')
            print(f'[epoch {ep + 1:3d}/{n_epochs}] lr={lr:.4f} '
                  f'train={correct / max(seen, 1):.4f} loss={loss_sum / max(nb, 1):.3f} '
                  f'test={acc:.4f} best={best:.4f} ens={ens:.4f} '
                  f'best_ens={best_ens:.4f}{mem}', flush=True)
            if ck_path:
                torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                            'epoch': ep + 1, 'best': best, 'best_ens': best_ens}, ck_path)
        dist.barrier()
    if rank == 0:
        print('=' * 60)
        print(f'FINAL best top-1: {best:.4f}  (mode=dsn-ddp, r50in, K={model.K}) '
              f'best_ens={best_ens:.4f}')
        print('=' * 60)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
