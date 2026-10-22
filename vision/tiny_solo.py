"""SOLO / BP on Tiny-ImageNet-200 (64x64, 200 classes).

The model and all its options come from solo_cifar.MDHeadResNet32; this script adds the Tiny-ImageNet
data pipeline (train via ImageFolder, val via val_annotations.txt) and the training loop.

Data: DATA_ROOT (default ./data)/tiny-imagenet-200/{train,val}
Usage: ARCH=r50 MODE=solo GS_MODE=jacobi python tiny_solo.py
       VERIFY=1 python tiny_solo.py   (smoke test)
Env: as solo_cifar.py; BATCH defaults to 64.
"""

import os
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from solo_cifar import MDHeadResNet32, evaluate, DEVICE

TINY_MEAN = (0.4802, 0.4481, 0.3975)
TINY_STD = (0.2302, 0.2265, 0.2262)


class TinyValSet(Dataset):
    """val/images + val_annotations.txt -> (img, label), using the train set's class_to_idx."""
    def __init__(self, root, class_to_idx, transform):
        self.dir = os.path.join(root, 'val', 'images')
        self.transform = transform
        self.samples = []
        ann = os.path.join(root, 'val', 'val_annotations.txt')
        with open(ann) as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) >= 2 and p[1] in class_to_idx:
                    self.samples.append((p[0], class_to_idx[p[1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fn, y = self.samples[i]
        img = Image.open(os.path.join(self.dir, fn)).convert('RGB')
        return self.transform(img), y


def train_tiny(n_epochs=50, batch_size=64, lr=0.1, K=2, gs_mode='gs', mode='solo',
               sched='s3', lam_floor=0.1, a_tgt=0.15, alpha_max=0.95, mu=1.0,
               ratchet=True, delta_t=0.2, aux_ce=0.1, autoaug=True,
               data_root='./data'):
    device = DEVICE
    root = os.path.join(data_root, 'tiny-imagenet-200')
    ls = 0.1 if mode in ('sll', 'bp') else 0.0
    print(f'device={device} | {mode.upper()} {os.environ.get("ARCH", "r32").upper()} '
          f'TINY-IMAGENET-200 K={K}(+BP head) gs={gs_mode} sched={sched} ls={ls} '
          f'bs={batch_size} autoaug={autoaug}', flush=True)

    extra = []
    if autoaug:
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            extra = [AutoAugment(AutoAugmentPolicy.IMAGENET)]
            print('  [aug] AutoAugment(IMAGENET) ON')
        except Exception as e:
            print(f'  [aug] AutoAugment unavailable ({e})')
    train_tf = transforms.Compose([
        transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(),
        *extra, transforms.ToTensor(), transforms.Normalize(TINY_MEAN, TINY_STD)])
    test_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(TINY_MEAN, TINY_STD)])

    train_set = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_tf)
    val_set = TinyValSet(root, train_set.class_to_idx, test_tf)
    print(f'  train={len(train_set)} val={len(val_set)} classes={len(train_set.classes)}')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = MDHeadResNet32(num_classes=200, lr=lr, K=K, gs_mode=gs_mode, mode=mode,
                           sched=sched, lam_floor=lam_floor, a_tgt=a_tgt,
                           alpha_max=alpha_max, mu=mu, ratchet=ratchet,
                           delta_t=delta_t, aux_ce=aux_ce, label_smoothing=ls,
                           t_max=int(os.environ.get('T_MAX', str(n_epochs)))).to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    best, best_ens, start_ep = 0.0, 0.0, 0
    ckpt_path = os.environ.get('CKPT', '')
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        for o, s in zip(model.opts, ck['opts']):
            o.load_state_dict(s)
        start_ep, best = ck['epoch'], ck['best']
        for _ in range(start_ep):
            model.scheduler.step()
        model.update_lr()
        print(f'  [ckpt] resumed @ epoch {start_ep} best={best:.4f}')

    for epoch in range(start_ep, n_epochs):
        model.train()
        correct, seen = 0, 0
        c_acc = dict(u0=0.0, u_last=0.0, bound=0.0, n=0)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{n_epochs}')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            out, loss, cert = model.forward_with_update(data, target)
            correct += out.argmax(1).eq(target).sum().item(); seen += target.size(0)
            if cert:
                for kk in ('u0', 'u_last', 'bound'):
                    c_acc[kk] += cert[kk]
                c_acc['n'] += 1
            pbar.set_postfix({'loss': f'{loss:.3f}', 'acc': f'{correct / seen:.4f}'})
        model.scheduler.step(); model.update_lr()
        acc, acc_ens = evaluate(model, val_loader, device)
        best = max(best, acc)
        ens_str = ''
        if acc_ens is not None:
            best_ens = max(best_ens, acc_ens)
            ens_str = f' ens={acc_ens:.4f} best_ens={best_ens:.4f}'
        mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
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
    print(f'FINAL best test acc: {best:.4f}  (mode={mode}, arch={os.environ.get("ARCH", "r32")}, '
          f'TINY, K={K})  best_ens={best_ens:.4f}')
    print('=' * 64)
    return model


if __name__ == '__main__':
    if os.environ.get('VERIFY'):
        x = torch.randn(4, 3, 64, 64, device=DEVICE)
        y = torch.randint(0, 200, (4,), device=DEVICE)
        net = MDHeadResNet32(num_classes=200, K=2,
                             mode=os.environ.get('MODE', 'solo'),
                             gs_mode=os.environ.get('GS_MODE', 'jacobi'), t_max=50).to(DEVICE)
        net.train()
        out, loss, cert = net.forward_with_update(x, y)
        print(f'[smoke tiny 64x64] ok loss={loss:.3f} out={tuple(out.shape)} '
              f'params={sum(p.numel() for p in net.parameters()) / 1e6:.1f}M')
    else:
        train_tiny(
            n_epochs=int(os.environ.get('EPOCHS', '50')),
            batch_size=int(os.environ.get('BATCH', '64')),
            lr=float(os.environ.get('LR', '0.1')),
            K=int(os.environ.get('K_MODULES', '2')),
            gs_mode=os.environ.get('GS_MODE', 'gs'),
            mode=os.environ.get('MODE', 'solo'),
            sched=os.environ.get('SCHED', 's3'),
            lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
            a_tgt=float(os.environ.get('A_TGT', '0.15')),
            alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
            mu=float(os.environ.get('MU', '1.0')),
            ratchet=os.environ.get('RATCHET', '1') == '1',
            delta_t=float(os.environ.get('DELTA_T', '0.2')),
            aux_ce=float(os.environ.get('AUX_CE', '0.1')),
            autoaug=os.environ.get('AUTOAUG', '1') == '1',
            data_root=os.environ.get('DATA_ROOT', './data'),
        )
