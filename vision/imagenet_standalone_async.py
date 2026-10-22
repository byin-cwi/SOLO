"""ImageNet-1k ResNet-50 SOLO with per-module independent updates.

Identical to imagenet_standalone.py except in MODE=solo: as soon as module k's backward finishes, its
own gradients are all-reduced (DDP only), opt_k.step() runs and its gradients are zeroed, so there is no
global optimizer barrier. This is numerically equivalent to the joint update, because module gradients
are independent and downstream modules consume pre-update activations. bp/dsn keep the global sync.
Options, data layout, BENCH/SYNTH, DDP and checkpoint resume are as in imagenet_standalone.py.

Usage:
  K=2, split 8|8:
    MODE=solo SCHED=s3 STRIDED_HEAD=1 HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 SPLIT=8,8 \\
      EPOCHS=90 BATCH=256 CKPT=ckpt/in_solo.pt python -u imagenet_standalone_async.py
  K=4, split balanced by activation bytes:
    MODE=solo SCHED=s3 STRIDED_HEAD=1 HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 SPLIT=2,3,6,5 \\
      EPOCHS=90 BATCH=256 CKPT=ckpt/in_solo_k4.pt python -u imagenet_standalone_async.py
  Smoke test: VERIFY=1 python imagenet_standalone_async.py
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import torch.distributed as dist

def dist_on():
    return dist.is_available() and dist.is_initialized()


def sync_grads(model):
    """All-reduce (mean) gradients after the full backward and before the optimizer step."""
    if not dist_on():
        return
    world = dist.get_world_size()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def sync_module_grads(model, k):
    """All-reduce (mean) only module k's gradients."""
    if not dist_on():
        return
    world = dist.get_world_size()
    for p in model.module_params[k]:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def sync_protos(model):
    if not dist_on() or not hasattr(model, 'protos'):
        return
    dist.all_reduce(model.protos, op=dist.ReduceOp.SUM)
    model.protos.div_(dist.get_world_size())


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS_FLOOR = float(os.environ.get('EPS_FLOOR', '1e-5'))    # small enough for C=1000 feasibility
CERT_C = 2 * math.sqrt(2) / math.sqrt(EPS_FLOOR)


def floor_simplex(q, eps=EPS_FLOOR):
    q = q.clamp_min(eps)
    return q / q.sum(dim=-1, keepdim=True)


def hellinger2(q, m):
    return (1.0 - (q * m).clamp_min(0).sqrt().sum(dim=-1)).clamp_min(0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        out = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.downsample = None
        if stride != 1 or inplanes != out:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, out, 1, stride, bias=False),
                                            nn.BatchNorm2d(out))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + identity)


class ReadoutHead(nn.Module):
    def __init__(self, inplanes, num_classes, hidden=128, widen=1, depth=1, strided=False):
        super().__init__()
        mid = int(min(inplanes // 2, 256) * widen)
        stride = 2 if inplanes <= 512 else 1
        layers = [nn.Conv2d(inplanes, mid, 3, stride, 1, bias=False),
                  nn.BatchNorm2d(mid), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(mid, mid, 3, 2 if strided else 1, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU()]
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                   nn.Linear(mid, int(hidden * widen)), nn.ReLU(inplace=True),
                   nn.Linear(int(hidden * widen), num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class R50ImageNet(nn.Module):
    """ImageNet R50 with SOLO training; attribute names are checkpoint-compatible with solo_cifar models."""

    def __init__(self, num_classes=1000, lr=0.1, K=2, mode='solo', sched='s3',
                 lam_floor=0.1, a_tgt=0.15, alpha_max=0.95, mu=1.0, ratchet=True,
                 delta_t=0.2, aux_ce=0.1, label_smoothing=None):
        super().__init__()
        assert mode in ('bp', 'solo', 'md_ema', 'dsn')
        self.mode, self.sched = mode, sched
        self.lam_floor, self.a_tgt, self.alpha_max, self.mu = lam_floor, a_tgt, alpha_max, mu
        self.ratchet, self.delta_t, self.aux_ce = ratchet, delta_t, aux_ce
        self.num_classes = num_classes
        self.ls = (0.1 if mode in ('bp', 'dsn') else 0.0) \
            if label_smoothing is None else label_smoothing
        self.ema_m = float(os.environ.get('EMA_M', '0.99'))
        self.dsn_aux = float(os.environ.get('DSN_AUX', '0.3'))

        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        self.blocks = nn.ModuleList()
        block_ch, in_ch = [], 64
        for planes, n, first_stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
            for b in range(n):
                self.blocks.append(Bottleneck(in_ch, planes, first_stride if b == 0 else 1))
                in_ch = planes * Bottleneck.expansion
                block_ch.append(in_ch)
        self.fc = nn.Linear(2048, num_classes)
        self.n_blocks = len(self.blocks)

        K = max(2, min(K, self.n_blocks))
        self.groups = [list(g) for g in np.array_split(np.arange(self.n_blocks), K)]
        split_env = os.environ.get('SPLIT', '')
        if split_env:
            sizes = [int(s) for s in split_env.split(',')]
            assert sum(sizes) == self.n_blocks, f'SPLIT {sizes} != {self.n_blocks} blocks'
            idx, self.groups = 0, []
            for s in sizes:
                self.groups.append(list(range(idx, idx + s)))
                idx += s
            K = len(sizes)
        self.K, self.R = K, K - 1
        self._init_weights()

        head_widen = float(os.environ.get('HEAD_WIDEN', '1'))
        head_depth = int(os.environ.get('HEAD_DEPTH', '1'))
        strided = os.environ.get('STRIDED_HEAD', '0') == '1'
        self.heads = nn.ModuleList(
            [ReadoutHead(block_ch[self.groups[k][-1]], num_classes, widen=head_widen,
                         depth=head_depth, strided=strided) for k in range(K - 1)])
        self.ach_ema = [math.log(num_classes)] * (K - 1)
        self.s4_kappa = float(os.environ.get('S4_KAPPA', '0.7'))

        wd = float(os.environ.get('WD', '1e-4'))
        self.module_params = []
        for k in range(K):
            params = list(self.stem.parameters()) if k == 0 else []
            for bi in self.groups[k]:
                params += list(self.blocks[bi].parameters())
            params += list(self.heads[k].parameters()) if k < K - 1 \
                else list(self.fc.parameters())
            self.module_params.append(params)
        self.opts = [SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
                     for p in self.module_params]

        in_proj = 2048
        self.input_proj = torch.randn(in_proj, num_classes) / np.sqrt(in_proj)
        self.register_buffer('protos', torch.full((K - 1, num_classes, num_classes),
                                                  1.0 / num_classes))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight); nn.init.constant_(m.bias, 0)

    def input_readout(self, x):
        b = x.size(0)
        flat = F.adaptive_avg_pool1d(x.reshape(b, 1, -1), self.input_proj.shape[0]).squeeze(1)
        return F.softmax(flat @ self.input_proj.to(x.device), dim=1)

    def pool_features(self, h):
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def run_group(self, k, x_in):
        h = x_in
        if k == 0:
            h = self.stem(h)
        for bi in self.groups[k]:
            h = self.blocks[bi](h)
        return h

    @torch.no_grad()
    def milestone(self, prev_q, q_detached, y, k, alpha_s2):
        B = prev_q.size(0)
        p = floor_simplex(prev_q)
        p_y = p.gather(1, y[:, None]).squeeze(1).clamp(EPS_FLOOR, 1 - 1e-6)
        a = -p_y.log()
        u = (a - self.lam_floor).clamp_min(0)
        below = a <= self.lam_floor
        gap = max(self.a_tgt - self.lam_floor, 1e-3)
        if self.sched == 's1':
            alpha = torch.full_like(a, 1 - (gap / max(math.log(self.num_classes)
                                                      - self.lam_floor, gap)) ** (1.0 / self.R))
        elif self.sched == 's2':
            alpha = alpha_s2
        else:                                             # s3/s4: closed loop over the remaining depth
            rem = self.R - k
            alpha = 1 - (gap / u.clamp_min(gap)) ** (1.0 / rem)
        alpha = alpha.clamp(0.0, self.alpha_max)
        a_tar = torch.where(below, a, self.lam_floor + (1 - alpha) * u)
        if self.sched == 's4':                            # reachability clamp
            a_tar = torch.maximum(a_tar, torch.full_like(a_tar, self.s4_kappa * self.ach_ema[k]))
        m_y = (-a_tar).exp()
        t = ((m_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.ratchet:                                  # windowed ratchet
            q_y = floor_simplex(q_detached).gather(1, y[:, None]).squeeze(1) \
                .clamp(EPS_FLOOR, 1 - 1e-6)
            t_stu = ((q_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
            t = torch.minimum(torch.maximum(t, t_stu), t + self.delta_t)
        m = (1 - t)[:, None] * p
        m[torch.arange(B, device=p.device), y] += t
        m = torch.where(below[:, None], floor_simplex(q_detached), m)   # below-floor: m := sg(q)
        a_tar_real = -m.gather(1, y[:, None]).squeeze(1).clamp_min(EPS_FLOOR).log()
        return m, a_tar_real, alpha, below

    def forward_with_update(self, img, y):
        if self.mode == 'bp':
            for opt in self.opts:
                opt.zero_grad()
            logits = self.forward(img)
            loss = F.cross_entropy(logits, y, label_smoothing=self.ls)
            loss.backward()
            sync_grads(self)
            for opt in self.opts:
                opt.step()
            return logits, loss.item(), None
        if self.mode == 'dsn':                            # BP + the same aux heads, full backprop
            for opt in self.opts:
                opt.zero_grad()
            logits, outs = self.forward_all(img)
            loss = F.cross_entropy(logits, y, label_smoothing=self.ls) \
                + self.dsn_aux * sum(F.cross_entropy(z, y, label_smoothing=self.ls)
                                     for z in outs)
            loss.backward()
            sync_grads(self)
            for opt in self.opts:
                opt.step()
            return logits, loss.item(), None

        for opt in self.opts:
            opt.zero_grad()
        prev_q = self.input_readout(img.detach())
        x = img
        q0f = floor_simplex(prev_q)                       # certificate quantities
        u0 = ((-q0f.gather(1, y[:, None]).squeeze(1).log()) - self.lam_floor).clamp_min(0)
        gap = max(self.a_tgt - self.lam_floor, 1e-3)
        alpha_s2 = (1 - (gap / u0.clamp_min(gap)) ** (1.0 / self.R)).clamp(0.0, self.alpha_max)
        alphas, errs, u_last = [], [], None

        for k in range(self.K - 1):                       # local modules (Jacobi)
            x_in = x.detach()
            x = self.run_group(k, x_in)
            logits = self.heads[k](x)
            q = F.softmax(logits, dim=-1)
            ce_per = F.cross_entropy(logits, y, reduction='none')
            if k < len(self.ach_ema):
                self.ach_ema[k] = 0.95 * self.ach_ema[k] + 0.05 * ce_per.mean().item()
            parent_for_m = self.protos[k][y]              # lagged self-prototype anchor
            with torch.no_grad():                         # prototype EMA: solo tracks its own output
                src = q.detach() if self.mode == 'solo' else prev_q
                qd = floor_simplex(src)
                oh = F.one_hot(y, self.num_classes).float()
                cnt = oh.sum(0)
                present = cnt > 0
                means = (oh.t() @ qd)[present] / cnt[present].unsqueeze(1)
                self.protos[k][present] = self.ema_m * self.protos[k][present] \
                    + (1 - self.ema_m) * means
            m, a_tar_real, alpha_k, below = self.milestone(parent_for_m, q.detach(),
                                                           y, k, alpha_s2)
            hinge = (ce_per - a_tar_real).clamp_min(0)
            loss = hinge.mean() + self.mu * hellinger2(floor_simplex(q), m).mean()
            if self.aux_ce > 0:                           # auxiliary CE
                loss = loss + self.aux_ce * F.cross_entropy(logits, y, label_smoothing=self.ls)
            loss.backward()
            sync_module_grads(self, k)
            self.opts[k].step()                           # per-module independent update
            self.opts[k].zero_grad()
            prev_q = q.detach()
            prev_z = logits.detach()
            x = x.detach()
            with torch.no_grad():                         # certificate on the readout passed downstream
                ce_c = F.cross_entropy(prev_z, y, reduction='none')
                qf_c = floor_simplex(F.softmax(prev_z, dim=-1))
                hinge_c = (ce_c - a_tar_real).clamp_min(0)
                h_c = hellinger2(qf_c, m).clamp_min(0).sqrt()
                errs.append(torch.minimum(hinge_c, CERT_C * h_c))
                alphas.append(torch.where(below, torch.zeros_like(alpha_k), alpha_k))
                u_last = (ce_c - self.lam_floor).clamp_min(0)

        x_in = x.detach()                                 # last module: BP CE head
        x = self.run_group(self.K - 1, x_in)
        logits = self.fc(self.pool_features(x))
        loss_head = F.cross_entropy(logits, y, label_smoothing=self.ls)
        loss_head.backward()
        sync_module_grads(self, self.K - 1)
        sync_protos(self)
        self.opts[self.K - 1].step()                      # last module is also updated independently
        self.opts[self.K - 1].zero_grad()

        cert = None
        if alphas:
            with torch.no_grad():
                bound = u0.clone()
                for al, e in zip(alphas, errs):
                    bound = (1 - al) * bound + e
                cert = dict(u0=u0.mean().item(), u_last=u_last.mean().item(),
                            bound=bound.mean().item())
        return logits, loss_head.item(), cert

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(self.pool_features(h))

    def forward_all(self, x):
        h, outs = x, []
        for k in range(self.K - 1):
            h = self.run_group(k, h)
            outs.append(self.heads[k](h))
        h = self.run_group(self.K - 1, h)
        return self.fc(self.pool_features(h)), outs


def evaluate(model, loader, device):
    model.eval()
    correct, correct_ens = 0, 0
    use_ens = model.mode not in ('bp',)
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if use_ens:
                fin, outs = model.forward_all(data)
                correct += fin.argmax(1).eq(target).sum().item()
                lp = F.log_softmax(fin, -1)
                for z in outs:
                    lp = lp + F.log_softmax(z, -1)
                correct_ens += lp.argmax(1).eq(target).sum().item()
            else:
                correct += model(data).argmax(1).eq(target).sum().item()
    n = len(loader.dataset)
    return correct / n, (correct_ens / n if use_ens else None)


def build_loaders(data_root, batch_size, train_res, val_res, workers, world=1, rank=0):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torchvision import datasets, transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    extra = []
    if os.environ.get('AUTOAUG', '1') == '1':
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        extra = [AutoAugment(AutoAugmentPolicy.IMAGENET)]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(train_res), transforms.RandomHorizontalFlip(),
        *extra, transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([
        transforms.Resize(int(val_res * 256 / 224)), transforms.CenterCrop(val_res),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_tf)
    if rank == 0:
        print(f'  data: train={len(tr)} val={len(va)} classes={len(tr.classes)}', flush=True)
    sampler = DistributedSampler(tr, world, rank, shuffle=True, drop_last=True) \
        if world > 1 else None
    return (DataLoader(tr, batch_size, shuffle=sampler is None, sampler=sampler,
                       num_workers=workers, drop_last=True,
                       pin_memory=True, persistent_workers=workers > 0),
            DataLoader(va, batch_size, shuffle=False, num_workers=max(workers // 2, 2),
                       pin_memory=True, persistent_workers=True), len(tr.classes))


def main():
    global DEVICE
    rank, world = 0, 1
    if 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
            DEVICE = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    seed = int(os.environ.get('SEED', '42')) + rank * 0  # same init on all ranks
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    data_root = os.environ.get('DATA_ROOT', './data/imagenet')
    n_epochs = int(os.environ.get('EPOCHS', '90'))
    bs = int(os.environ.get('BATCH', '256'))
    warmup = int(os.environ.get('WARMUP', '5'))
    base_lr = float(os.environ.get('LR', '0.1')) * bs / 256
    mode = os.environ.get('MODE', 'solo')

    bench = int(os.environ.get('BENCH', '0'))
    synth = os.environ.get('SYNTH', '0') == '1'
    if bench and synth:                                   # synthetic benchmark: no dataset needed
        train_loader, val_loader = None, None
        n_cls = int(os.environ.get('NUM_CLASSES', '1000'))
    else:
        train_loader, val_loader, n_cls = build_loaders(
            data_root, bs, int(os.environ.get('TRAIN_RES', '224')),
            int(os.environ.get('VAL_RES', '224')), int(os.environ.get('WORKERS', '8')),
            world, rank)
    base_lr = float(os.environ.get('LR', '0.1')) * bs * world / 256   # global-batch scaling

    model = R50ImageNet(
        num_classes=n_cls, lr=base_lr, K=int(os.environ.get('K_MODULES', '2')),
        mode=mode, sched=os.environ.get('SCHED', 's3'),
        lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
        a_tgt=float(os.environ.get('A_TGT', '0.15')),
        alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
        mu=float(os.environ.get('MU', '1.0')),
        ratchet=os.environ.get('RATCHET', '1') == '1',
        delta_t=float(os.environ.get('DELTA_T', '0.2')),
        aux_ce=float(os.environ.get('AUX_CE', '0.1')),
        label_smoothing=float(os.environ['LS']) if 'LS' in os.environ else None,
    ).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f'device={DEVICE} | {mode.upper()} R50 IMAGENET-1K K={model.K} '
              f'split={[len(g) for g in model.groups]} bs={bs}x{world}(global {bs * world}) '
              f'lr={base_lr:.3f} warmup={warmup} eps={EPS_FLOOR} params={n_par:.1f}M '
              f'standalone-async(per-module update)', flush=True)

    best, best_ens, start_ep = 0.0, 0.0, 0
    ckpt_path = os.environ.get('CKPT', '')
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(ck['model'], strict=False)
        if missing or unexpected:
            print(f'  [ckpt] non-strict load: missing={missing} unexpected={unexpected}',
                  flush=True)
        for o, s in zip(model.opts, ck.get('opts', [])):
            o.load_state_dict(s)
        start_ep, best = ck['epoch'], ck['best']
        print(f'  [ckpt] resumed @ epoch {start_ep} best={best:.4f}', flush=True)

    if bench:                                             # 10 warmup steps + BENCH timed steps, then exit
        model.train()
        res = int(os.environ.get('TRAIN_RES', '224'))
        it = iter(train_loader) if not synth else None

        def next_batch():
            if synth:
                return (torch.randn(bs, 3, res, res, device=DEVICE)
                        .to(memory_format=torch.channels_last),
                        torch.randint(0, n_cls, (bs,), device=DEVICE))
            d, tg = next(it)
            return (d.to(DEVICE, non_blocking=True, memory_format=torch.channels_last),
                    tg.to(DEVICE, non_blocking=True))

        for _ in range(10):
            data, target = next_batch()
            model.forward_with_update(data, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist_on():
            dist.barrier()
        t0 = time.time()
        for _ in range(bench):
            data, target = next_batch()
            model.forward_with_update(data, target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist_on():
            dist.barrier()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        if rank == 0:
            print(f'BENCH mode={mode} world={world} synth={int(synth)} bs={bs}x{world}: {bench} steps '
                  f'in {dt:.1f}s -> {bench * bs * world / dt:.0f} img/s '
                  f'({dt / bench * 1000:.0f} ms/step, peak_mem {mem:.2f}GB/rank)', flush=True)
        return

    def set_lr(ep):
        lr = base_lr * (ep + 1) / warmup if ep < warmup else \
            base_lr * 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(n_epochs - warmup, 1)))
        for opt in model.opts:
            for g in opt.param_groups:
                g['lr'] = lr
        return lr

    for epoch in range(start_ep, n_epochs):
        lr_now = set_lr(epoch)
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        correct, seen = 0, 0
        c_acc = dict(u0=0.0, u_last=0.0, bound=0.0, n=0)
        pbar = tqdm(train_loader, desc=f'Ep {epoch + 1}/{n_epochs} lr={lr_now:.4f}',
                    disable=rank != 0)
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
        if dist_on():
            dist.barrier()
        if rank != 0:
            continue                                      # eval/logging/checkpointing on rank 0 only
        acc, acc_ens = evaluate(model, val_loader, DEVICE)
        best = max(best, acc)
        ens_str = ''
        if acc_ens is not None:
            best_ens = max(best_ens, acc_ens)
            ens_str = f' ens={acc_ens:.4f} best_ens={best_ens:.4f}'
        mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
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
    print(f'FINAL best top-1: {best:.4f}  (mode={mode}, r50in-standalone, '
          f'K={model.K})  best_ens={best_ens:.4f}')
    print('=' * 64)


if __name__ == '__main__':
    if os.environ.get('VERIFY'):
        x = torch.randn(2, 3, 224, 224, device=DEVICE)
        y = torch.randint(0, 1000, (2,), device=DEVICE)
        for mode, split in [('bp', '8,8'), ('solo', '8,8'), ('solo', '2,3,6,5'),
                            ('dsn', '8,8')]:
            os.environ['SPLIT'] = split
            net = R50ImageNet(num_classes=1000, K=2, mode=mode).to(DEVICE)
            net.train()
            out, loss, cert = net.forward_with_update(x, y)
            n = sum(p.numel() for p in net.parameters()) / 1e6
            print(f'[smoke standalone {mode:4s} split={split}] loss={loss:.3f} '
                  f'params={n:.1f}M out={tuple(out.shape)} '
                  f'cert={"y" if cert else "n"} ok', flush=True)
    else:
        main()
