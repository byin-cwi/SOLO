"""SOLO module-parallel pipeline on ImageNet-1k (self-contained; torch/torchvision only).

One process and one module per GPU. Between GPUs there is only a one-way stream of forward activations
(isend/irecv); no gradients are communicated and there is no global sync point. The model, milestone and
certificate code matches imagenet_standalone.py.

Topology (K = WORLD_SIZE = number of SPLIT groups):
  rank 0:   data -> stem + group 0 -> local milestone loss + step -> isend(h, y, u0, a2)
  rank k:   irecv -> group k -> local loss + step -> isend
  rank K-1: irecv -> group K-1 + fc -> CE step; reports accuracy, throughput and memory
Eval runs the same pipeline and accumulates each head's log-softmax; the last stage reports ens.

Notes: each rank logs the certificate residual of its own module (the chained bound is composed offline);
val uses drop_last (up to BATCH-1 of 50000 samples are skipped); downstream modules consume activations
produced by upstream parameters from the previous step, which Jacobi semantics tolerates. There is no BP
mode; use DDP for a multi-GPU BP baseline. Checkpoints are saved per rank as CKPT.rank<k>.

Usage:
  Speed test (no dataset; read rank K-1's "[epoch 1/1] ... img/s"):
    CUDA_VISIBLE_DEVICES=0,1 FAKE_DATA=1 STEPS=100 NET=r50 SPLIT=8,8 \\
      SCHED=s3 STRIDED_HEAD=1 HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 \\
      EPOCHS=1 BATCH=256 NUM_CLASSES=1000 \\
      torchrun --nproc_per_node=2 --master_port=29621 imagenet_pipeline_solo.py
    Single-GPU reference: imagenet_standalone.py with BENCH=100. K=4: SPLIT=2,3,6,5 --nproc_per_node=4.
  Full training (K=2, split 8|8):
    CUDA_VISIBLE_DEVICES=0,1 NET=r50 SPLIT=8,8 SCHED=s3 STRIDED_HEAD=1 \\
      HEAD_DEPTH=3 HEAD_WIDEN=2 EMA_M=0.9 EPOCHS=90 BATCH=256 CKPT=ckpt/in_pipe88 \\
      torchrun --nproc_per_node=2 --master_port=29622 imagenet_pipeline_solo.py
  CPU self-check:
    BACKEND=gloo FAKE_DATA=1 NET=r50 SPLIT=2,3,6,5 BATCH=2 STEPS=2 \\
      torchrun --nproc_per_node=4 imagenet_pipeline_solo.py

Env: NET (r50|r101), SPLIT (8,8 for r50, 3,6,11,13 for r101), BACKEND (nccl if available, else gloo),
FAKE_DATA (0), STEPS (100, steps per epoch with FAKE_DATA), NUM_CLASSES (1000), LS (0.0), and the
SOLO / optimization options of imagenet_standalone.py with the same defaults.
"""

import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

EPS_FLOOR = float(os.environ.get('EPS_FLOOR', '1e-5'))
CERT_C = 2 * math.sqrt(2) / math.sqrt(EPS_FLOOR)
R50_STAGES = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]      # 16 blocks
R101_STAGES = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]    # 33 blocks


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
    """ImageNet ResNet with SOLO training (same as imagenet_standalone.py)."""

    def __init__(self, num_classes=1000, K=2, sched='s3', stages=None,
                 lam_floor=0.1, a_tgt=0.15, alpha_max=0.95, mu=1.0, ratchet=True,
                 delta_t=0.2, aux_ce=0.1):
        super().__init__()
        self.mode, self.sched = 'solo', sched
        self.lam_floor, self.a_tgt, self.alpha_max, self.mu = lam_floor, a_tgt, alpha_max, mu
        self.ratchet, self.delta_t, self.aux_ce = ratchet, delta_t, aux_ce
        self.num_classes = num_classes
        self.ls = float(os.environ.get('LS', '0.0'))
        self.ema_m = float(os.environ.get('EMA_M', '0.99'))

        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool2d(3, 2, 1))
        self.blocks = nn.ModuleList()
        block_ch, in_ch = [], 64
        for planes, n, first_stride in (stages or R50_STAGES):
            for b in range(n):
                self.blocks.append(Bottleneck(in_ch, planes, first_stride if b == 0 else 1))
                in_ch = planes * Bottleneck.expansion
                block_ch.append(in_ch)
        self.fc = nn.Linear(in_ch, num_classes)
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
        else:
            rem = self.R - k
            alpha = 1 - (gap / u.clamp_min(gap)) ** (1.0 / rem)
        alpha = alpha.clamp(0.0, self.alpha_max)
        a_tar = torch.where(below, a, self.lam_floor + (1 - alpha) * u)
        if self.sched == 's4':
            a_tar = torch.maximum(a_tar, torch.full_like(a_tar, self.s4_kappa * self.ach_ema[k]))
        m_y = (-a_tar).exp()
        t = ((m_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.ratchet:
            q_y = floor_simplex(q_detached).gather(1, y[:, None]).squeeze(1) \
                .clamp(EPS_FLOOR, 1 - 1e-6)
            t_stu = ((q_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
            t = torch.minimum(torch.maximum(t, t_stu), t + self.delta_t)
        m = (1 - t)[:, None] * p
        m[torch.arange(B, device=p.device), y] += t
        m = torch.where(below[:, None], floor_simplex(q_detached), m)
        a_tar_real = -m.gather(1, y[:, None]).squeeze(1).clamp_min(EPS_FLOOR).log()
        return m, a_tar_real, alpha, below


def log(rank, msg):
    print(f'[rank {rank}] {msg}', flush=True)


def stage_modules(model, rank, K):
    mods = [model.stem] if rank == 0 else []
    mods += [model.blocks[bi] for bi in model.groups[rank]]
    mods += [model.heads[rank]] if rank < K - 1 else [model.fc]
    return mods


@torch.no_grad()
def interface_shapes(model, res):
    model.eval()
    h = torch.zeros(1, 3, res, res)
    shapes = []
    for k in range(model.K):
        h = model.run_group(k, h)
        shapes.append(tuple(h.shape[1:]))
    model.train()
    return shapes


def local_solo_step(model, k, x_in, y, alpha_s2, opt):
    """One local SOLO step of module k (solo branch of imagenet_standalone.py).
    Returns the detached activation passed downstream and the mean hinge."""
    x = model.run_group(k, x_in)
    logits = model.heads[k](x)
    q = F.softmax(logits, dim=-1)
    ce_per = F.cross_entropy(logits, y, reduction='none')
    if k < len(model.ach_ema):
        model.ach_ema[k] = 0.95 * model.ach_ema[k] + 0.05 * ce_per.mean().item()
    parent = model.protos[k][y]
    with torch.no_grad():
        qd = floor_simplex(q.detach())
        oh = F.one_hot(y, model.num_classes).float()
        cnt = oh.sum(0)
        present = cnt > 0
        means = (oh.t() @ qd)[present] / cnt[present].unsqueeze(1)
        model.protos[k][present] = model.ema_m * model.protos[k][present] \
            + (1 - model.ema_m) * means
    m, a_tar_real, _, _ = model.milestone(parent, q.detach(), y, k, alpha_s2)
    hinge = (ce_per - a_tar_real).clamp_min(0)
    loss = hinge.mean() + model.mu * hellinger2(floor_simplex(q), m).mean()
    if model.aux_ce > 0:
        loss = loss + model.aux_ce * F.cross_entropy(logits, y, label_smoothing=model.ls)
    loss.backward()
    opt.step(); opt.zero_grad()
    return x.detach(), hinge.detach().mean().item()


def main():
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    nccl_ok = torch.cuda.is_available() and dist.is_nccl_available()
    backend = os.environ.get('BACKEND', 'nccl' if nccl_ok else 'gloo')
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
    torch.manual_seed(seed)                               # same init on all ranks
    net = os.environ.get('NET', 'r50')
    split = os.environ.get('SPLIT', '8,8' if net == 'r50' else '3,6,11,13')
    os.environ['SPLIT'] = split
    K = len(split.split(','))
    assert K == world, f'WORLD_SIZE={world} must equal the number of modules K={K}'
    n_epochs = int(os.environ.get('EPOCHS', '90'))
    bs = int(os.environ.get('BATCH', '256'))
    warmup = int(os.environ.get('WARMUP', '5'))
    base_lr = float(os.environ.get('LR', '0.1')) * bs / 256
    wd = float(os.environ.get('WD', '1e-4'))
    res = int(os.environ.get('TRAIN_RES', '224'))
    n_cls = int(os.environ.get('NUM_CLASSES', '1000'))

    model = R50ImageNet(num_classes=n_cls, K=K, sched=os.environ.get('SCHED', 's3'),
                        stages=R101_STAGES if net == 'r101' else R50_STAGES,
                        lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
                        a_tgt=float(os.environ.get('A_TGT', '0.15')),
                        alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
                        mu=float(os.environ.get('MU', '1.0')),
                        ratchet=os.environ.get('RATCHET', '1') == '1',
                        delta_t=float(os.environ.get('DELTA_T', '0.2')),
                        aux_ce=float(os.environ.get('AUX_CE', '0.1')))
    shapes = interface_shapes(model, res)
    mods = stage_modules(model, rank, K)
    for m in mods:
        m.to(dev)
    model.protos = model.protos.to(dev)
    if rank == 0:
        model.input_proj = model.input_proj.to(dev)
    params = [p for m in mods for p in m.parameters()]
    opt = SGD(params, lr=base_lr, momentum=0.9, nesterov=True, weight_decay=wd)
    start_ep = 0
    ck = os.environ.get('CKPT', '')
    if ck and os.path.exists(f'{ck}.rank{rank}'):
        st = torch.load(f'{ck}.rank{rank}', map_location=dev)
        for m, s in zip(mods, st['mods']):
            m.load_state_dict(s)
        if 'opt' in st:
            opt.load_state_dict(st['opt'])
        model.protos.copy_(st['protos'].to(dev))
        start_ep = int(st.get('epoch', 0))
        log(rank, f'[ckpt] loaded {ck}.rank{rank} @ epoch {start_ep}')
    ep_t = torch.tensor([start_ep], dtype=torch.long,
                        device=dev if backend == 'nccl' else 'cpu')
    dist.all_reduce(ep_t, op=dist.ReduceOp.MIN)           # all ranks resume from the minimum epoch
    start_ep = int(ep_t.item())
    n_par = sum(p.numel() for p in params) / 1e6
    log(rank, f'stage={net}/{split} blocks={len(model.groups[rank])} '
              f'in={shapes[rank - 1] if rank else (3, res, res)} out={shapes[rank]} '
              f'params={n_par:.1f}M dev={dev} backend={backend}')

    if rank == 0:
        if fake:
            steps = int(os.environ.get('STEPS', '100'))
            val_steps = 1
        else:
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
            root = os.environ.get('DATA_ROOT', './data/imagenet')
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            extra = []
            if os.environ.get('AUTOAUG', '1') == '1':
                from torchvision.transforms import AutoAugment, AutoAugmentPolicy
                extra = [AutoAugment(AutoAugmentPolicy.IMAGENET)]
            tr_tf = transforms.Compose([transforms.RandomResizedCrop(res),
                                        transforms.RandomHorizontalFlip(), *extra,
                                        transforms.ToTensor(), transforms.Normalize(mean, std)])
            va_tf = transforms.Compose([transforms.Resize(int(res * 256 / 224)),
                                        transforms.CenterCrop(res),
                                        transforms.ToTensor(), transforms.Normalize(mean, std)])
            tr = datasets.ImageFolder(os.path.join(root, 'train'), transform=tr_tf)
            va = datasets.ImageFolder(os.path.join(root, 'val'), transform=va_tf)
            wk = int(os.environ.get('WORKERS', '8'))
            train_loader = DataLoader(tr, bs, shuffle=True, num_workers=wk, drop_last=True,
                                      pin_memory=use_cuda, persistent_workers=wk > 0)
            val_loader = DataLoader(va, bs, shuffle=False, num_workers=max(wk // 2, 2),
                                    drop_last=True, pin_memory=use_cuda,
                                    persistent_workers=True)
            steps, val_steps = len(train_loader), len(val_loader)
        meta = torch.tensor([steps, val_steps], dtype=torch.long, device=dev)
    else:
        meta = torch.zeros(2, dtype=torch.long, device=dev)
    dist.broadcast(meta, src=0)
    steps, val_steps = int(meta[0]), int(meta[1])
    if rank == 0:
        log(rank, f'steps/epoch={steps} val_steps={val_steps} bs={bs}')

    C_in = shapes[rank - 1] if rank > 0 else None

    def set_lr(ep):
        lr = base_lr * (ep + 1) / warmup if ep < warmup else \
            base_lr * 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(n_epochs - warmup, 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        return lr

    def recv_pack(extra_logp=False):
        h = torch.empty(bs, *C_in, device=dev)
        y = torch.empty(bs, dtype=torch.long, device=dev)
        u0 = torch.empty(bs, device=dev)
        a2 = torch.empty(bs, device=dev)
        pack = [h, y, u0, a2]
        if extra_logp:
            pack.append(torch.empty(bs, n_cls, device=dev))
        for t in pack:
            dist.recv(t, src=rank - 1)
        return pack

    def send_pack(tensors):
        reqs = [dist.isend(t.contiguous(), dst=rank + 1) for t in tensors]
        return reqs, tensors

    for epoch in range(start_ep, n_epochs):
        lr_now = set_lr(epoch)
        for m in mods:
            m.train()
        t0, pend, hinge_sum = time.time(), None, 0.0
        correct, seen = 0, 0
        data_iter = iter(train_loader) if (rank == 0 and not fake) else None
        for it in range(steps):
            if rank == 0:
                if fake:
                    img = torch.randn(bs, 3, res, res, device=dev)
                    y = torch.randint(0, n_cls, (bs,), device=dev)
                else:
                    img, y = next(data_iter)
                    img = img.to(dev, non_blocking=True)
                    y = y.to(dev, non_blocking=True)
                with torch.no_grad():
                    q0f = floor_simplex(model.input_readout(img))
                    u0 = ((-q0f.gather(1, y[:, None]).squeeze(1).log())
                          - model.lam_floor).clamp_min(0)
                    gap = max(model.a_tgt - model.lam_floor, 1e-3)
                    a2 = (1 - (gap / u0.clamp_min(gap)) ** (1.0 / model.R)) \
                        .clamp(0.0, model.alpha_max)
                h, hg = local_solo_step(model, 0, img, y, a2, opt)
                hinge_sum += hg
                if pend:
                    for r in pend[0]:
                        r.wait()
                pend = send_pack([h, y, u0.contiguous(), a2.contiguous()])
            elif rank < K - 1:
                h_in, y, u0, a2 = recv_pack()
                h, hg = local_solo_step(model, rank, h_in, y, a2, opt)
                hinge_sum += hg
                if pend:
                    for r in pend[0]:
                        r.wait()
                pend = send_pack([h, y, u0, a2])
            else:
                h_in, y, u0, a2 = recv_pack()
                x = model.run_group(K - 1, h_in)
                logits = model.fc(model.pool_features(x))
                loss = F.cross_entropy(logits, y, label_smoothing=model.ls)
                loss.backward()
                opt.step(); opt.zero_grad()
                correct += logits.argmax(1).eq(y).sum().item(); seen += bs
        if pend:
            for r in pend[0]:
                r.wait()
        dist.barrier()
        dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated(dev) / 1e9 if use_cuda else 0.0
        if rank == K - 1:
            log(rank, f'[epoch {epoch + 1}/{n_epochs}] train={correct / max(seen, 1):.4f} '
                      f'lr={lr_now:.4f} {steps * bs / dt:.0f} img/s wall={dt:.0f}s '
                      f'peak_mem={mem:.2f}GB')
        else:
            log(rank, f'[epoch {epoch + 1}] mean_hinge={hinge_sum / steps:.3f} '
                      f'wall={dt:.0f}s peak_mem={mem:.2f}GB')

        for m in mods:                                    # eval: same pipeline + ens accumulation
            m.eval()
        with torch.no_grad():
            val_iter = iter(val_loader) if (rank == 0 and not fake) else None
            correct = ens_correct = seen = 0
            pend = None
            for it in range(val_steps):
                if rank == 0:
                    if fake:
                        img = torch.randn(bs, 3, res, res, device=dev)
                        y = torch.randint(0, n_cls, (bs,), device=dev)
                    else:
                        img, y = next(val_iter)
                        img = img.to(dev, non_blocking=True)
                        y = y.to(dev, non_blocking=True)
                    h = model.run_group(0, img)
                    logp = F.log_softmax(model.heads[0](h), dim=-1)
                    z = torch.zeros(bs, device=dev)
                    if pend:
                        for r in pend[0]:
                            r.wait()
                    pend = send_pack([h, y, z, z.clone(), logp])
                elif rank < K - 1:
                    h_in, y, u0, a2, logp = recv_pack(extra_logp=True)
                    h = model.run_group(rank, h_in)
                    logp = logp + F.log_softmax(model.heads[rank](h), dim=-1)
                    if pend:
                        for r in pend[0]:
                            r.wait()
                    pend = send_pack([h, y, u0, a2, logp])
                else:
                    h_in, y, u0, a2, logp = recv_pack(extra_logp=True)
                    x = model.run_group(K - 1, h_in)
                    logits = model.fc(model.pool_features(x))
                    correct += logits.argmax(1).eq(y).sum().item()
                    ens = logp + F.log_softmax(logits, dim=-1)
                    ens_correct += ens.argmax(1).eq(y).sum().item()
                    seen += bs
            if pend:
                for r in pend[0]:
                    r.wait()
            dist.barrier()
            if rank == K - 1:
                log(rank, f'[epoch {epoch + 1}] val={correct / max(seen, 1):.4f} '
                          f'ens={ens_correct / max(seen, 1):.4f}')

        if ck:
            torch.save({'mods': [m.state_dict() for m in mods], 'opt': opt.state_dict(),
                        'protos': model.protos.detach().cpu(), 'epoch': epoch + 1},
                       f'{ck}.rank{rank}')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
