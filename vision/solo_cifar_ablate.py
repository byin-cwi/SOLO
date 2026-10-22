# solo_cifar_ablate.py: solo_cifar.py plus ablation arms (target-side baselines, prefix-ensemble
# self-distillation, optimizer switch). Milestone, certificate, schedules and shared env vars are as in
# the solo_cifar.py header (TERM_LS is not available here).
#
# MODE (default md): md | md_ce | md_ema | solo | sll | bp | dsn as in solo_cifar.py, plus
#   ols   online class-level soft labels (EMA variant of Zhang et al. 2021): per interface a (C,C) table,
#         class-mean EMA (OLS_M) of correctly classified samples' predictions; loss = softCE(table[y]) + AUX_CE*CE
#   tfkd  Tf-KD_reg virtual teacher (Yuan et al. 2020): p^d = TFKD_A on y, (1-TFKD_A)/(C-1) elsewhere;
#         loss = (1-TFKD_ALPHA)*CE + TFKD_ALPHA*tau^2*KL(p^d || softmax(z/tau)) + AUX_CE*CE, tau = TFKD_TAU
#   ols/tfkd change only the local target; no certificate is logged and INNER_T>1 is not supported.
#
# DISTILL=w > 0: add w*tau^2*KL(teacher || softmax(z/tau)), tau = DTAU, teacher = softmax of the mean of the
#   upstream heads' detached (post-GS-refresh) log_softmax(z/tau). No cross-module gradient.
#   DISTILL_WHO=final: terminal head only; all: every module k also distills its prefix <= k-1.
# OPT=adamw: AdamW with epoch-level linear warmup (WARMUP_EP) + cosine; default SGD.
# Reference: md, K=15, HEAD_DEPTH=4, HEAD_WIDEN=2, 200 ep: 92.68 (ensemble 94.46).
#
# Usage:
#   python solo_cifar_ablate.py                 # md, K=2
#   MODE=tfkd K_MODULES=15 python solo_cifar_ablate.py
#   VERIFY=1 python solo_cifar_ablate.py        # smoke test
#
# Extra env (default): OPT(sgd) WARMUP_EP(5) DISTILL(0) DTAU(3.0) DISTILL_WHO(final)
#   OLS_M(0.99) TFKD_A(0.9) TFKD_ALPHA(0.5) TFKD_TAU(6.0)

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

DEVICE = os.environ.get('SLL_DEVICE', 'cuda:0') if torch.cuda.is_available() else 'cpu'
EPS_FLOOR = float(os.environ.get('EPS_FLOOR', '1e-3'))   # simplex floor (1e-4 recommended for C100)
CERT_C = 2 * math.sqrt(2) / math.sqrt(EPS_FLOOR)


def floor_simplex(q, eps=EPS_FLOOR):
    q = q.clamp_min(eps)
    return q / q.sum(dim=-1, keepdim=True)


def hellinger2(q, m):
    """D_H2(q,m) = 1 - sum sqrt(q m), per sample."""
    return (1.0 - (q * m).clamp_min(0).sqrt().sum(dim=-1)).clamp_min(0)


def align_loss(q, p, detach_p=True, align_type='bc', eps=1e-12):
    """Adjacent-readout alignment of the SLL baseline (used by MODE=sll)."""
    if align_type == 'none':
        return q.sum() * 0.0
    if detach_p:
        p = p.detach()
    q = q.clamp_min(eps); q = q / q.sum(dim=-1, keepdim=True)
    p = p.clamp_min(eps); p = p / p.sum(dim=-1, keepdim=True)
    if align_type == 'kl':
        per = (q * (q.log() - p.log())).sum(dim=-1)
    elif align_type == 'euclidean':
        per = 0.5 * ((q - p) ** 2).sum(dim=-1)
    else:
        log_bc = torch.logsumexp(0.5 * (q.log() + p.log()), dim=-1)
        per = (1.0 - log_bc.exp()) if align_type == 'hellinger' else (-log_bc)
    return per.mean()


def conv3x3(i, o, s=1):
    return nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class VGGBlock(nn.Module):
    """VGG conv+BN+ReLU(+MaxPool) unit."""
    def __init__(self, i, o, pool=False):
        super().__init__()
        layers = [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class PatchEmbed(nn.Module):
    """ViT patch embedding (CIFAR 32x32, patch 4 -> 64 tokens) + learnable positional embedding."""
    def __init__(self, dim=192, patch=4, img=32):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        n = (img // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos


class ViTBlock(nn.Module):
    """Pre-LN Transformer encoder block."""
    def __init__(self, dim=192, heads=3, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class TokenHead(nn.Module):
    """ViT local readout head: LN + mean-pool tokens + MLP -> logits. depth sets the number of MLP layers."""
    def __init__(self, dim, num_classes, hidden=256, widen=1, depth=1):
        super().__init__()
        h = int(hidden * widen)
        layers = [nn.LayerNorm(dim), nn.Linear(dim, h), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(h, h), nn.GELU()]
        layers.append(nn.Linear(h, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x.mean(dim=1))


class ConvBlock(nn.Module):
    """conv+BN+ReLU(+MaxPool) unit of ResNet-9 (DAWNBench lineage)."""
    def __init__(self, i, o, pool=False):
        super().__init__()
        layers = [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class Res9Block(nn.Module):
    """ResNet-9 residual unit: x + convblock(convblock(x))."""
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(ConvBlock(ch, ch), ConvBlock(ch, ch))

    def forward(self, x):
        return x + self.f(x)


class Bottleneck(nn.Module):
    """ResNet-50 bottleneck (CIFAR-adapted)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        out = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
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
    """Trainable readout head: depth 3x3 convs + GAP + 2 FC -> num_classes."""
    def __init__(self, inplanes, num_classes=10, hidden=128, widen=1, depth=1, strided=False):
        super().__init__()
        if inplanes <= 64:                      # R32 channels (16/32/64)
            mid = int((32 if inplanes == 16 else 64) * widen)
            stride = 2 if inplanes in (16, 32) else 1
        else:                                   # R50 channels (256/512/1024/2048)
            mid = int(min(inplanes // 2, 256) * widen)
            stride = 2 if inplanes <= 512 else 1
        layers = [nn.Conv2d(inplanes, mid, 3, stride, 1, bias=False),
                  nn.BatchNorm2d(mid), nn.ReLU()]
        for _ in range(depth - 1):              # strided: downsample in every extra conv
            layers += [nn.Conv2d(mid, mid, 3, 2 if strided else 1, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU()]
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                   nn.Linear(mid, int(hidden * widen)), nn.ReLU(inplace=True),
                   nn.Linear(int(hidden * widen), num_classes)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class MDHeadResNet32(nn.Module):
    def __init__(self, num_classes=10, lr=0.1, K=2, gs_mode='gs', mode='md',
                 sched='s2', lam_floor=0.1, a_tgt=0.15, alpha_max=0.95, mu=1.0,
                 ratchet=True, delta_t=0.2, aux_ce=0.1,
                 lam=1.0, align_type='none', label_smoothing=0.0, carry=False, t_max=200):
        super().__init__()
        assert gs_mode in ('jacobi', 'gs') and mode in ('md', 'md_ce', 'md_ema', 'solo', 'sll', 'bp', 'dsn',
                                                       'ols', 'tfkd') \
            and sched in ('s1', 's2', 's3', 's4')
        self.device = DEVICE
        self.gs_mode, self.mode, self.sched = gs_mode, mode, sched
        self.lam_floor, self.a_tgt, self.alpha_max, self.mu = lam_floor, a_tgt, alpha_max, mu
        self.ratchet, self.delta_t = ratchet, delta_t
        self.aux_ce = max(aux_ce, 0.1) if (carry and mode != 'sll') else aux_ce   # with carry (non-sll): aux CE >= 0.1
        self.lam, self.align_type, self.ls = lam, align_type, label_smoothing
        self.num_classes, self.carry = num_classes, carry
        # FREEZE_M0=1: module 0 (backbone+head) stays at initialization, BN in eval mode
        self.freeze_m0 = os.environ.get('FREEZE_M0', '0') == '1'

        # ---- backbone (ARCH) ----
        self.arch = os.environ.get('ARCH', 'r32')
        self.blocks = nn.ModuleList()
        block_ch = []
        if self.arch == 'r9':                 # DAWNBench-style ResNet-9
            self.stem = ConvBlock(3, 64)
            for m, ch in [(ConvBlock(64, 128, pool=True), 128), (Res9Block(128), 128),
                          (ConvBlock(128, 256, pool=True), 256), (ConvBlock(256, 512, pool=True), 512),
                          (Res9Block(512), 512), (ConvBlock(512, 1028, pool=True), 1028),
                          (Res9Block(1028), 1028)]:
                self.blocks.append(m)
                block_ch.append(ch)
            self.fc = nn.Linear(1028, num_classes)
        elif self.arch == 'vgg11':               # VGG-11+BN; MaxPool folded into the preceding block
            self.stem = nn.Identity()
            cfg = [(3, 64, True), (64, 128, True), (128, 256, False), (256, 256, True),
                   (256, 512, False), (512, 512, True), (512, 512, False), (512, 512, True)]
            for i_ch, o_ch, pool in cfg:
                self.blocks.append(VGGBlock(i_ch, o_ch, pool))
                block_ch.append(o_ch)
            self.fc = nn.Linear(512, num_classes)
        elif self.arch == 'vit':                 # ViT-Tiny style: patch4, dim192, depth8, heads3
            dim = int(os.environ.get('VIT_DIM', '192'))
            self.stem = PatchEmbed(dim=dim, patch=4, img=32)
            for _ in range(int(os.environ.get('VIT_DEPTH', '8'))):
                self.blocks.append(ViTBlock(dim=dim, heads=int(os.environ.get('VIT_HEADS', '3'))))
                block_ch.append(dim)
            self.fc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        elif self.arch == 'r50':
            self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            in_ch = 64
            for planes, n, first_stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
                for b in range(n):
                    stride = first_stride if b == 0 else 1
                    self.blocks.append(Bottleneck(in_ch, planes, stride))
                    in_ch = planes * Bottleneck.expansion
                    block_ch.append(in_ch)
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.stem = nn.Sequential(conv3x3(3, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            in_ch = 16
            for out_ch, n, first_stride in [(16, 5, 1), (32, 5, 2), (64, 5, 2)]:
                for b in range(n):
                    stride = first_stride if b == 0 else 1
                    self.blocks.append(BasicBlock(in_ch, out_ch, stride))
                    in_ch = out_ch
                    block_ch.append(out_ch)
            self.fc = nn.Linear(64, num_classes)
        self.n_blocks = len(self.blocks)

        K = max(2, min(K, self.n_blocks))
        self.groups = [list(g) for g in np.array_split(np.arange(self.n_blocks), K)]
        split_env = os.environ.get('SPLIT', '')          # non-uniform split, e.g. SPLIT=12,4
        if split_env:
            sizes = [int(s) for s in split_env.split(',')]
            assert sum(sizes) == self.n_blocks, f'SPLIT {sizes} != {self.n_blocks} blocks'
            idx, self.groups = 0, []
            for s in sizes:
                self.groups.append(list(range(idx, idx + s)))
                idx += s
            K = len(sizes)
        self.K = K
        self.R = K - 1                                    # local modules = ladder steps
        self._init_weights()
        head_widen = float(os.environ.get('HEAD_WIDEN', '1'))
        head_depth = int(os.environ.get('HEAD_DEPTH', '1'))
        strided = os.environ.get('STRIDED_HEAD', '0') == '1'
        if self.arch == 'vit':
            self.heads = nn.ModuleList(
                [TokenHead(block_ch[self.groups[k][-1]], num_classes, widen=head_widen,
                           depth=head_depth) for k in range(K - 1)])
        else:
            self.heads = nn.ModuleList(
                [ReadoutHead(block_ch[self.groups[k][-1]], num_classes, widen=head_widen,
                             depth=head_depth, strided=strided) for k in range(K - 1)])
        # s4: EMA of the attained local CE per module; ladder target >= S4_KAPPA * ach_ema
        self.ach_ema = [math.log(num_classes)] * (K - 1)
        self.s4_kappa = float(os.environ.get('S4_KAPPA', '0.7'))
        self.dsn_aux = float(os.environ.get('DSN_AUX', '0.3'))    # aux CE weight for MODE=dsn
        self.distill = float(os.environ.get('DISTILL', '0'))      # prefix-ensemble self-distillation weight (0=off)
        self.dtau = float(os.environ.get('DTAU', '3.0'))          # distillation temperature
        self.distill_who = os.environ.get('DISTILL_WHO', 'final') # final | all

        self.module_params = []
        for k in range(K):
            params = list(self.stem.parameters()) if k == 0 else []
            for bi in self.groups[k]:
                params += list(self.blocks[bi].parameters())
            if k < K - 1:
                params += list(self.heads[k].parameters())
            else:
                params += list(self.fc.parameters())
            self.module_params.append(params)
        wd = float(os.environ.get('WD', '1e-4'))
        # OPT=adamw: AdamW + epoch-level warmup + cosine; default SGD
        self.opt_name = os.environ.get('OPT', 'sgd')
        if self.opt_name == 'adamw':
            self.opts = [torch.optim.AdamW(p, lr=lr, weight_decay=wd)
                         for p in self.module_params]
        else:
            self.opts = [SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
                         for p in self.module_params]
        self.inner_t = max(1, int(os.environ.get('INNER_T', '1')))   # local optimizer steps per batch (gs only)
        # ols / tfkd knobs
        self.ols_m = float(os.environ.get('OLS_M', '0.99'))
        self.tfkd_a = float(os.environ.get('TFKD_A', '0.9'))
        self.tfkd_alpha = float(os.environ.get('TFKD_ALPHA', '0.5'))
        self.tfkd_tau = float(os.environ.get('TFKD_TAU', '6.0'))
        assert self.inner_t == 1 or mode not in ('ols', 'tfkd'), 'ols/tfkd ablation does not support INNER_T>1'
        if self.opt_name == 'adamw':
            wu = max(1, int(os.environ.get('WARMUP_EP', '5')))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.opts[0],
                lambda e: min((e + 1) / wu, 0.5 * (1 + math.cos(math.pi * e / max(t_max, 1)))))
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opts[0], T_max=t_max, eta_min=0.0)

        # ---- label-free input readout q_0 (fixed random projection) ----
        in_proj = min(3 * 32 * 32, 2048)
        self.input_proj = torch.randn(in_proj, num_classes, device=self.device) / np.sqrt(in_proj)

        # ---- ols_tab (ols) and protos (md_ema / solo): (K-1, C, C) class tables in the simplex, init uniform ----
        self.register_buffer('ols_tab', torch.full((K - 1, num_classes, num_classes),
                                                   1.0 / num_classes))   # OLS online soft-label table
        self.register_buffer('protos', torch.full((K - 1, num_classes, num_classes),
                                                  1.0 / num_classes))
        self.ema_m = float(os.environ.get('EMA_M', '0.99'))

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
        return F.softmax(flat @ self.input_proj, dim=1)

    def pool_features(self, h):
        """Final feature pooling: conv -> GAP; vit -> token mean (LN is inside fc)."""
        if self.arch == 'vit':
            return h.mean(dim=1)
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def run_group(self, k, x_in, bn_eval=False):
        mods = [self.stem] if k == 0 else []
        mods += [self.blocks[bi] for bi in self.groups[k]]
        bns = [m for top in mods for m in top.modules() if isinstance(m, nn.BatchNorm2d)]
        if bn_eval:
            for bn in bns:
                bn.eval()
        h = x_in
        for top in mods:
            h = top(h)
        if bn_eval:
            for bn in bns:
                bn.train()
        return h

    # ------------------------------------------------------------------
    #  Milestone block (all no_grad, closed form): returns m [B,C], a_tar_realized [B], alpha [B]
    # ------------------------------------------------------------------
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
            alpha = torch.full_like(a, 1 - (gap / max(math.log(self.num_classes) - self.lam_floor, gap)) ** (1.0 / self.R))
        elif self.sched == 's2':                          # per-sample equi-progress, computed once from q0
            alpha = alpha_s2
        else:                                             # s3/s4: closed loop on the remaining depth
            rem = self.R - k
            alpha = 1 - (gap / u.clamp_min(gap)) ** (1.0 / rem)
        alpha = alpha.clamp(0.0, self.alpha_max)
        a_tar = torch.where(below, a, self.lam_floor + (1 - alpha) * u)
        if self.sched == 's4':                            # s4: reachability clamp
            a_tar = torch.maximum(a_tar, torch.full_like(a_tar, self.s4_kappa * self.ach_ema[k]))
        m_y = (-a_tar).exp()                              # = p_y^(1-alpha) * exp(-lam)^alpha
        t = ((m_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.ratchet:                                  # windowed ratchet: do not pull back a student already past the milestone
            q_y = floor_simplex(q_detached).gather(1, y[:, None]).squeeze(1).clamp(EPS_FLOOR, 1 - 1e-6)
            t_stu = ((q_y - p_y) / (1 - p_y).clamp_min(1e-6)).clamp(0.0, 1.0)
            t = torch.minimum(torch.maximum(t, t_stu), t + self.delta_t)
        m = (1 - t)[:, None] * p
        m[torch.arange(B, device=p.device), y] += t       # m = (1-t) p + t e_y
        # below-floor: m := sg(q); the gradients of soft-CE and Hellinger are exactly zero at q=m
        m = torch.where(below[:, None], floor_simplex(q_detached), m)
        a_tar_real = -m.gather(1, y[:, None]).squeeze(1).clamp_min(EPS_FLOOR).log()
        return m, a_tar_real, alpha, below

    def forward_with_update(self, img, y):
        if self.mode == 'bp':                             # end-to-end BP baseline
            for opt in self.opts:
                opt.zero_grad()
            logits = self.forward(img)
            loss = F.cross_entropy(logits, y, label_smoothing=self.ls)
            loss.backward()
            for opt in self.opts:
                opt.step()
            return logits, loss.item(), None
        if self.mode == 'dsn':                            # DSN: backprop through backbone and auxiliary heads
            for opt in self.opts:
                opt.zero_grad()
            logits, outs = self.forward_all(img)          # no detach: gradients flow end to end
            loss = F.cross_entropy(logits, y, label_smoothing=self.ls) \
                + self.dsn_aux * sum(F.cross_entropy(z, y, label_smoothing=self.ls) for z in outs)
            loss.backward()
            for opt in self.opts:
                opt.step()
            return logits, loss.item(), None
        if self.freeze_m0:                                # FREEZE_M0: module 0 in eval mode (no updates, frozen BN stats)
            self.stem.eval()
            for bi in self.groups[0]:
                self.blocks[bi].eval()
            self.heads[0].eval()
        for opt in self.opts:
            opt.zero_grad()
        gs = self.gs_mode == 'gs'
        prev_q = self.input_readout(img.detach())         # q_0, label-free
        prev_z = None
        x = img
        # certificate quantities
        q0f = floor_simplex(prev_q)
        u0 = ((-q0f.gather(1, y[:, None]).squeeze(1).log()) - self.lam_floor).clamp_min(0)
        gap = max(self.a_tgt - self.lam_floor, 1e-3)
        alpha_s2 = (1 - (gap / u0.clamp_min(gap)) ** (1.0 / self.R)).clamp(0.0, self.alpha_max)
        alphas, errs, u_last = [], [], None
        lp_sum, lp_n = None, 0                            # sum of upstream log_softmax(z/tau), detached

        for k in range(self.K - 1):                       # local modules (MD / SLL loss)
            x_in = x.detach()
            x = self.run_group(k, x_in, bn_eval=False)
            logits = self.heads[k](x)
            if self.carry and prev_z is not None:
                logits = logits + prev_z
            q = F.softmax(logits, dim=-1)
            ce_per = F.cross_entropy(logits, y, reduction='none')
            if k < len(self.ach_ema):                     # s4: attained local CE (EMA)
                self.ach_ema[k] = 0.95 * self.ach_ema[k] + 0.05 * ce_per.mean().item()

            if self.mode == 'sll':                        # baseline: CE + alignment
                loss = F.cross_entropy(logits, y, label_smoothing=self.ls) \
                    + self.lam * align_loss(q, prev_q, align_type=self.align_type)
            elif self.mode == 'ols':                      # online class-level soft labels
                m_t = self.ols_tab[k][y]                  # (n,C), detached buffer
                loss = -(m_t * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                if self.aux_ce > 0:
                    loss = loss + self.aux_ce * F.cross_entropy(logits, y, label_smoothing=self.ls)
                with torch.no_grad():                     # class-mean EMA of correct samples' predictions
                    correct = logits.argmax(1) == y
                    if correct.any():
                        qd = floor_simplex(q[correct].detach())
                        oh = F.one_hot(y[correct], self.num_classes).float()
                        cnt = oh.sum(0)
                        present = cnt > 0
                        means = (oh.t() @ qd)[present] / cnt[present].unsqueeze(1)
                        self.ols_tab[k][present] = self.ols_m * self.ols_tab[k][present] \
                            + (1 - self.ols_m) * means
            elif self.mode == 'tfkd':                     # Tf-KD_reg virtual teacher
                with torch.no_grad():
                    p_d = torch.full_like(q, (1 - self.tfkd_a) / (self.num_classes - 1))
                    p_d.scatter_(1, y[:, None], self.tfkd_a)
                    t_lp = p_d.log()
                s_lp = F.log_softmax(logits / self.tfkd_tau, dim=-1)
                kd = (p_d * (t_lp - s_lp)).sum(-1).mean()
                loss = (1 - self.tfkd_alpha) * F.cross_entropy(logits, y, label_smoothing=self.ls) \
                    + self.tfkd_alpha * (self.tfkd_tau ** 2) * kd
                if self.aux_ce > 0:
                    loss = loss + self.aux_ce * F.cross_entropy(logits, y, label_smoothing=self.ls)
            else:
                if self.mode in ('md_ema', 'solo'):       # anchor = class prototype
                    parent_for_m = self.protos[k][y]      # read before the EMA update (target lags)
                    with torch.no_grad():                 # md_ema: parent readout; solo: own readout
                        src = q.detach() if self.mode == 'solo' else prev_q
                        qd = floor_simplex(src)
                        oh = F.one_hot(y, self.num_classes).float()
                        cnt = oh.sum(0)
                        present = cnt > 0
                        means = (oh.t() @ qd)[present] / cnt[present].unsqueeze(1)
                        self.protos[k][present] = self.ema_m * self.protos[k][present] \
                            + (1 - self.ema_m) * means
                else:
                    parent_for_m = prev_q
                m, a_tar_real, alpha_k, below = self.milestone(parent_for_m, q.detach(), y, k, alpha_s2)
                hinge = (ce_per - a_tar_real).clamp_min(0)          # certificate term (c=1)
                if self.mode in ('md', 'md_ema', 'solo'):  # default: hinge + mu * Hellinger^2(q, m)
                    d_h2 = hellinger2(floor_simplex(q), m)
                    loss = hinge.mean() + self.mu * d_h2.mean()
                else:                                     # md_ce: soft-CE to the milestone
                    loss = -(m * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                if self.aux_ce > 0:                       # auxiliary CE
                    loss = loss + self.aux_ce * F.cross_entropy(logits, y, label_smoothing=self.ls)
                if self.distill > 0 and self.distill_who == 'all' and lp_n > 0:
                    with torch.no_grad():                 # prefix (<=k-1) ensemble soft target, detached
                        t_lp = F.log_softmax(lp_sum / lp_n, dim=-1)
                    s_lp = F.log_softmax(logits / self.dtau, dim=-1)
                    loss = loss + self.distill * (self.dtau ** 2) \
                        * (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
                pend_cert = (m, a_tar_real, alpha_k, below)   # certificate accounting waits for the final (GS-refreshed) readout

            loss.backward()
            # INNER_T>1 (gs only): extra local steps, each re-running this module's forward
            if gs and self.inner_t > 1 and not (self.freeze_m0 and k == 0):
                for _t in range(self.inner_t - 1):
                    self.opts[k].step(); self.opts[k].zero_grad()
                    x_t = self.run_group(k, x_in, bn_eval=True)
                    logit_t = self.heads[k](x_t)
                    if self.carry and prev_z is not None:
                        logit_t = logit_t + prev_z
                    if self.mode == 'sll':
                        loss_t = F.cross_entropy(logit_t, y, label_smoothing=self.ls) \
                            + self.lam * align_loss(F.softmax(logit_t, -1), prev_q,
                                                    align_type=self.align_type)
                    else:
                        ce_t = F.cross_entropy(logit_t, y, reduction='none')
                        hinge_t = (ce_t - a_tar_real).clamp_min(0)
                        if self.mode in ('md', 'md_ema', 'solo'):
                            loss_t = hinge_t.mean() + self.mu * hellinger2(
                                floor_simplex(F.softmax(logit_t, -1)), m).mean()
                        else:
                            loss_t = -(m * F.log_softmax(logit_t, -1)).sum(-1).mean()
                        if self.aux_ce > 0:
                            loss_t = loss_t + self.aux_ce * F.cross_entropy(
                                logit_t, y, label_smoothing=self.ls)
                    loss_t.backward()
            prev_q = q.detach()
            prev_z = logits.detach()
            if gs:                                        # GS: step, then recompute output and readout passed downstream
                if not (self.freeze_m0 and k == 0):
                    self.opts[k].step()
                with torch.no_grad():
                    x = self.run_group(k, x_in, bn_eval=True)
                    self.heads[k].eval()
                    logits_ref = self.heads[k](x)
                    self.heads[k].train()
                    if self.carry and prev_z is not None and k > 0:
                        pass                              # prev_z already includes upstream carry; only refresh this module's head
                    prev_q = F.softmax(logits_ref, dim=-1).detach()
                    prev_z = logits_ref.detach()
            else:
                x = x.detach()

            if self.mode not in ('sll', 'ols', 'tfkd'):   # no certificate for sll/ols/tfkd
                # certificate chain on prev_z, the readout actually passed downstream (pre-update under jacobi,
                # refreshed under gs), so u_last <= bound holds per sample.
                with torch.no_grad():
                    m_c, atar_c, alpha_c, below_c = pend_cert
                    ce_c = F.cross_entropy(prev_z, y, reduction='none')
                    qf_c = floor_simplex(F.softmax(prev_z, dim=-1))
                    hinge_c = (ce_c - atar_c).clamp_min(0)
                    h_c = hellinger2(qf_c, m_c).clamp_min(0).sqrt()
                    errs.append(torch.minimum(hinge_c, CERT_C * h_c))
                    alphas.append(torch.where(below_c, torch.zeros_like(alpha_c), alpha_c))
                    u_last = (ce_c - self.lam_floor).clamp_min(0)

            if self.distill > 0:                          # add this module's final readout to the prefix pool
                with torch.no_grad():
                    lp_k = F.log_softmax(prev_z / self.dtau, dim=-1)
                    lp_sum = lp_k if lp_sum is None else lp_sum + lp_k
                    lp_n += 1

        # ---- last module: CE head ----
        x_in = x.detach()
        x = self.run_group(self.K - 1, x_in, bn_eval=False)
        logits = self.fc(self.pool_features(x))
        loss_head = F.cross_entropy(logits, y, label_smoothing=self.ls)
        if self.distill > 0 and lp_n > 0:                 # terminal head distills the full-prefix ensemble
            with torch.no_grad():
                t_lp = F.log_softmax(lp_sum / lp_n, dim=-1)
            s_lp = F.log_softmax(logits / self.dtau, dim=-1)
            loss_head = loss_head + self.distill * (self.dtau ** 2) \
                * (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean()
        loss_head.backward()
        if gs:
            self.opts[self.K - 1].step()
        else:
            for ki, opt in enumerate(self.opts):
                if not (self.freeze_m0 and ki == 0):
                    opt.step()

        cert = None
        if self.mode not in ('sll', 'ols', 'tfkd') and alphas:   # certified bound (per sample, then averaged)
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
        """Returns (final_logits, [logits of each local head]); carry semantics match training."""
        h, outs, prev_z = x, [], None
        for k in range(self.K - 1):
            h = self.run_group(k, h)
            z = self.heads[k](h)
            if self.carry and prev_z is not None:
                z = z + prev_z
            outs.append(z)
            prev_z = z
        h = self.run_group(self.K - 1, h)
        return self.fc(self.pool_features(h)), outs

    def update_lr(self):
        lr = self.scheduler.get_last_lr()[0]
        for opt in self.opts:
            for g in opt.param_groups:
                g['lr'] = lr


def evaluate(model, loader, device):
    """Returns (final-head accuracy, ensemble accuracy or None). Ensemble = sum of per-head log-softmax."""
    model.eval()
    correct, correct_ens = 0, 0
    use_ens = model.mode != 'bp'
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
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


def train_cifar10(n_epochs=200, batch_size=128, lr=0.1, K=2, gs_mode='gs', mode='md',
                  sched='s2', lam_floor=0.1, a_tgt=0.15, alpha_max=0.95, mu=1.0,
                  ratchet=True, delta_t=0.2, aux_ce=0.1,
                  lam=1.0, align_type='none', label_smoothing=None, autoaug=True,
                  carry=False, data_root='./data'):
    device = DEVICE
    dataset = os.environ.get('DATASET', 'c10')
    if dataset == 'c100':
        num_classes, ds_cls = 100, datasets.CIFAR100
        mean = (0.5071, 0.4865, 0.4409); std = (0.2673, 0.2564, 0.2762)
    else:
        num_classes, ds_cls = 10, datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    if label_smoothing is None:
        label_smoothing = 0.1 if mode in ('sll', 'bp', 'dsn') else 0.0   # md modes: ls=0 (the floor softens the target)
    print(f'device={device} | {mode.upper()} ResNet32 {dataset.upper()} K={K}(+BP head) gs={gs_mode} '
          f'sched={sched} lam_floor={lam_floor} a_tgt={a_tgt} mu={mu} ratchet={ratchet} '
          f'aux_ce={aux_ce} ls={label_smoothing} autoaug={autoaug} carry={carry}')
    extra = []
    print(f'  [cfg] arch={os.environ.get("ARCH", "r32")} '
          f'head_depth={os.environ.get("HEAD_DEPTH", "1")} '
          f'head_widen={os.environ.get("HEAD_WIDEN", "1")} '
          f'opt={os.environ.get("OPT", "sgd")} '
          f'distill={os.environ.get("DISTILL", "0")} dtau={os.environ.get("DTAU", "3.0")} '
          f'who={os.environ.get("DISTILL_WHO", "final")} '
          f'ols_m={os.environ.get("OLS_M", "0.99")} '
          f'tfkd={os.environ.get("TFKD_A", "0.9")}/{os.environ.get("TFKD_ALPHA", "0.5")}'
          f'/{os.environ.get("TFKD_TAU", "6.0")}', flush=True)
    if autoaug:
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            extra = [AutoAugment(AutoAugmentPolicy.CIFAR10)]
            print('  [aug] AutoAugment(CIFAR10) ON')
        except Exception as e:
            print(f'  [aug] AutoAugment unavailable ({e}); crop+flip only')
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        *extra, transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = ds_cls(root=data_root, train=True, download=True, transform=train_tf)
    test_set = ds_cls(root=data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                              drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MDHeadResNet32(num_classes=num_classes, lr=lr, K=K, gs_mode=gs_mode, mode=mode,
                           sched=sched, lam_floor=lam_floor, a_tgt=a_tgt, alpha_max=alpha_max,
                           mu=mu, ratchet=ratchet, delta_t=delta_t, aux_ce=aux_ce,
                           lam=lam, align_type=align_type, label_smoothing=label_smoothing,
                           carry=carry,
                           t_max=int(os.environ.get('T_MAX', str(n_epochs)))).to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    best = 0.0
    start_ep = 0
    anneal = os.environ.get('ANNEAL', '0') == '1'         # annealed floor lam_t
    lam_start = float(os.environ.get('LAM_START', '0.6'))
    lam_end = float(os.environ.get('LAM_END', '0.05'))
    a_gap = float(os.environ.get('A_GAP', '0.05'))
    ckpt_path = os.environ.get('CKPT', '')                # e.g. CKPT=ckpt/md_k4.pt
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        for o, s in zip(model.opts, ck['opts']):
            o.load_state_dict(s)
        start_ep, best = ck['epoch'], ck['best']
        for _ in range(start_ep):
            model.scheduler.step()
        model.update_lr()
        print(f'  [ckpt] resumed {ckpt_path} @ epoch {start_ep} best={best:.4f}')
    for epoch in range(start_ep, n_epochs):
        if anneal:                                        # cosine-annealed floor
            model.lam_floor = lam_end + (lam_start - lam_end) * 0.5 * (
                1 + math.cos(math.pi * epoch / max(n_epochs - 1, 1)))
            model.a_tgt = model.lam_floor + a_gap
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
        acc, acc_ens = evaluate(model, test_loader, device)
        best = max(best, acc)
        if acc_ens is not None:
            if not hasattr(model, '_best_ens'):
                model._best_ens = 0.0
            model._best_ens = max(model._best_ens, acc_ens)
        mem = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        cert_str = ''
        if c_acc['n']:
            n = c_acc['n']
            u_l, bd = c_acc['u_last'] / n, c_acc['bound'] / n
            cert_str = (f' | cert: u0={c_acc["u0"] / n:.3f} u_last={u_l:.3f} '
                        f'bound={bd:.3f} tight={u_l / max(bd, 1e-8):.2f}')
        ens_str = '' if acc_ens is None else f' ens={acc_ens:.4f} best_ens={model._best_ens:.4f}'
        print(f'[epoch {epoch + 1:3d}/{n_epochs}] train={correct / seen:.4f} test={acc:.4f} '
              f'best={best:.4f}{ens_str} peak_mem={mem:.2f}GB{cert_str}', flush=True)
        if ckpt_path:
            torch.save({'model': model.state_dict(), 'opts': [o.state_dict() for o in model.opts],
                        'epoch': epoch + 1, 'best': best}, ckpt_path)
    print('=' * 64)
    print(f'FINAL best test acc: {best:.4f}  (mode={mode}, K={K}, sched={sched})')
    print('=' * 64)
    return model


def verify(batch=8):
    torch.manual_seed(0)
    x = torch.randn(batch, 3, 32, 32, device=DEVICE)
    y = torch.randint(0, 10, (batch,), device=DEVICE)
    for K in (2, 4):
        for mode in ('md', 'md_ce', 'sll'):
            for gs in ('jacobi', 'gs'):
                m = MDHeadResNet32(num_classes=10, K=K, gs_mode=gs, mode=mode,
                                   sched='s2', carry=(mode == 'md')).to(DEVICE)
                m.train()
                out, loss, cert = m.forward_with_update(x.clone(), y.clone())
                cs = f" cert(u0={cert['u0']:.2f},bound={cert['bound']:.2f})" if cert else ''
                print(f'[smoke] K={K} mode={mode:5s} gs={gs:6s} ok head_loss={loss:.4f}'
                      f' out={tuple(out.shape)}{cs}')
    # milestone math self-check: ell(m)=a_tar exactly; t in [0,1]
    mm = MDHeadResNet32(num_classes=10, K=4, mode='md', ratchet=False).to(DEVICE)
    p = torch.softmax(torch.randn(64, 10, device=DEVICE) * 2, dim=-1)
    q = torch.softmax(torch.randn(64, 10, device=DEVICE) * 2, dim=-1)
    yy = torch.randint(0, 10, (64,), device=DEVICE)
    al = torch.full((64,), 0.5, device=DEVICE)
    m_, a_tar, _, below = mm.milestone(p, q, yy, k=0, alpha_s2=al)
    err = (-(m_.gather(1, yy[:, None]).squeeze(1).log()) - a_tar).abs().max().item()
    print(f'[math] milestone exactness max|ell(m)-a_tar| = {err:.2e} (expect ~0); '
          f'below-floor n={below.sum().item()}; simplex sum err='
          f'{(m_.sum(-1) - 1).abs().max().item():.2e}')


SEED = int(os.environ.get('SEED', '42'))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

if __name__ == '__main__':
    if os.environ.get('VERIFY'):
        verify()
    else:
        train_cifar10(
            n_epochs=int(os.environ.get('EPOCHS', '200')),
            batch_size=int(os.environ.get('BATCH', '128')),
            lr=float(os.environ.get('LR', '0.1')),
            K=int(os.environ.get('K_MODULES', '2')),
            gs_mode=os.environ.get('GS_MODE', 'gs'),
            mode=os.environ.get('MODE', 'md'),
            sched=os.environ.get('SCHED', 's2'),
            lam_floor=float(os.environ.get('LAM_FLOOR', '0.1')),
            a_tgt=float(os.environ.get('A_TGT', '0.15')),
            alpha_max=float(os.environ.get('ALPHA_MAX', '0.95')),
            mu=float(os.environ.get('MU', '1.0')),
            ratchet=os.environ.get('RATCHET', '1') == '1',
            delta_t=float(os.environ.get('DELTA_T', '0.2')),
            aux_ce=float(os.environ.get('AUX_CE', '0.1')),
            lam=float(os.environ.get('LAM', '1.0')),
            align_type=os.environ.get('ALIGN_TYPE', 'none'),
            label_smoothing=(float(os.environ['LABEL_SMOOTH'])
                             if 'LABEL_SMOOTH' in os.environ else None),
            autoaug=os.environ.get('AUTOAUG', '1') == '1',
            carry=os.environ.get('CARRY', '0') == '1',
            data_root=os.environ.get('DATA_ROOT', './data'),
        )
