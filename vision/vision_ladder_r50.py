"""vision_ladder_r50.py: ResNet-50 version of vision_ladder_r32.py.

Same env interface, training loop, evaluation, CKPT and FINAL line as vision_ladder_r32.py (run_ladder_abl.sh
runs either via SCRIPT). Differences:
  (1) backbone: 16 Bottlenecks in 4 stages [3,4,6,3], terminal feature d=2048;
  (2) TIE_DIM is forced to 2048 (the terminal fc reads the pooled feature directly), so the shared readout is C x 2048;
  (3) auxiliary-head channel rule has an R50 branch (inplanes 256/512/1024/2048), as in solo_cifar_ablate.py.

Arms:
  random  TIE=1 TIE_MODE=random ; untie TIE=0 ; tie TIE=1 TIE_MODE=trained
  stale   TIE=1 TIE_MODE=stale STALE_FC=<ckpt saved by this script's tie arm>
Env vars as in vision_ladder_r32.py, except defaults K_MODULES 16 (15 aux heads), EPOCHS 100, BATCH 128;
CKPT also saves .ep10 / .ep25 / .ep50 snapshots.
Self-check: DRYRUN=1 DATASET=tiny python vision_ladder_r50.py (shapes and parameter count, CPU).
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

E = os.environ.get
DEVICE = torch.device(E("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
SEED = int(E("SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET = E("DATASET", "tiny")                    # tiny | c100 | c10
assert DATASET in ("tiny", "c100", "c10")
MODE = E("MODE", "sll")
K_MODULES = int(E("K_MODULES", "16"))             # R50: 16 bottlenecks => 15 aux heads
HEAD_DEPTH = int(E("HEAD_DEPTH", "4"))
HEAD_WIDEN = float(E("HEAD_WIDEN", "2"))
HEAD_HIDDEN = int(E("HEAD_HIDDEN", "128"))
STRIDED_HEAD = E("STRIDED_HEAD", "0") == "1"
GS_MODE = E("GS_MODE", "jacobi")
TIE = E("TIE", "1") == "1"
TIE_MODE = E("TIE_MODE", "trained")                  # trained | random | stale
STALE_FC = E("STALE_FC", "")                     # stale: fc.weight from a ckpt, frozen
TIE_SCALE = E("TIE_SCALE", "1") == "1"
FEAT_DIM = 2048                                    # R50 terminal pooled feature dim (512 x 4)
# TIE_DIM must equal FEAT_DIM (the terminal fc reads the pooled feature directly); other values are overridden.
_want = int(E("TIE_DIM", str(FEAT_DIM)))
if _want != FEAT_DIM:
    print(f"[warn] R50 ignores given TIE_DIM={_want}, forced = terminal feature dim {FEAT_DIM}", flush=True)
TIE_DIM = FEAT_DIM
LS = float(E("LABEL_SMOOTH", "0.0"))              # intermediate aux heads
TERM_LS = float(E("TERM_LS", str(LS)))            # terminal (defaults to LS)
WD = float(E("WD", "5e-4"))
NUM_CLASSES = {"tiny": 200, "c100": 100, "c10": 10}[DATASET]
IMG = 64 if DATASET == "tiny" else 32
# mean/std as in vision_ladder_r32.py
_STATS = {"tiny": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
          "c100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
          "c10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))}
MEAN, STD = _STATS[DATASET]

_TERM = {}          # terminal readout registry: 'fc' -> nn.Linear, 'rand'/'rand_std' -> random arm, 'stale' -> frozen readout


# ---------------------------------------------------------------- data (as in vision_ladder_r32.py)
class TinyValSet(Dataset):
    """val/images + val_annotations.txt -> (img, label), reusing the class_to_idx of train."""

    def __init__(self, root, class_to_idx, transform):
        self.dir = os.path.join(root, "val", "images")
        self.transform = transform
        self.samples = []
        with open(os.path.join(root, "val", "val_annotations.txt")) as f:
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 2 and p[1] in class_to_idx:
                    self.samples.append((p[0], class_to_idx[p[1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fn, y = self.samples[i]
        img = Image.open(os.path.join(self.dir, fn)).convert("RGB")
        return self.transform(img), y


def build_loaders(data_root, batch_size, autoaug, workers):
    extra = []
    if autoaug:
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy
            pol = AutoAugmentPolicy.IMAGENET if DATASET == "tiny" else AutoAugmentPolicy.CIFAR10
            extra = [AutoAugment(pol)]
            print(f"  [aug] AutoAugment({pol.value if hasattr(pol, 'value') else pol}) ON")
        except Exception as e:                                    # noqa: BLE001
            print(f"  [aug] AutoAugment unavailable ({e}); crop+flip only")
    pad = 8 if DATASET == "tiny" else 4
    train_tf = transforms.Compose([
        transforms.RandomCrop(IMG, padding=pad), transforms.RandomHorizontalFlip(),
        *extra, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    if DATASET == "tiny":
        root = os.path.join(data_root, "tiny-imagenet-200")
        assert os.path.isdir(root), f"not found: {root} (set DATA_ROOT to the directory containing tiny-imagenet-200)"
        train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        test_set = TinyValSet(root, train_set.class_to_idx, test_tf)
        n_cls = len(train_set.classes)
    else:
        ds_cls = datasets.CIFAR100 if DATASET == "c100" else datasets.CIFAR10
        train_set = ds_cls(data_root, train=True, download=True, transform=train_tf)
        test_set = ds_cls(data_root, train=False, download=True, transform=test_tf)
        n_cls = len(train_set.classes)
    assert n_cls == NUM_CLASSES, f"class count mismatch: {n_cls} vs {NUM_CLASSES}"
    print(f"  {DATASET}: train={len(train_set)} test={len(test_set)} classes={n_cls} img={IMG}")
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True,
                       num_workers=workers, drop_last=True, pin_memory=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True))


# ---------------------------------------------------------------- model
def conv3x3(i, o, s=1):
    return nn.Conv2d(i, o, 3, s, 1, bias=False)


class Bottleneck(nn.Module):
    """ResNet-50 bottleneck (CIFAR/tiny stem, as in solo_cifar_ablate.py)."""
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
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = F.relu(self.bn2(self.conv2(h)), inplace=True)
        h = self.bn3(self.conv3(h))
        return F.relu(h + identity, inplace=True)


class TiedOut(nn.Module):
    """Readout = detached read-only copy of the terminal classifier weight. Zero parameters; no gradient flows back to the terminal."""

    def forward(self, x):
        if TIE_MODE == "random":
            w = _TERM.get("rand")
            if w is None:
                raise RuntimeError("[tied] random readout not registered")
            fc = _TERM.get("fc")
            if fc is not None and TIE_SCALE:
                w = w * (fc.weight.detach().std() / _TERM["rand_std"])
        elif TIE_MODE == "stale":
            w = _TERM.get("stale")
            if w is None:
                raise RuntimeError("[tied] stale readout not registered")
        else:
            fc = _TERM.get("fc")
            if fc is None:
                raise RuntimeError("[tied] terminal fc not registered")
            w = fc.weight
        if x.shape[-1] != w.shape[1]:
            raise RuntimeError(f"[tied] dim mismatch: head projects to {x.shape[-1]}, terminal expects {w.shape[1]}")
        return F.linear(x, w.detach())


class Head(nn.Module):
    """Auxiliary head; the final C x TIE_DIM readout is either shared (TIE=1) or private and trainable (TIE=0).
    Channel rule as ReadoutHead in solo_cifar_ablate.py (R32 and R50 branches)."""

    def __init__(self, inplanes, num_classes, hidden=128, widen=1.0, depth=1, strided=False):
        super().__init__()
        if inplanes <= 64:                       # R32 channels (16/32/64)
            mid = int((32 if inplanes == 16 else 64) * widen)
            stride = 2 if inplanes in (16, 32) else 1
        else:                                    # R50 channels (256/512/1024/2048)
            mid = int(min(inplanes // 2, 256) * widen)
            stride = 2 if inplanes <= 512 else 1
        layers = [nn.Conv2d(inplanes, mid, 3, stride, 1, bias=False),
                  nn.BatchNorm2d(mid), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(mid, mid, 3, 2 if strided else 1, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU()]
        layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                   nn.Linear(mid, int(hidden * widen)), nn.ReLU(inplace=True),
                   nn.Linear(int(hidden * widen), TIE_DIM, bias=False)]   # project to readout input dim = 2048
        layers.append(TiedOut() if TIE else nn.Linear(TIE_DIM, num_classes, bias=False))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class R50Net(nn.Module):
    """ResNet-50 (16 Bottlenecks) split into K gradient-isolated modules, each with a local CE. Terminal feature d=2048."""

    def __init__(self, num_classes=NUM_CLASSES, lr=0.1, K=16, t_max=100):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList()
        block_ch, in_ch = [], 64
        for planes, n, first_stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
            for b in range(n):
                self.blocks.append(Bottleneck(in_ch, planes, first_stride if b == 0 else 1))
                in_ch = planes * Bottleneck.expansion
                block_ch.append(in_ch)
        self.fc = nn.Linear(FEAT_DIM, num_classes)      # 2048 -> C
        self.n_blocks = len(self.blocks)                # 16

        K = max(2, min(K, self.n_blocks))
        self.groups = [list(g) for g in np.array_split(np.arange(self.n_blocks), K)]
        self.K = K
        self._init_weights()          # backbone init before heads are built

        self.heads = nn.ModuleList([
            Head(block_ch[self.groups[k][-1]], num_classes, hidden=HEAD_HIDDEN,
                 widen=HEAD_WIDEN, depth=HEAD_DEPTH, strided=STRIDED_HEAD)
            for k in range(K - 1)])

        _TERM["fc"] = self.fc
        if TIE and TIE_MODE == "random":                # frozen random readout: same shape and std, fixed seed
            g = torch.Generator().manual_seed(1234)
            w = torch.randn(self.fc.weight.shape, generator=g) * self.fc.weight.std().item()
            _TERM["rand"] = w
            _TERM["rand_std"] = w.std()
        if TIE and TIE_MODE == "stale":                 # freeze a copy of a trained terminal readout
            assert STALE_FC, "TIE_MODE=stale requires STALE_FC=<ckpt saved by this script>"
            ck = torch.load(STALE_FC, map_location="cpu")
            sd = ck.get("model", ck)
            w = sd.get("fc.weight")
            assert w is not None and tuple(w.shape) == tuple(self.fc.weight.shape), \
                f"STALE_FC lacks fc.weight or has wrong shape {None if w is None else tuple(w.shape)} vs {tuple(self.fc.weight.shape)}"
            _TERM["stale"] = w.float().clone()
            print(f"[stale] readout loaded from {STALE_FC} (std {float(w.float().std()):.4f}); terminal fc still trained from zero", flush=True)

        self.module_params = []
        for k in range(K):
            params = list(self.stem.parameters()) if k == 0 else []
            for bi in self.groups[k]:
                params += list(self.blocks[bi].parameters())
            params += list(self.heads[k].parameters()) if k < K - 1 else list(self.fc.parameters())
            self.module_params.append(params)
        self.opts = [SGD(p, lr=lr, momentum=0.9, nesterov=True, weight_decay=WD)
                     for p in self.module_params]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opts[0], T_max=t_max, eta_min=0.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def to(self, *a, **kw):
        out = super().to(*a, **kw)
        if "rand" in _TERM:
            _TERM["rand"] = _TERM["rand"].to(self.fc.weight.device)
            _TERM["rand_std"] = _TERM["rand_std"].to(self.fc.weight.device)
        if "stale" in _TERM:
            _TERM["stale"] = _TERM["stale"].to(self.fc.weight.device)
        return out

    def run_group(self, k, x_in, bn_eval=False):
        mods = ([self.stem] if k == 0 else []) + [self.blocks[bi] for bi in self.groups[k]]
        bns = [m for top in mods for m in top.modules() if isinstance(m, nn.BatchNorm2d)]
        if bn_eval:
            for bn in bns:
                bn.eval()
        h = x_in
        for m in mods:
            h = m(h)
        if bn_eval:
            for bn in bns:
                bn.train()
        return h

    def pool(self, h):
        return F.adaptive_avg_pool2d(h, 1).flatten(1)

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(self.pool(h))

    def forward_all(self, x):
        """Terminal logits + logits of each intermediate exit (for evaluation)."""
        outs, h = [], x
        for k in range(self.K):
            h = self.run_group(k, h)
            if k < self.K - 1:
                outs.append(self.heads[k](h))
        return self.fc(self.pool(h)), outs

    def step(self, img, y):
        """One minibatch: gradient-isolated per-module local updates. Returns (terminal logits, terminal loss)."""
        if MODE == "bp":                                   # end-to-end baseline (single loss, uses TERM_LS)
            for opt in self.opts:
                opt.zero_grad()
            logits = self.forward(img)
            loss = F.cross_entropy(logits, y, label_smoothing=TERM_LS)
            loss.backward()
            for opt in self.opts:
                opt.step()
            return logits, loss.item()

        gs = GS_MODE == "gs"
        for opt in self.opts:
            opt.zero_grad()
        x = img
        for k in range(self.K - 1):                        # intermediate modules: local CE (intermediate smoothing)
            x_in = x.detach()                              # gradient isolation: no gradient across modules
            x = self.run_group(k, x_in)
            logits = self.heads[k](x)
            loss = F.cross_entropy(logits, y, label_smoothing=LS)
            loss.backward()
            if gs:                                         # GS: recompute after this module's update, then pass downstream
                self.opts[k].step()
                with torch.no_grad():
                    x = self.run_group(k, x_in, bn_eval=True)
            else:
                x = x.detach()
        x_in = x.detach()
        x = self.run_group(self.K - 1, x_in)                # terminal: uses TERM_LS
        logits = self.fc(self.pool(x))
        loss_head = F.cross_entropy(logits, y, label_smoothing=TERM_LS)
        with torch.no_grad():                          # logging: terminal train-side 1 - sum q^2
            q = F.softmax(logits.detach(), -1)
            tr = getattr(self, "_tr", None)
            if tr is None:
                tr = self._tr = {"gini": 0.0, "n": 0}
            tr["gini"] += float((1 - q.pow(2).sum(-1)).mean())
            tr["n"] += 1
        loss_head.backward()
        if gs:
            self.opts[self.K - 1].step()
        else:
            for opt in self.opts:
                opt.step()
        return logits, loss_head.item()

    def update_lr(self):
        lr = self.scheduler.get_last_lr()[0]
        for opt in self.opts:
            for g in opt.param_groups:
                g["lr"] = lr


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (terminal acc, ensemble acc or None, {te_gini, exits}); as in vision_ladder_r32.py."""
    model.eval()
    correct = correct_ens = 0
    use_ens = MODE != "bp"
    exit_c = [0] * (model.K - 1 if use_ens else 0)
    gini, nb = 0.0, 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if use_ens:
            fin, outs = model.forward_all(data)
            correct += fin.argmax(1).eq(target).sum().item()
            q = F.softmax(fin, -1)
            gini += float((1 - q.pow(2).sum(-1)).mean())
            nb += 1
            lp = F.log_softmax(fin, -1)
            for j, z in enumerate(outs):
                exit_c[j] += z.argmax(1).eq(target).sum().item()
                lp = lp + F.log_softmax(z, -1)
            correct_ens += lp.argmax(1).eq(target).sum().item()
        else:
            correct += model(data).argmax(1).eq(target).sum().item()
    model.train()
    n = len(loader.dataset)
    med = {"te_gini": gini / max(nb, 1), "exits": [c / n for c in exit_c]}
    return correct / n, (correct_ens / n if use_ens else None), med


# ---------------------------------------------------------------- self-check
def dryrun():
    print(f"[dry] R50 DATASET={DATASET}(C={NUM_CLASSES}, img={IMG}, controlled_subspace={min(NUM_CLASSES - 1, TIE_DIM) / TIE_DIM:.1%}) "
          f"MODE={MODE} TIE={int(TIE)}({TIE_MODE}) TIE_DIM={TIE_DIM} "
          f"K={K_MODULES} hd={HEAD_DEPTH} widen={HEAD_WIDEN} LS={LS} TERM_LS={TERM_LS}")
    model = R50Net(K=K_MODULES, t_max=1).to(DEVICE)
    x = torch.randn(2, 3, IMG, IMG, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (2,), device=DEVICE)
    fin, outs = model.forward_all(x)
    print(f"[dry] final={tuple(fin.shape)} exits={len(outs)}×{tuple(outs[0].shape)} "
          f"blocks={model.n_blocks} groups={[len(g) for g in model.groups]}")
    print(f"[dry] terminal readout {tuple(model.fc.weight.shape)} ({model.fc.weight.numel():,} params)")
    _, loss = model.step(x, y)
    tot = sum(p.numel() for p in model.parameters())
    inopt = sum(p.numel() for grp in model.module_params for p in grp)
    n_head = sum(p.numel() for p in model.heads.parameters())
    print(f"[dry] one step passed, terminal loss={loss:.3f}")
    print(f"[dry] total params {tot:,}; in optimizer {inopt:,}; aux heads {n_head:,}; "
          f"{'consistent OK' if tot == inopt else 'MISMATCH - some params are not optimized!'}")
    print("[dry] note: the tie vs untie param difference should be exactly (K-1)×C×TIE_DIM = "
          f"{(model.K - 1) * NUM_CLASSES * TIE_DIM:,}")
    sys.exit(0)


# ---------------------------------------------------------------- training
def main():
    from tqdm import tqdm
    epochs = int(E("EPOCHS", "100"))
    batch = int(E("BATCH", "128"))
    lr = float(E("LR", "0.1"))
    t_max = int(E("T_MAX", str(epochs)))
    print(f"device={DEVICE} | {MODE.upper()} R50 {DATASET.upper()}(C={NUM_CLASSES}) K={K_MODULES}(+BP head) "
          f"gs={GS_MODE} tie={int(TIE)}({TIE_MODE}) tie_dim={TIE_DIM} "
          f"ls={LS} term_ls={TERM_LS} bs={batch} wd={WD} seed={SEED}", flush=True)
    train_loader, val_loader = build_loaders(
        E("DATA_ROOT", "./data"), batch, E("AUTOAUG", "1") == "1", int(E("NUM_WORKERS", "4")))
    model = R50Net(K=K_MODULES, lr=lr, t_max=t_max).to(DEVICE)
    n_head = sum(p.numel() for p in model.heads.parameters())
    print(f"  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"(aux_heads={n_head / 1e6:.2f}M) terminal_readout={model.fc.weight.numel():,}",
          flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    best, best_ens = 0.0, 0.0
    ckpt = E("CKPT", "")
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, nb = 0.0, 0
        for data, target in tqdm(train_loader, desc=f"ep{ep}/{epochs}", ncols=80, leave=False):
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            _, loss = model.step(data, target)
            tot_loss += loss
            nb += 1
        model.scheduler.step()
        model.update_lr()
        acc, ens, med = evaluate(model, val_loader, DEVICE)
        best = max(best, acc)
        if ens is not None:
            best_ens = max(best_ens, ens)
        mem = torch.cuda.max_memory_allocated(DEVICE) / 2**30 if torch.cuda.is_available() else 0
        print(f"[epoch {ep:3d}/{epochs}] loss={tot_loss / max(nb, 1):.3f} acc={acc:.4f} "
              f"best={best:.4f}" + (f" ens={ens:.4f} best_ens={best_ens:.4f}" if ens else "")
              + f" lr={model.scheduler.get_last_lr()[0]:.4f} mem={mem:.2f}GB", flush=True)
        tr = getattr(model, "_tr", None)
        tr_gini = tr["gini"] / max(tr["n"], 1) if tr else float("nan")
        model._tr = None
        ex = med["exits"]
        print(f"[med   {ep:3d}] tr_gini={tr_gini:.4f} te_gini={med['te_gini']:.4f} "
              f"fc_norm={model.fc.weight.norm().item():.3f}"
              + (f" best_exit={max(ex):.4f}@{ex.index(max(ex))}"
                 f" exits=[{','.join(f'{a:.3f}' for a in ex)}]" if ex else ""), flush=True)
        if ckpt:
            blob = {"model": model.state_dict(), "epoch": ep, "best": best,
                    "cfg": {k: os.environ.get(k, "") for k in (
                        "DATASET", "MODE", "SEED", "TIE", "TIE_MODE", "LABEL_SMOOTH",
                        "TERM_LS", "BATCH", "LR", "WD", "EPOCHS", "K_MODULES",
                        "HEAD_DEPTH", "HEAD_WIDEN", "GS_MODE", "AUTOAUG")}}
            torch.save(blob, ckpt)
            if ep in (10, 25, 50):
                torch.save(blob, f"{ckpt}.ep{ep}")
    print(f"FINAL best test acc: {best:.4f}  (data={DATASET}, mode={MODE}, K={model.K}, tie={int(TIE)}"
          f"({TIE_MODE}), ls={LS}, term_ls={TERM_LS})"
          + (f"  best_ens={best_ens:.4f}" if MODE != "bp" else ""), flush=True)


if __name__ == "__main__":
    if E("DRYRUN", "0") == "1":
        dryrun()
    main()
