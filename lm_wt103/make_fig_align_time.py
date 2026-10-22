"""make_fig_align_time.py - figures and table from the lm_solo.py alignment probe logs (PROBE_EVERY; *.align.jsonl).

  python make_fig_align_time.py [logdir] [out_prefix]   # aggregate (stdlib only) -> <out_prefix>_data.json + table;
                                                       #   also draws figures if matplotlib is importable
  python make_fig_align_time.py --from X_data.json      # plot from a previously written aggregate file
Defaults: logdir=logs_align, out_prefix=<logdir>/fig_align.
Figures: <prefix>_time.{pdf,png}: one panel per boundary k, x = optimizer step, y = cos(g_hat_k, g_k^BP) on the
fixed probe batch, one curve per arm, line = mean over seeds, band = min..max.
<prefix>_terms.png: |E_read|, |E_resid|, |E_path| relative to |g_hat - g_bp| (mean over seeds).
Input file names: [wt103al_]<arm>_s<seed>.align.jsonl (as written by run_wt103_align_ladder.sh).
"""
import glob, json, os, re, sys

LABEL = {"rand": "RAND", "priv": "PRIV", "solo": "SOLO", "conv_loc": "CONVERGED (loc)", "conv_bp": "CONVERGED (BP)"}
COLOR = {"rand": "#7f7f7f", "priv": "#1f77b4", "solo": "#d62728", "conv_loc": "#2ca02c", "conv_bp": "#9467bd"}
ORDER = ["rand", "priv", "solo", "conv_loc", "conv_bp"]
KEYS = ("cos", "cos_pos", "diff", "e_read", "e_resid", "e_path", "g_hat", "g_bp")


def load_runs(logdir):
    runs = {}  # arm -> seed -> [records]
    for f in sorted(glob.glob(os.path.join(logdir, "*.align.jsonl"))):
        m = re.search(r"(?:wt103al_)?(.+)_s(\d+)\.align\.jsonl$", os.path.basename(f))
        if not m:
            continue
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if recs:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = recs
    if not runs:
        sys.exit(f"no *.align.jsonl under {logdir}")
    return runs


def aggregate(runs):
    """-> {arm: {k: {key: {steps, mean, min, max, n}}, "last": {...}}} using stdlib only."""
    agg = {}
    for arm, seeds in runs.items():
        nk = max(len(r["rows"]) for recs in seeds.values() for r in recs)
        A = {"nk": nk, "seeds": sorted(seeds), "k": {}}
        for k in range(nk):
            per = []  # per seed: {step: row}
            for recs in seeds.values():
                per.append({r["step"]: r["rows"][k] for r in recs if len(r["rows"]) > k})
            common = sorted(set.intersection(*[set(d) for d in per])) if per else []
            Ak = {}
            for key in KEYS:
                cols = [[d[t][key] for d in per] for t in common]
                Ak[key] = dict(steps=common, mean=[sum(c) / len(c) for c in cols], min=[min(c) for c in cols],
                               max=[max(c) for c in cols], n=len(per))
            last = [d[max(d)][ "cos"] for d in per if d]
            mu = sum(last) / len(last) if last else float("nan")
            sd = (sum((v - mu) ** 2 for v in last) / len(last)) ** 0.5 if last else float("nan")
            Ak["last_cos"] = dict(mean=mu, sd=sd, n=len(last), values=last)
            A["k"][str(k)] = Ak
        agg[arm] = A
    return agg


def print_table(agg):
    arms = [a for a in ORDER if a in agg] + sorted(a for a in agg if a not in ORDER)
    nk = max(A["nk"] for A in agg.values())
    print(f"{'arm':18s} " + " ".join(f"k={k}: cos(last) mean±sd [n]  " for k in range(nk)))
    for a in arms:
        cells = []
        for k in range(nk):
            L = agg[a]["k"].get(str(k), {}).get("last_cos")
            cells.append((f"{L['mean']:.3f}±{L['sd']:.3f} [{L['n']}]" if L and L["n"] else "-").ljust(27))
        print(f"{LABEL.get(a, a):18s} " + " ".join(cells))
    return arms, nk


def draw(agg, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[no figure] matplotlib unavailable here ({type(e).__name__}: {e}).")
        print(f"  -> copy {out}_data.json (or the whole logdir/*.align.jsonl) to a machine with matplotlib and run:")
        print(f"     python make_fig_align_time.py --from {os.path.basename(out)}_data.json")
        return
    arms, nk = [a for a in ORDER if a in agg] + sorted(a for a in agg if a not in ORDER), max(A["nk"] for A in agg.values())
    fig, axes = plt.subplots(1, nk, figsize=(4.2 * nk, 3.4), sharey=True, squeeze=False)
    for k in range(nk):
        ax = axes[0, k]
        for a in arms:
            S = agg[a]["k"].get(str(k), {}).get("cos")
            if not S or not S["steps"]:
                continue
            ax.plot(S["steps"], S["mean"], color=COLOR.get(a), lw=1.8, label=f"{LABEL.get(a, a)} (n={S['n']})")
            if S["n"] > 1:
                ax.fill_between(S["steps"], S["min"], S["max"], color=COLOR.get(a), alpha=0.18, lw=0)
        ax.set_title(f"boundary $k={k}$"); ax.set_xlabel("optimizer step"); ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel(r"$\cos(\hat g_k,\, g_k^{\mathrm{BP}})$")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(r"$\cos(\hat g_k, g_k^{\mathrm{BP}})$ on a fixed probe batch; band = min–max over seeds", fontsize=10)
    fig.tight_layout(); fig.savefig(out + "_time.pdf"); fig.savefig(out + "_time.png", dpi=160)
    fig2, axes2 = plt.subplots(len(arms), nk, figsize=(4.2 * nk, 2.6 * len(arms)), sharex=True, squeeze=False)
    for i, a in enumerate(arms):
        for k in range(nk):
            ax = axes2[i, k]; Ak = agg[a]["k"].get(str(k))
            if Ak and Ak["diff"]["steps"]:
                for key, ls, lab in (("e_read", "-", r"$|\mathcal{E}_{\mathrm{read}}|$"), ("e_resid", "--", r"$|\mathcal{E}_{\mathrm{resid}}|$"),
                                     ("e_path", ":", r"$|\mathcal{E}_{\mathrm{path}}|$")):
                    y = [v / max(d, 1e-30) for v, d in zip(Ak[key]["mean"], Ak["diff"]["mean"])]
                    ax.plot(Ak[key]["steps"], y, ls=ls, color=COLOR.get(a), lw=1.5, label=lab)
            ax.set_title(f"{LABEL.get(a, a)}, k={k}", fontsize=9); ax.grid(alpha=0.3)
            if i == len(arms) - 1:
                ax.set_xlabel("optimizer step")
            if k == 0:
                ax.set_ylabel(r"$|\mathcal{E}|\,/\,|\hat g-g^{\mathrm{BP}}|$")
    axes2[0, 0].legend(fontsize=8)
    fig2.tight_layout(); fig2.savefig(out + "_terms.png", dpi=140)
    print(f"figures: {out}_time.pdf/.png  {out}_terms.png")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--from"]:
        src = args[1]; agg = json.load(open(src))
        out = args[2] if len(args) > 2 else re.sub(r"_data\.json$", "", src)
    else:
        logdir = args[0] if args else "logs_align"
        out = args[1] if len(args) > 1 else os.path.join(logdir, "fig_align")
        agg = aggregate(load_runs(logdir))
        json.dump(agg, open(out + "_data.json", "w"))
        print(f"aggregate written: {out}_data.json ({os.path.getsize(out + '_data.json') // 1024} KB)")
    print_table(agg)
    draw(agg, out)
