"""summarize_pipep.py - summarize the PIPEP records of pipep.py logs (one or more directories) into a table.
Usage: python summarize_pipep.py logs_pipep_p2 logs_pipep_p6 [--ranks] [--mlt]
  --ranks  also print per-rank (rank 0..p-1) peak and activation memory of every arm.
  --mlt    table for the fixed global-batch sweep of run_pipep_mlt.sh (see below).

Default table, one row per (p, M, method); tok/s = slowest rank; ratios > 1 mean faster:
  vs BP+PP     tok/s / tok/s of bp at the same (p, M), for comparison across parallelisms;
  vs same-par  tok/s / tok/s of the BP arm with the same parallelism (bp for PP and hybrid arms, bp_dp for DP arms).
  peak GB, act GB: rank with the highest peak; peak avg, act avg: mean over the p ranks; act = peak - state - grad.
  comm MB/it: analytic communication per iteration, summed over all ranks.

--mlt table, per (p, G) and m (microbatches per replica): R = replicas, S = stages per replica, M = m*R, MB = G/M.
  /bestBP = tok/s / max tok/s of all BP arms (any H) at the same (p, G, m); * marks that BP arm.
  BP arms do not depend on H and are merged (fastest measurement kept); SOLO arms get one row per H.
"""
import glob
import json
import os
import sys

NAMES = {"bp": "BP + PP", "bp_fw": "BP + PP fw", "bp_il_v2": "BP + PP il2", "bp_il_v4": "BP + PP il4",
         "bp_zb_v2": "BP + PP zb2", "bp_zb_v4": "BP + PP zb4",
         "solo": "SOLO + PP", "solo@10": "SOLO@10", "solo@1": "SOLO@1",
         "bp_dp": "BP + DP", "solo_dp": "SOLO + DP",
         "solo_kr2x4": "SOLO 2x4", "solo_kr4x2": "SOLO 4x2", "solo_kr2x2": "SOLO 2x2",
         "bp_pd4": "BP PP4xDP2", "bp_pd2": "BP PP2xDP4"}
ORDER = ["bp", "bp_fw", "bp_il_v2", "bp_il_v4", "bp_zb_v2", "bp_zb_v4", "bp_pd4", "bp_pd2", "bp_dp",
         "solo", "solo@10", "solo@1", "solo_kr2x4", "solo_kr4x2", "solo_kr2x2", "solo_dp"]

SHOW_RANKS = "--ranks" in sys.argv
MLT = "--mlt" in sys.argv
recs = []
for d in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for f in sorted(glob.glob(os.path.join(d, "*.r*.log"))):
        for line in open(f, errors="ignore"):
            if line.startswith("PIPEP "):
                recs.append(json.loads(line[6:]))
if not recs:
    sys.exit("no PIPEP records found")


def lab_of(r):
    # bp_il / bp_zb with v=2 and v=4 share a label but come from different directories: keep them apart by v
    return r["label"] + (f"_v{r['v']}" if r["arm"] in ("bp_il", "bp_zb") else "")


def cell_of(rs):
    """One cell (the p rank records of one p/M/arm) -> summary: tok/s of the slowest rank, memory of the highest-peak rank."""
    p = rs[0]["p"]
    top = max(rs, key=lambda x: x["peak_gb"])
    rs = sorted(rs, key=lambda x: x["rank"])
    return dict(tok=min(x["tok_per_s"] for x in rs), peak=top["peak_gb"], act=top["act_gb_est"],
                peak_mean=sum(x["peak_gb"] for x in rs) / p, act_mean=sum(x["act_gb_est"] for x in rs) / p,
                peak_ranks=[x["peak_gb"] for x in rs], act_ranks=[x["act_gb_est"] for x in rs],
                comm=rs[0]["mb_per_iter_analytic"], p2p=rs[0]["mb_p2p_analytic"],
                bcast=rs[0]["mb_bcast_analytic"], ar=rs[0].get("mb_allreduce_analytic", 0.0),
                rows=rs[0]["rows_per_iter"], par=rs[0].get("parallel", "pp"),
                arm=rs[0]["arm"], H=rs[0]["H"], MB=rs[0]["MB"], M=rs[0]["M"],
                m=rs[0].get("M_local", rs[0]["M"]), R=rs[0].get("replicas", 1),
                stages=rs[0].get("stages", p))


def order_key(l):
    return (ORDER.index(l) if l in ORDER else len(ORDER), l)


if MLT:
    # Fixed global-batch table. Cell key = (p, G, m, arm, H); BP arms do not depend on H and are merged as H=0
    # (the fastest of repeated measurements is kept, the strictest baseline for SOLO).
    groups = {}
    for r in recs:
        lab = lab_of(r)
        is_bp = r["arm"].startswith("bp")
        key = (r["p"], r["rows_per_iter"], r.get("M_local", r["M"]), lab, 0 if is_bp else r["H"],
               r["M"], r["MB"], r["v"])
        groups.setdefault(key, []).append(r)
    cells = {}
    for key, rs in groups.items():
        p = key[0]
        by_rank = {}
        for r in rs:                      # same cell run several times (BP_EVERY_H=1): keep the last record per rank
            by_rank[r["rank"]] = r
        if len(by_rank) != p:
            continue
        c = cell_of(list(by_rank.values()))
        k2 = key[:5]
        if k2 in cells and cells[k2]["tok"] >= c["tok"]:
            continue
        cells[k2] = c
    print("Fixed global-batch sweep: G rows per step, m microbatches per replica, microbatch MB = G/(m*R). tok/s = slowest rank (tied solo")
    print("readout broadcast amortized over TIE_SYNC). '/bestBP' = tok/s of this arm / max tok/s of all BP arms (any H) at the same (p, G, m);")
    print("for a BP arm this is its ratio to the best BP, so the 1.000 row is the best BP. peak GB = highest-peak rank; comm = analytic per step")
    print("(P2P activations + amortized readout broadcast + gradient all-reduce, summed over all ranks).")
    print()
    for p, G in sorted({(k[0], k[1]) for k in cells}):
        print(f"p={p}  G={G} rows/step")
        print(f"{'m':>2} {'arm':<12} {'H':>2} {'R':>2} {'S':>2} {'M':>3} {'MB':>3} {'tok/s':>8} {'/bestBP':>8} "
              f"{'peak GB':>7} {'act GB':>7} {'P2P MB':>8} {'bcastMB':>7} {'AllRed MB':>10} {'total MB':>9}")
        for m in sorted({k[2] for k in cells if k[0] == p and k[1] == G}):
            here = {k: c for k, c in cells.items() if k[0] == p and k[1] == G and k[2] == m}
            if len({c["rows"] for c in here.values()}) > 1:
                print(f"{m:>2}  [dropped] rows per step differ across arms {sorted({c['rows'] for c in here.values()})}")
                continue
            bps = {k: c for k, c in here.items() if c["arm"].startswith("bp")}
            best = max(bps.values(), key=lambda c: c["tok"]) if bps else None
            best_lab = next((k[3] for k, c in bps.items() if c is best), None)
            for k, c in sorted(here.items(), key=lambda kc: (order_key(kc[0][3]), kc[0][4])):
                ratio = f"{c['tok'] / best['tok']:.3f}" if best else "-"
                h = "-" if c["arm"].startswith("bp") else str(c["H"])
                mark = "*" if k[3] == best_lab else " "
                print(f"{m:>2} {NAMES.get(k[3], k[3]):<11}{mark} {h:>1} {c['R']:>2} {c['stages']:>2} {c['M']:>3} "
                      f"{c['MB']:>3} {c['tok']:>8} {ratio:>8} {c['peak']:>7.2f} {c['act']:>7.2f} "
                      f"{c['p2p']:>8.1f} {c['bcast']:>7.2f} {c['ar']:>10.1f} {c['comm']:>9.1f}")
            if best is None:
                print(f"{'':>3}(no BP arm fully reported in this cell, ratios missing)")
            print()
    print("Legend: '/bestBP' = arm tok/s / best BP tok/s at that (G, m); * = best BP arm.")
    sys.exit(0)

groups = {}
for r in recs:
    groups.setdefault((r["p"], r["M"], lab_of(r)), []).append(r)

rows = {}
for (p, m, lab), rs in groups.items():
    if len(rs) != p:
        continue                                   # some rank did not report: incomplete cell
    rows[(p, m, lab)] = cell_of(rs)

labels = [l for l in ORDER if any(k[2] == l for k in rows)] + \
         sorted({k[2] for k in rows} - set(ORDER))
print(f"{'p':>2} {'M':>3}  {'method':<10} {'tok/s':>8} {'vs BP+PP':>11} {'vs same-par':>13} "
      f"{'peak GB':>7} {'peak avg':>8} {'act GB':>7} {'act avg':>8} {'comm MB/it':>10}")
for p, m in sorted({(k[0], k[1]) for k in rows}):
    cell = {l: rows[(p, m, l)] for l in labels if (p, m, l) in rows}
    if len({c["rows"] for c in cell.values()}) > 1:
        print(f"{p:>2} {m:>3}  [dropped] rows per step differ across arms")
        continue
    base = cell.get("bp")
    for l, c in cell.items():
        ratio = f"{c['tok'] / base['tok']:.3f}" if base else "-"
        same = cell.get("bp_dp" if c["par"] == "dp" else "bp")
        ratio_same = f"{c['tok'] / same['tok']:.3f}" if same else "-"
        print(f"{p:>2} {m:>3}  {NAMES.get(l, l):<10} {c['tok']:>8} {ratio:>11} {ratio_same:>13} "
              f"{c['peak']:>7.2f} {c['peak_mean']:>8.2f} {c['act']:>7.2f} {c['act_mean']:>8.2f} "
              f"{c['comm']:>10.1f}")
        if SHOW_RANKS:
            print(f"{'':>8}per-rank peak {c['peak_ranks']}")
            print(f"{'':>8}per-rank act {c['act_ranks']}")
    print()
print("vs BP+PP > 1 means faster than BP+PP; vs same-par is relative to BP+PP for PP arms, BP+DP for DP arms (BP+PP for hybrid arms).")
print("peak GB / act GB: GPU with the highest peak; avg = mean over the p GPUs; act = peak - state - grad, an estimate.")
print("comm: PP = inter-stage activations (SOLO adds the readout broadcast, amortized over TIE_SYNC), DP = gradient all-reduce; hybrid arms have all three.")
