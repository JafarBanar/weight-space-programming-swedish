"""Turn the cached results into tables, figures and example files.
  python src/analyze.py   ->  results/summary.{json,md}, results/examples.md, figures/*.png
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "figures"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b1"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})

LABELS = {"B0_baseline": "B0  Unedited model", "P1_prompt": "P1  Prompt: \"Svara alltid på svenska.\"",
          "W0_weights_only": "W0  Weights-only edit (2 neurons, 0 forward passes)", "W1_tuned": "W1  Feedback-tuned edit",
          "S1_steering": "S1  Activation steering", "F1_lora": "F1  LoRA fine-tune (self-distilled)",
          "R0_random": "R0  Random 2 neurons, same layers (5 seeds)", "R1_random": "R1  Random neurons matched to W1 (5 seeds)",
          "W0a_only_L23N981": "W0a  Delete only L23/N981 (post-hoc)", "W0b_only_L22N1821": "W0b  Delete only L22/N1821 (post-hoc)"}
SHORT = {"B0_baseline": "B0  Unedited", "R0_random": "R0  Random 2 neurons", "W0_weights_only": "W0  Weights-only edit",
         "W0a_only_L23N981": "W0a  Only L23/N981 (post-hoc)", "W0b_only_L22N1821": "W0b  Only L22/N1821 (post-hoc)",
         "S1_steering": "S1  Activation steering", "F1_lora": "F1  LoRA fine-tune", "P1_prompt": "P1  Prompt instruction"}


def load(name):
    p = f"{R}/test_{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def hits(r):
    """16 x 20 matrix of Swedish-response indicators: rows = templates (8 dev + 8 test), cols = test topics."""
    return np.concatenate([np.array(r["sv_A"]["hit"]).reshape(8, 20), np.array(r["sv_B"]["hit"]).reshape(8, 20)])


def marker_hits(r):
    """Post-hoc robustness metric that does not use the detector: an answer counts as Swedish if
    more than half of its Swedish + Danish/Norwegian function words are Swedish (detector label
    is used only when the answer contains no such word)."""
    from evaluate import marker_share
    out = []
    for s_ in ("sv_A", "sv_B"):
        for o, h in zip(r[s_]["outputs"], r[s_]["hit"]):
            m = marker_share(o)
            out.append(h if m is None else int(m > 0.5))
    return np.array(out)


def boot(h, h0=None, n=2000, seed=0):
    """Bootstrap over templates (within each block of 8) and topics jointly. Returns 95% CI of the
    Swedish rate, or of the difference to h0 when given."""
    g = np.random.default_rng(seed); out = []
    for _ in range(n):
        rows = np.concatenate([g.integers(0, 8, 8), 8 + g.integers(0, 8, 8)]); cols = g.integers(0, 20, 20)
        v = h[np.ix_(rows, cols)].mean()
        out.append(v - h0[np.ix_(rows, cols)].mean() if h0 is not None else v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    os.makedirs(F, exist_ok=True)
    base = load("B0_baseline"); h0 = hits(base)
    rows = []
    for name in ["B0_baseline", "P1_prompt", "W0_weights_only", "W1_tuned", "S1_steering", "F1_lora", "W0a_only_L23N981", "W0b_only_L22N1821"]:
        r = load(name)
        if r is None:
            continue
        h = hits(r)
        rows.append({"id": name, "label": LABELS[name], "sv_rate": float(h.mean()), "sv_ci": boot(h),
                     "sv_A": r["sv_A"]["rate"], "sv_B": r["sv_B"]["rate"],
                     "diff_vs_B0": float(h.mean() - h0.mean()), "diff_ci": boot(h, h0),
                     "fixed": int(((h == 1) & (h0 == 0)).sum()), "broken": int(((h == 0) & (h0 == 1)).sum()),
                     "mcnemar_p": float(binomtest(int(((h == 1) & (h0 == 0)).sum()), int((h != h0).sum()), 0.5).pvalue) if (h != h0).any() else 1.0,
                     "marker_share": float(np.mean([r["sv_A"]["marker_share"], r["sv_B"]["marker_share"]])),
                     "sv_rate_markers": float(marker_hits(r).mean()),
                     "sv_rate_markers_A": float(marker_hits(r)[:160].mean()), "sv_rate_markers_B": float(marker_hits(r)[160:].mean()),
                     "marker_A": r["sv_A"]["marker_share"], "marker_B": r["sv_B"]["marker_share"],
                     "A_ci": [float(x) for x in np.percentile([h[:8][np.ix_(g_.integers(0, 8, 8), g_.integers(0, 20, 20))].mean() for g_ in [np.random.default_rng(1)] for _ in range(2000)], [2.5, 97.5])],
                     "B_ci": [float(x) for x in np.percentile([h[8:][np.ix_(g_.integers(0, 8, 8), g_.integers(0, 20, 20))].mean() for g_ in [np.random.default_rng(2)] for _ in range(2000)], [2.5, 97.5])],
                     "degenerate": float(np.mean([r["sv_A"]["degenerate"], r["sv_B"]["degenerate"]])),
                     "langs": {k: r["sv_A"]["langs"].get(k, 0) + r["sv_B"]["langs"].get(k, 0)
                               for k in set(r["sv_A"]["langs"]) | set(r["sv_B"]["langs"])},
                     "en_rate": r["en_B"]["rate"], "da_rate": r["da_B"]["rate"], "da_langs": r["da_B"]["langs"],
                     "ppl": r["ppl"], "mp": r["minimal_pairs"]["accuracy"], "mp_by": r["minimal_pairs"]["by_phenomenon"]})
    for fam in ["R0_random", "R1_random"]:
        rs = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/test_{fam}_seed*.json"))]
        if rs:
            rates = [float(hits(r).mean()) for r in rs]
            rows.append({"id": fam, "label": LABELS[fam], "sv_rate": float(np.mean(rates)), "seed_rates": rates,
                         "sv_A": float(np.mean([r["sv_A"]["rate"] for r in rs])), "sv_B": float(np.mean([r["sv_B"]["rate"] for r in rs])),
                         "A_ci": [min(r["sv_A"]["rate"] for r in rs), max(r["sv_A"]["rate"] for r in rs)],
                         "B_ci": [min(r["sv_B"]["rate"] for r in rs), max(r["sv_B"]["rate"] for r in rs)],
                         "marker_share": float(np.mean([(r["sv_A"]["marker_share"] + r["sv_B"]["marker_share"]) / 2 for r in rs])),
                         "sv_ci": [min(rates), max(rates)], "diff_vs_B0": float(np.mean(rates) - h0.mean()),
                         "ppl": {"sv": float(np.mean([r["ppl"]["sv"] for r in rs]))}})
    json.dump(rows, open(f"{R}/summary.json", "w"), indent=1, ensure_ascii=False)

    # ---- summary table (markdown) ----
    b = rows[0]
    L = ["| Condition | Swedish answers, held-out (n=320) | Δ vs B0 [95% CI] | fixed / broken | sv_A / sv_B | Answers in da / no / other | Swedish function-word share | English→English | Danish→Danish | PPL sv / en / da / no | Grammar pairs |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "en_rate" in r:
            L.append(f"| {r['label']} | {100*r['sv_rate']:.1f}% | {100*r['diff_vs_B0']:+.1f} [{100*r['diff_ci'][0]:+.1f}, {100*r['diff_ci'][1]:+.1f}] | "
                     f"{r['fixed']} / {r['broken']} | {100*r['sv_A']:.1f} / {100*r['sv_B']:.1f} | {r['langs'].get('da', 0)} / {r['langs'].get('no', 0)} / {320 - r['langs'].get('sv', 0) - r['langs'].get('da', 0) - r['langs'].get('no', 0)} | {100*r['marker_share']:.1f}% | {100*r['en_rate']:.1f}% | {100*r['da_rate']:.1f}% | "
                     f"{r['ppl']['sv']:.1f} / {r['ppl']['en']:.1f} / {r['ppl']['da']:.1f} / {r['ppl']['no']:.1f} | {100*r['mp']:.1f}% |")
        else:
            L.append(f"| {r['label']} | {100*r['sv_rate']:.1f}% (range {100*min(r['seed_rates']):.1f}–{100*max(r['seed_rates']):.1f}) | "
                     f"{100*r['diff_vs_B0']:+.1f} | – | {100*r['sv_A']:.1f} / {100*r['sv_B']:.1f} | – | {100*r['marker_share']:.1f}% | – | – | {r['ppl']['sv']:.1f} / – / – / – | – |")
    open(f"{R}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    # ---- post-hoc notice questions: correct fact in the requested language ----
    N = ["| Condition | Swedish notice → Swedish (fact / language / both) | English notice → Swedish | English notice → English |", "|---|---|---|---|"]
    notice = {}
    for name in ["B0_baseline", "P1_prompt", "W0_weights_only", "W0a_only_L23N981", "S1_steering", "F1_lora"]:
        pth = f"{R}/notices_{name}.json"
        if os.path.exists(pth):
            n = json.load(open(pth)); notice[name] = {v: {k: n[v][k] for k in ("fact", "language", "joint")} for v in n}
            N.append(f"| {SHORT[name]} | " + " | ".join(f"{100*n[v]['fact']:.0f} / {100*n[v]['language']:.0f} / {100*n[v]['joint']:.0f}" for v in ("sv_sv", "en_sv", "en_en")) + " |")
    rn = [json.load(open(p_)) for p_ in sorted(glob.glob(f"{R}/notices_R0_random_seed*.json"))]
    if rn:
        N.append("| R0  Random 2 neurons (3 seeds, mean) | " + " | ".join(
            f"{100*np.mean([n[v]['fact'] for n in rn]):.0f} / {100*np.mean([n[v]['language'] for n in rn]):.0f} / {100*np.mean([n[v]['joint'] for n in rn]):.0f}" for v in ("sv_sv", "en_sv", "en_en")) + " |")
    open(f"{R}/notices_summary.md", "w").write("\n".join(N) + "\n"); print("\n".join(N))
    json.dump(notice, open(f"{R}/notices_summary.json", "w"), indent=1)

    # ---- examples: prompts where the unedited model drifted ----
    ex = ["# Held-out examples where the unedited model did not answer in Swedish\n",
          "Greedy decoding, first 48 new tokens. Language labels from the lingua detector.\n"]
    from prompts import test_sets
    T = test_sets(); others = {n: load(n) for n in ["W0_weights_only", "W0a_only_L23N981", "P1_prompt", "F1_lora"]}
    k = 0
    for split in ["sv_B", "sv_A"]:
        for i, hit in enumerate(base[split]["hit"]):
            if hit == 0 and k < 14:
                k += 1
                ex.append(f"\n**{split} #{i}: {T[split][i]['sv']}**\n")
                ex.append(f"- B0: {base[split]['outputs'][i]!r}")
                for n, r in others.items():
                    if r is not None:
                        ex.append(f"- {n.split('_')[0]} [{'sv' if r[split]['hit'][i] else 'not sv'}]: {r[split]['outputs'][i]!r}")
    open(f"{R}/examples.md", "w").write("\n".join(ex) + "\n")

    # ---- Figure 1: what the weights show (zero forward passes) ----
    s1 = json.load(open(f"{R}/stage1_weight_reading.json"))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.hist(s1["z_flat_sample"], bins=80, color=GREY, log=True)
    for (l, j), z, dy in zip(s1["W0_neurons"], s1["W0_z"], (40, 8)):
        ax.plot([z], [1.2], marker="v", color=ORANGE, ms=8, clip_on=False)
        ax.annotate(f"layer {l}, neuron {j}\nz = {z:.1f}", (z, 1.5), xytext=(z, dy), ha="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_xlim(-12, 7); ax.set_ylim(0.8, None)
    ax.set_xlabel("How strongly a neuron writes towards Swedish (+) or Danish/Norwegian (−), z-score within its layer")
    ax.set_ylabel("Neurons (log scale)")
    ax.set_title("Two neurons stand out from 250,880 when the weights are read directly")
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{F}/fig1_weight_reading.png", dpi=220); plt.close(fig)

    # ---- Figure 2: wrong-language answers on held-out prompts, seen vs unseen templates ----
    ids = ["B0_baseline", "R0_random", "W0_weights_only", "W0a_only_L23N981", "S1_steering", "F1_lora", "P1_prompt"]
    order = [next(r for r in rows if r["id"] == i) for i in ids if any(r["id"] == i for r in rows)]
    fig, ax = plt.subplots(figsize=(6.6, 0.56 * len(order) + 1.25))
    y = np.arange(len(order))[::-1].astype(float)
    for key, ci, off, col, lab in [("sv_A", "A_ci", 0.19, BLUE, "Seen templates, unseen topics (sv_A, n=160)"),
                                    ("sv_B", "B_ci", -0.19, ORANGE, "Unseen templates and topics (sv_B, n=160)")]:
        vals = [100 * (1 - r[key]) for r in order]
        ax.barh(y + off, vals, height=0.34, color=col, label=lab)
        for yi, r, v in zip(y + off, order, vals):
            lo, hi = 100 * (1 - r[ci][1]), 100 * (1 - r[ci][0])
            ax.plot([lo, min(hi, 44)], [yi, yi], color=INK2, lw=0.8)
            ax.text(min(max(hi, v), 44) + 0.7, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[r["id"]] for r in order])
    ax.set_xlabel("Swedish prompts answered in another language (%)\nthin line: 95% interval, bootstrap over templates and topics (seed range for R0)")
    ax.set_title("Wrong-language answers to Swedish prompts (held-out)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0, 48)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{F}/fig2_drift.png", dpi=220); plt.close(fig)

    # ---- Figure 3: dev tuning, deleting vs reversing the two neurons ----
    g = [x for x in json.load(open(f"{R}/dev_tune_W1.json"))["grid"] if x["K"] == 2]
    dvb = json.load(open(f"{R}/dev_baseline.json"))
    xs = ["unedited"] + [f"× {x['c']}" if x["c"] else "× 0\n(delete)" for x in g]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(xs, [100 * dvb["sv_rate"]] + [100 * x["sv_rate"] for x in g], color=BLUE, lw=2, marker="o", ms=6, label="Answers in Swedish")
    ax.plot(xs, [100 * dvb["degenerate"]] + [100 * x["degenerate"] for x in g], color=ORANGE, lw=2, marker="s", ms=6, label="Degenerate (looping) answers")
    ax.set_ylabel("Dev-val prompts (%)"); ax.set_xlabel("Multiplier applied to the two neurons' output weights")
    ax.set_title("Deleting the two neurons helps; reversing them breaks the model (dev set)")
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.legend(frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(f"{F}/fig3_dev_scale.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
