"""Round-2 tables and figures from results/r2/.   python src/analyze_round2.py"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import marker_share
from minimal_pairs import build as build_pairs

R = "results/r2"
NAMES = {"qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct", "qwen2.5-0.5b": "Qwen2.5-0.5B-Instruct",
         "eurollm-1.7b": "EuroLLM-1.7B-Instruct", "qwen3-1.7b": "Qwen3-1.7B"}
LAB = {"B0": "B0 Unedited", "W0a": "W0a Delete L23/N981 (pre-registered)", "W0": "W0 Delete both neurons", "RECIPE": "Recipe edit",
       "P1": "P1 Prompt instruction", "S2_steering": "S2 Steering, best of 54", "R": "R Random neuron(s), 2 seeds"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def boot_t1(h, h0=None, n=2000, seed=0):
    g = np.random.default_rng(seed); out = []
    for _ in range(n):
        r, c = g.integers(0, 40, 40), g.integers(0, 8, 8)
        out.append(h[np.ix_(r, c)].mean() - (h0[np.ix_(r, c)].mean() if h0 is not None else 0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def boot_iid(h, h0=None, n=2000, seed=0):
    g = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = g.integers(0, len(h), len(h))
        out.append(h[i].mean() - (h0[i].mean() if h0 is not None else 0))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def cleaned_mask():
    """Synthetic pairs whose noun the Wiktionary/Korp checks could not confirm are dropped."""
    pairs = build_pairs(seed=0); drop = {"kort"}
    p = "results/language_checks.json"
    if os.path.exists(p):
        for r in json.load(open(p))["nouns"]:
            if r["both_genders_listed"] or (not r["ok"] and not r.get("lookup_failed")):
                drop.add(r["noun"])
    return np.array([not (set(x["good"].rstrip(".").split()) & drop) for x in pairs]), sorted(drop)


def marker_rate(x):
    out = []
    for o, h in zip(x["outputs"], x["hit"]):
        m = marker_share(o); out.append(h if m is None else int(m > 0.5))
    return float(np.mean(out))


def row(r, b, mask):
    def cnt(x, *ks):
        return sum(x["langs"].get(k, 0) for k in ks)
    h = np.array(r["T1_sv"]["hit"]).reshape(40, 8); h0 = np.array(b["T1_sv"]["hit"]).reshape(40, 8)
    m = np.array(r["MK_sv"]["hit"]); m0 = np.array(b["MK_sv"]["hit"])
    d = {"T1_sv": float(h.mean()), "T1_ci": boot_t1(h), "T1_diff": float(h.mean() - h0.mean()), "T1_diff_ci": boot_t1(h, h0),
         "T1_fixed": int(((h == 1) & (h0 == 0)).sum()), "T1_broken": int(((h == 0) & (h0 == 1)).sum()),
         "T1_dano": cnt(r["T1_sv"], "da", "no"), "T1_en": cnt(r["T1_sv"], "en"), "T1_n": 320, "T1_langs": r["T1_sv"]["langs"],
         "T1_markers": marker_rate(r["T1_sv"]), "T1_degenerate": r["T1_sv"]["degenerate"],
         "MK_sv": float(m.mean()), "MK_ci": boot_iid(m), "MK_diff": float(m.mean() - m0.mean()), "MK_diff_ci": boot_iid(m, m0),
         "MK_dano": cnt(r["MK_sv"], "da", "no"), "MK_en": cnt(r["MK_sv"], "en"), "MK_n": len(m), "MK_langs": r["MK_sv"]["langs"],
         "MK_markers": marker_rate(r["MK_sv"])}
    if "ppl" in r:
        ps = np.array(r["pairs_synthetic"]["hit"])
        d.update({"da_da": r["MK_da"]["rate"], "da_langs": r["MK_da"]["langs"], "en_en_mkqa": r["MK_en"]["rate"], "en_en_t1": r["T1_en"]["rate"],
                  "ppl": r["ppl"], "pairs_synthetic": float(ps.mean()), "pairs_cleaned": float(ps[mask].mean()),
                  "pairs_native": r["pairs_native"]["accuracy"], "pairs_synthetic_by": r["pairs_synthetic"]["by_phenomenon"]})
    return d


def main():
    mask, dropped = cleaned_mask()
    S = {"cleaned_pairs": {"kept": int(mask.sum()), "of": int(len(mask)), "dropped_nouns": dropped}, "models": {}}
    for key in NAMES:
        d = f"{R}/{key}"
        if not os.path.exists(f"{d}/test_B0.json"):
            continue
        b = json.load(open(f"{d}/test_B0.json"))
        M = {"reading": json.load(open(f"{d}/reading.json")), "recipe": json.load(open(f"{d}/recipe_decision.json")), "conds": {}}
        for f in sorted(glob.glob(f"{d}/test_*.json")):
            name = os.path.basename(f)[5:-5]
            if not name.startswith("R_seed"):
                M["conds"][name] = row(json.load(open(f)), b, mask)
        rs = [row(json.load(open(f)), b, mask) for f in sorted(glob.glob(f"{d}/test_R_seed*.json"))]
        if rs:
            M["conds"]["R"] = {k: float(np.mean([x[k] for x in rs])) for k in ("T1_sv", "T1_diff", "MK_sv", "MK_diff", "T1_dano", "MK_dano", "T1_en", "MK_en")}
            M["conds"]["R"]["seeds_T1_sv"] = [x["T1_sv"] for x in rs]; M["conds"]["R"]["seeds_T1_dano"] = [x["T1_dano"] for x in rs]
        if os.path.exists(f"{d}/steer_selected.json"):
            M["steering"] = json.load(open(f"{d}/steer_selected.json"))
        S["models"][key] = M
    json.dump(S, open(f"{R}/summary.json", "w"), indent=1, ensure_ascii=False)

    L = []
    def pct(x):
        return f"{100 * x:.1f}%"
    for key, M in S["models"].items():
        rd, rc = M["reading"], M["recipe"]
        L.append(f"\n### {NAMES[key]}\n")
        L.append(f"Weight reading: **{rd['tier']}**; candidates " + ", ".join(f"L{l}/N{j} (z = {z:.1f})" for (l, j), z in zip(rd["candidates"], rd["z"])) +
                 ". Sign step: " + "; ".join(f"L{x['neuron'][0]}/N{x['neuron'][1]} fires {x['act_danish_answers']:+.1f} on Danish/Norwegian answers and {x['act_swedish_answers']:+.1f} on Swedish answers → {'delete' if x['delete'] else 'keep'}" for x in rc["decision"]) +
                 f" ({rc['n_danish_answers']} Danish/Norwegian and {rc['n_swedish_answers']} Swedish dev answers).\n")
        L.append("| Condition | T1: Swedish answers (320 new-template prompts) | Δ vs B0 [95% CI] | T1 answers in da+no / en | MKQA: Swedish answers (300 human-translated questions) | Δ vs B0 [95% CI] | MKQA answers in da+no / en | Danish→Danish | English→English (MKQA / T1) | PPL sv / da | Grammar pairs: synthetic / cleaned / native |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        order = [c for c in ("B0", "R", "W0a", "W0", "RECIPE", "S2_steering", "P1") if c in M["conds"]]
        for c in order:
            x = M["conds"][c]
            if c == "R":
                L.append(f"| {LAB[c]} | {pct(x['T1_sv'])} | {100*x['T1_diff']:+.1f} | {x['T1_dano']:.0f} / {x['T1_en']:.0f} | {pct(x['MK_sv'])} | {100*x['MK_diff']:+.1f} | {x['MK_dano']:.0f} / {x['MK_en']:.0f} | – | – | – | – |")
                continue
            ci = "–" if c == "B0" else f"{100*x['T1_diff']:+.1f} [{100*x['T1_diff_ci'][0]:+.1f}, {100*x['T1_diff_ci'][1]:+.1f}]"
            ci2 = "–" if c == "B0" else f"{100*x['MK_diff']:+.1f} [{100*x['MK_diff_ci'][0]:+.1f}, {100*x['MK_diff_ci'][1]:+.1f}]"
            tail = (f"{pct(x['da_da'])} | {pct(x['en_en_mkqa'])} / {pct(x['en_en_t1'])} | {x['ppl']['sv']:.1f} / {x['ppl']['da']:.1f} | "
                    f"{pct(x['pairs_synthetic'])} / {pct(x['pairs_cleaned'])} / {pct(x['pairs_native'])}") if "ppl" in x else "– | – | – | –"
            L.append(f"| {LAB[c]} | {pct(x['T1_sv'])} | {ci} | {x['T1_dano']} / {x['T1_en']} | {pct(x['MK_sv'])} | {ci2} | {x['MK_dano']} / {x['MK_en']} | {tail} |")
        if "steering" in M:
            st = M["steering"]
            L.append(f"\nSteering search: {st['configs']} configurations, {st['dev_prompts_run']} dev prompts; dev-val Swedish rate unedited {pct(st['baseline']['sv_rate'])}, selected: {st['selected']}.")
    open(f"{R}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))

    # ---- Figure 4: survey across models ----
    keys = [k for k in NAMES if k in S["models"]]
    if keys:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
        for ax, (pre, ttl) in zip(axes, (("T1", "New template prompts (n = 320)"), ("MK", "Human-translated questions (n = 300)"))):
            x = np.arange(len(keys)); w = 0.26
            for i, (cond_of, col, lab) in enumerate(((lambda M: "B0", BLUE, "Unedited"),
                                                     (lambda M: "W0a" if "W0a" in M["conds"] else ("RECIPE" if "RECIPE" in M["conds"] else None), ORANGE, "Recipe edit"),
                                                     (lambda M: "P1", AQUA, "Prompt instruction"))):
                vals = []
                for k in keys:
                    c = cond_of(S["models"][k]); vals.append(np.nan if c is None or c not in S["models"][k]["conds"] else 100 * (1 - S["models"][k]["conds"][c][f"{pre}_sv"]))
                ax.bar(x + (i - 1) * w, np.nan_to_num(vals), width=w - 0.03, color=col, label=lab)
                for xi, v in zip(x + (i - 1) * w, vals):
                    ax.text(xi, (0 if np.isnan(v) else v) + 1.2, "no edit" if np.isnan(v) else f"{v:.0f}", ha="center", fontsize=7.5, color=INK, rotation=90 if np.isnan(v) else 0)
            ax.set_xticks(x); ax.set_xticklabels([NAMES[k].replace("-Instruct", "").replace("-", "-\n", 1) for k in keys], fontsize=8)
            ax.set_title(ttl, fontsize=9); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
        axes[0].set_ylabel("Swedish prompts answered\nin another language (%)"); axes[1].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle("Wrong-language answers across four models (lower is better)", x=0.01, ha="left", fontsize=10, fontweight="bold")
        fig.tight_layout(); fig.savefig("figures/fig4_survey.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
