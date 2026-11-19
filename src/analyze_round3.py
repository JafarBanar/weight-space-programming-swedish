"""Round-3 tables, pre-registered verdicts (G1-G5) and figure.
  python src/analyze_round3.py   ->  results/r3/summary.json, results/r3/summary.md, figures/fig5_round3.png"""
import json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from prompts3 import mkqa3, gold_answers, LANGS4

R = "results/r3"
NAMES = {"qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct", "qwen2.5-0.5b": "Qwen2.5-0.5B-Instruct"}
LAB = {"B0": "B0 Unedited", "P1": "P1 Prompt: always Swedish", "P2": "P2 Prompt: same language (round 2)",
       "P3": "P3 Prompt: best of six on dev", "W0a": "W0a Neuron edit (rounds 1–2)", "S2": "S2 Steering (round 2)",
       "E1": "E1 Closed-form rank-3 edit", "R3": "R3 Random rank-3 edit, same size", "F3": "F3 Learned rank-3 edit, same matrix",
       "F2": "F2 LoRA distillation"}
ORDER = ["B0", "P1", "P2", "P3", "W0a", "S2", "R3", "E1", "F3", "F2"]
INK, GRID = "#1f2933", "#d9dee3"
COL = {"sv": "#2a6fbb", "da": "#d1603d", "no": "#3a9d8f", "en": "#8a8f98"}
LANGNAME = {"sv": "Swedish", "da": "Danish", "no": "Norwegian", "en": "English"}


def boot_ci(a, b, n=10000, seed=0):
    """Paired bootstrap over questions for mean(b) - mean(a)."""
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return float(b.mean() - a.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def analyse(key):
    d = f"{R}/{key}"
    if not os.path.exists(f"{d}/test_B0.json"):
        return None
    _, _, test = mkqa3(); gold = gold_answers()
    G = [gold[x["id"]] for x in test]; scorable = np.array([len(g) > 0 for g in G])

    def fact_hits(outputs):
        f = np.array([any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", o.lower()) for a in gs) for o, gs in zip(outputs, G)])
        return f[scorable]

    def fact(outputs):
        return float(fact_hits(outputs).mean())
    res = {c: json.load(open(f"{d}/test_{c}.json")) for c in ORDER if os.path.exists(f"{d}/test_{c}.json")}
    b = res["B0"]; out = {"model": NAMES[key], "n_test": len(test), "scorable": int(scorable.sum()), "conds": {}}
    b_wrong = 1 - b["MK_sv"]["rate"]
    for c, r in res.items():
        x = {l: r[f"MK_{l}"]["rate"] for l in LANGS4}
        x["langs_sv"] = r["MK_sv"]["langs"]
        x["dano_merged"] = {l: (r[f"MK_{l}"]["langs"].get("da", 0) + r[f"MK_{l}"]["langs"].get("no", 0)) / len(test) for l in ("da", "no")}
        x["gain_sv"], x["gain_sv_ci"] = boot_ci(b["MK_sv"]["hit"], r["MK_sv"]["hit"])
        x["delta"] = {l: boot_ci(b[f"MK_{l}"]["hit"], r[f"MK_{l}"]["hit"])[0] for l in ("da", "no", "en")}
        x["delta_ci"] = {l: boot_ci(b[f"MK_{l}"]["hit"], r[f"MK_{l}"]["hit"])[1] for l in ("da", "no", "en")}
        x["fact_sv"] = fact(r["MK_sv"]["outputs"]); x["degenerate_sv"] = r["MK_sv"]["degenerate"]
        fh, bh = fact_hits(r["MK_sv"]["outputs"]), fact_hits(b["MK_sv"]["outputs"])
        x["fact_n"] = int(fh.sum()); x["fact_delta"], x["fact_delta_ci"] = boot_ci(bh, fh)
        x["joint_n"] = int((fh & (np.array(r["MK_sv"]["hit"])[scorable] == 1)).sum())      # gold answer found AND answered in Swedish
        if "ppl" in r:
            x["T1_sv"] = r["T1_sv"]["rate"]; x["ppl"] = r["ppl"]
            x["ppl_ratio"] = {k: r["ppl"][k] / b["ppl"][k] for k in LANGS4}
            x["pairs_native"] = r["pairs_native"]["accuracy"]; x["pairs_synthetic"] = r["pairs_synthetic"]["accuracy"]
            g1_level = x["sv"] >= 0.90 if key == "qwen2.5-1.5b" else (1 - x["sv"]) <= 0.5 * b_wrong
            v = {"G1": bool(g1_level and x["gain_sv_ci"][0] > 0),
                 "G2": bool(all(x[l] >= b[f"MK_{l}"]["rate"] - 0.025 for l in ("da", "no", "en"))),
                 "G3": bool(all(x["ppl_ratio"][k] <= 1.05 for k in LANGS4)
                            and abs(x["pairs_native"] - b["pairs_native"]["accuracy"]) < 0.03
                            and abs(x["pairs_synthetic"] - b["pairs_synthetic"]["accuracy"]) < 0.03),
                 "G4": bool(x["fact_sv"] >= fact(b["MK_sv"]["outputs"]) - 0.02),
                 "G5": bool(x["T1_sv"] >= b["T1_sv"]["rate"] - 0.01)}
            v["works"] = all(v.values()); x["verdict"] = v
        out["conds"][c] = x
    rnd = []                                                    # random edits of E1's size: the pre-registered one plus extra seeds
    for f in ["test_R3.json"] + sorted(x for x in os.listdir(d) if x.startswith("test_R3_seed")):
        if os.path.exists(f"{d}/{f}"):
            r = json.load(open(f"{d}/{f}"))
            rnd.append({"file": f, **{l: r[f"MK_{l}"]["rate"] - b[f"MK_{l}"]["rate"] for l in LANGS4},
                        "fact_n": int(fact_hits(r["MK_sv"]["outputs"]).sum())})
    out["random_edits"] = rnd
    for name in ("frozen", "train_F2"):
        if os.path.exists(f"{d}/{name}.json"):
            out[name] = json.load(open(f"{d}/{name}.json"))
    out["robust"] = {}
    for c in ORDER:
        p = f"{d}/robust_{c}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            out["robust"][c] = {"sampled": {l: [s["rate"] for s in r[f"sampled_{l}"]] for l in ("sv", "da")},
                                "long": {l: r[f"long_{l}"]["rate"] for l in ("sv", "da")},
                                "unseen_system": {l: v["rate"] for l, v in r.get("unseen_system", {}).items()}}
    return out


def table(S):
    def pct(x):
        return f"{100 * x:.1f}%"
    L = [f"\n### {S['model']} ({S['n_test']} fresh test questions per language; {S['scorable']} Swedish questions have a short gold answer)\n",
         "| Condition | Swedish→Swedish | Gain vs B0 [95% CI] | Danish→Danish | Norwegian→Norwegian | English→English | PPL change sv / da / no / en | Grammar pairs native / synthetic | Gold answer found (sv) | T1 Swedish | G1 G2 G3 G4 G5 | Works |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        if c not in S["conds"]:
            continue
        x = S["conds"][c]; v = x.get("verdict", {})
        gain = "–" if c == "B0" else f"{100 * x['gain_sv']:+.1f} [{100 * x['gain_sv_ci'][0]:+.1f}, {100 * x['gain_sv_ci'][1]:+.1f}]"
        ppl = " / ".join(f"{100 * (x['ppl_ratio'][k] - 1):+.1f}%" for k in LANGS4) if "ppl" in x else "–"
        marks = " ".join("✓" if v.get(g) else "✗" for g in ("G1", "G2", "G3", "G4", "G5")) if v and c != "B0" else "–"
        L.append(f"| {LAB[c]} | {pct(x['sv'])} | {gain} | {pct(x['da'])} | {pct(x['no'])} | {pct(x['en'])} | {ppl} | "
                 f"{pct(x['pairs_native'])} / {pct(x['pairs_synthetic'])} | {pct(x['fact_sv'])} | {pct(x['T1_sv'])} | {marks} | "
                 f"{'**yes**' if v.get('works') and c != 'B0' else ('–' if c == 'B0' else 'no')} |")
    L += ["", "Answer content on the Swedish questions with a short gold answer:", "",
          "| Condition | Gold answer found | Change vs B0 [95% CI] | Gold answer found and answer in Swedish | Languages of the answers to Swedish questions |", "|---|---|---|---|---|"]
    for c in ORDER:
        if c in S["conds"]:
            x = S["conds"][c]
            ch = "–" if c == "B0" else f"{100 * x['fact_delta']:+.1f} [{100 * x['fact_delta_ci'][0]:+.1f}, {100 * x['fact_delta_ci'][1]:+.1f}]"
            L.append(f"| {LAB[c]} | {x['fact_n']} / {S['scorable']} | {ch} | {x['joint_n']} / {S['scorable']} | "
                     + ", ".join(f"{k} {v}" for k, v in sorted(x["langs_sv"].items(), key=lambda kv: -kv[1])) + " |")
    if len(S["random_edits"]) > 1:
        L += ["", "Random rank-3 edits of the same matrix and norm as E1 (change vs B0 in points; the first is the pre-registered control, the rest were added after the test):", "",
              "| Edit | Swedish | Danish | Norwegian | English | Gold answer found |", "|---|---|---|---|---|---|"]
        L += [f"| {r['file'][5:-5]} | {100 * r['sv']:+.1f} | {100 * r['da']:+.1f} | {100 * r['no']:+.1f} | {100 * r['en']:+.1f} | {r['fact_n']} / {S['scorable']} |" for r in S["random_edits"]]
    if S["robust"]:
        L += ["", "Robustness (Swedish / Danish right-language rate):", "",
              "| Condition | Sampling T=0.7, three seeds (sv) | Sampling (da) | 160-token answers sv / da | Unseen English system prompt sv / da / no / en |", "|---|---|---|---|---|"]
        for c, r in S["robust"].items():
            us = " / ".join(pct(r["unseen_system"][l]) for l in LANGS4) if r["unseen_system"] else "–"
            L.append(f"| {LAB[c]} | {', '.join(pct(v) for v in r['sampled']['sv'])} | {', '.join(pct(v) for v in r['sampled']['da'])} | "
                     f"{pct(r['long']['sv'])} / {pct(r['long']['da'])} | {us} |")
    return L


def figure(S, path):
    conds = [c for c in ORDER if c in S["conds"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.40 * len(conds) + 1.25))
    y = np.arange(len(conds))[::-1]
    for l, dy in zip(LANGS4, (0.21, 0.07, -0.07, -0.21)):
        v = [100 * S["conds"][c][l] for c in conds]
        ax.scatter(v, y + dy, s=38, color=COL[l], edgecolor="white", linewidth=1.1, zorder=3, clip_on=False,
                   label=f"{LANGNAME[l]} question → {LANGNAME[l]} answer")
    for yi, c in zip(y, conds):
        if S["conds"][c].get("verdict", {}).get("works") and c != "B0":
            ax.text(104, yi, "passes all\nfive checks", va="center", fontsize=7.5, color=INK, fontweight="bold")
        ax.axhline(yi - 0.5, color=GRID, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([LAB[c] for c in conds], fontsize=8)
    ax.set_ylim(-0.5, len(conds) - 0.5); ax.set_xlim(0, 100)
    ax.set_xlabel("Answers in the language of the question (%)", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    top = 1 - 0.95 / fig.get_figheight()
    fig.subplots_adjust(left=0.33, right=0.86, bottom=0.11, top=top)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.2)
    fig.text(0.01, 1 - 0.22 / fig.get_figheight(), f"Round 3 on {S['n_test']} fresh human-translated questions per language",
             ha="left", va="center", fontsize=10, fontweight="bold")
    fig.savefig(path, dpi=220); plt.close(fig)


TEX = {"B0": "B0 Unedited", "P1": "P1 Prompt: always Swedish", "P2": "P2 Prompt: same language", "P3": "P3 Prompt: best of six",
       "W0a": "W0a Neuron edit", "S2": "S2 Steering", "E1": "E1 Closed-form rank-3 edit", "R3": "R3 Random rank-3 edit",
       "F3": "F3 Learned rank-3 edit", "F2": "F2 LoRA distillation"}


def latex(S, path, macro):
    """Table rows for the report, generated from the results so that no number is typed by hand."""
    rows = []
    for c in ORDER:
        if c not in S["conds"]:
            continue
        x = S["conds"][c]; v = x.get("verdict", {})
        gain = "--" if c == "B0" else f"${100 * x['gain_sv']:+.1f}$ [${100 * x['gain_sv_ci'][0]:+.1f}$, ${100 * x['gain_sv_ci'][1]:+.1f}$]"
        worst = max(x["ppl_ratio"].values()) - 1
        marks = "--" if c == "B0" else " ".join(r"\checkmark" if v[g] else r"$\times$" for g in ("G1", "G2", "G3", "G4", "G5"))
        works = "--" if c == "B0" else (r"\textbf{yes}" if v["works"] else "no")
        rows.append(f"{TEX[c]} & {100 * x['sv']:.1f} & {gain} & {100 * x['da']:.1f} & {100 * x['no']:.1f} & {100 * x['en']:.1f} & "
                    f"${100 * worst:+.1f}$ & {100 * x['pairs_native']:.1f} & {100 * x['fact_sv']:.1f} & {100 * x['T1_sv']:.1f} & {marks} & {works}\\\\")
    open(path, "w").write("\\newcommand{\\" + macro + "}{%\n" + "\n".join(rows) + "\n}\n")
    if S["robust"]:
        rr = []
        for c, r in S["robust"].items():
            us = " / ".join(f"{100 * r['unseen_system'][l]:.0f}" for l in LANGS4) if r["unseen_system"] else "--"
            rr.append(f"{TEX[c]} & {' / '.join(f'{100 * v:.0f}' for v in r['sampled']['sv'])} & {' / '.join(f'{100 * v:.0f}' for v in r['sampled']['da'])} & "
                      f"{100 * r['long']['sv']:.0f} / {100 * r['long']['da']:.0f} & {us}\\\\")
        open(path.replace(".tex", "_robust.tex"), "w").write("\\newcommand{\\" + macro + "Robust}{%\n" + "\n".join(rr) + "\n}\n")


def main():
    out, L = {}, ["# Round 3 summary (generated by src/analyze_round3.py)"]
    for key in NAMES:
        S = analyse(key)
        if S:
            out[key] = S; L += table(S)
    json.dump(out, open(f"{R}/summary.json", "w"), indent=1, ensure_ascii=False)
    open(f"{R}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    if "qwen2.5-1.5b" in out:
        figure(out["qwen2.5-1.5b"], "figures/fig5_round3.png"); latex(out["qwen2.5-1.5b"], "report/r3_table.tex", "RThreeRows")
    if "qwen2.5-0.5b" in out:
        latex(out["qwen2.5-0.5b"], "report/r3_table_second.tex", "RThreeRowsSecond")


if __name__ == "__main__":
    main()
