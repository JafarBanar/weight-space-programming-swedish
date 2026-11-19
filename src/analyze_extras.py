"""Analyses requested by an external review that need no new model run (plus a weights-only null).
  python src/analyze_extras.py   -> results/r2/extras/saved_output_analyses.{json,md}, blind_review_items.json
"""
import gzip, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from common import MODEL_ID, MODEL_REV, lang_of
from prompts import test_sets, DEV_TEMPLATES, TEST_TEMPLATES
from prompts2 import mkqa, MKQA_PATH
from minimal_pairs import build as build_pairs
from native_pairs import build as build_native
import edits

OUT = "results/r2/extras"; os.makedirs(OUT, exist_ok=True)
res, L = {}, ["# Analyses of saved outputs and a calibrated null for the weight reading\n"]
tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
R1 = lambda n: json.load(open(f"results/test_{n}.json"))

# ---------- 1. calibrated null for the weight reading (weights only, CPU) ----------
f = safe_open(hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REV), "pt")
E, gamma = f.get_tensor("model.embed_tokens.weight").float(), f.get_tensor("model.norm.weight").float()
pairs = [(edits.single_token(tok, a), edits.single_token(tok, b)) for a, b in edits.PAIRS]
pool = [t for p in pairs for t in p]
T = E[pool] * gamma                                               # 34 x d
P = torch.stack([T @ f.get_tensor(f"model.layers.{l}.mlp.down_proj.weight").float() for l in range(28)])   # layers x 34 x neurons
def zmin(idx_a, idx_b):
    d = (T[idx_a] - T[idx_b]).mean(0); s = (P[:, idx_a] - P[:, idx_b]).mean(1) / d.norm()
    z = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True)
    return z
def tstat(idx_a, idx_b):
    """Consistency across the 17 pairs: mean per-pair write along the pair's own difference, divided by its standard error.
    A neuron driven by one token scores high on z but low here; a language-general neuron scores high on both."""
    D = P[:, idx_a] - P[:, idx_b]                                # layers x 17 x neurons
    return D.mean(1) / (D.std(1) / 17 ** 0.5)
z_real = zmin(list(range(0, 34, 2)), list(range(1, 34, 2)))
t_real = tstat(list(range(0, 34, 2)), list(range(1, 34, 2)))
rng = random.Random(0); null_min, null_absmax, null_tmin = [], [], []
for _ in range(200):
    perm = list(range(34)); rng.shuffle(perm)
    z = zmin(perm[:17], perm[17:]); null_min.append(float(z.min())); null_absmax.append(float(z.abs().max()))
    null_tmin.append(float(tstat(perm[:17], perm[17:]).min()))
t_rank = sorted(t_real.flatten().tolist())
res["null_calibration"] = {"real_z_L23N981": float(z_real[23, 981]), "real_z_L22N1821": float(z_real[22, 1821]), "real_min_z": float(z_real.min()),
                           "null_min_z": {"median": float(np.median(null_min)), "most_extreme": float(min(null_min)), "p5": float(np.percentile(null_min, 5))},
                           "null_absmax_z": {"median": float(np.median(null_absmax)), "max": float(max(null_absmax))},
                           "share_of_null_directions_with_any_neuron_beyond_minus_7.9": float(np.mean([m <= -7.9 for m in null_min])),
                           "share_beyond_minus_9.2": float(np.mean([m <= -9.2 for m in null_min])), "n_null": 200,
                           "consistency_t": {"L23N981": float(t_real[23, 981]), "L22N1821": float(t_real[22, 1821]), "real_min_t": float(t_real.min()),
                                             "rank_L23N981": t_rank.index(float(t_real[23, 981])) + 1, "rank_L22N1821": t_rank.index(float(t_real[22, 1821])) + 1,
                                             "null_min_t_median": float(np.median(null_tmin)), "null_min_t_most_extreme": float(min(null_tmin)),
                                             "share_null_beyond_L23N981": float(np.mean([m <= float(t_real[23, 981]) for m in null_tmin])),
                                             "share_null_beyond_L22N1821": float(np.mean([m <= float(t_real[22, 1821]) for m in null_tmin]))},
                           "pairs_with_danish_sign": {"L23N981": int(((P[23, 0::2, 981] - P[23, 1::2, 981]) < 0).sum()), "L22N1821": int(((P[22, 0::2, 1821] - P[22, 1::2, 1821]) < 0).sum())}}
n = res["null_calibration"]
L.append(f"## 1. Calibrated null for the weight reading\n\n200 control directions built from the same 34 function-word tokens paired at random. "
         f"Most negative neuron z-score per control direction: median {n['null_min_z']['median']:.1f}, most extreme {n['null_min_z']['most_extreme']:.1f}. "
         f"Real direction: {n['real_z_L23N981']:.1f} (L23/N981) and {n['real_z_L22N1821']:.1f} (L22/N1821). "
         f"Share of control directions with any neuron at or beyond −7.9: {100*n['share_of_null_directions_with_any_neuron_beyond_minus_7.9']:.1f}%; at or beyond −9.2: {100*n['share_beyond_minus_9.2']:.1f}%. "
         f"So a large z alone is not rare: a neuron that promotes one token produces it.\n\n"
         f"Consistency across the 17 pairs (mean per-pair write divided by its standard error): L23/N981 t = {n['consistency_t']['L23N981']:.1f} (rank {n['consistency_t']['rank_L23N981']} of 250,880, "
         f"{n['pairs_with_danish_sign']['L23N981']}/17 pairs with the Danish sign); L22/N1821 t = {n['consistency_t']['L22N1821']:.1f} (rank {n['consistency_t']['rank_L22N1821']}, {n['pairs_with_danish_sign']['L22N1821']}/17). "
         f"Most negative t over all neurons for the control directions: median {n['consistency_t']['null_min_t_median']:.1f}, most extreme {n['consistency_t']['null_min_t_most_extreme']:.1f}; "
         f"share of control directions with any neuron as consistent as L23/N981: {100*n['consistency_t']['share_null_beyond_L23N981']:.1f}%, as L22/N1821: {100*n['consistency_t']['share_null_beyond_L22N1821']:.1f}%.\n")

# ---------- 2. per-template table, round 1 ----------
b, w0, w0a = R1("B0_baseline"), R1("W0_weights_only"), R1("W0a_only_L23N981")
rows = []
for split, tmpl in (("sv_A", DEV_TEMPLATES), ("sv_B", TEST_TEMPLATES)):
    hb, h0, ha = (np.array(x[split]["hit"]).reshape(8, 20) for x in (b, w0, w0a))
    for t in range(8):
        rows.append({"split": split, "template": tmpl[t][0], "baseline_failures": int((hb[t] == 0).sum()),
                     "W0_fixed": int(((h0[t] == 1) & (hb[t] == 0)).sum()), "W0_broken": int(((h0[t] == 0) & (hb[t] == 1)).sum()),
                     "W0a_fixed": int(((ha[t] == 1) & (hb[t] == 0)).sum()), "W0a_broken": int(((ha[t] == 0) & (hb[t] == 1)).sum())})
res["per_template_round1"] = rows
top3 = sorted(rows, key=lambda r: -r["baseline_failures"])[:3]
L.append(f"## 2. Round 1 by template\n\nThree templates account for {sum(r['baseline_failures'] for r in top3)} of the {sum(r['baseline_failures'] for r in rows)} unedited failures.\n")
L += ["| Set | Template | Unedited failures (of 20) | W0 fixed / broken | W0a fixed / broken |", "|---|---|---|---|---|"]
L += [f"| {r['split']} | {r['template']} | {r['baseline_failures']} | {r['W0_fixed']} / {r['W0_broken']} | {r['W0a_fixed']} / {r['W0a_broken']} |" for r in rows]

# ---------- 3. grammar pairs with equal token counts ----------
def eq_mask(ps):
    return np.array([len(tok.encode(p["good"])) == len(tok.encode(p["bad"])) for p in ps])
sp, npairs = build_pairs(seed=0), build_native()
m_s, m_n = eq_mask(sp), eq_mask(npairs)
g = {"synthetic_equal_length": int(m_s.sum()), "synthetic_total": len(sp), "native_equal_length": int(m_n.sum()), "native_total": len(npairs), "conditions": {}}
for name in ("B0_baseline", "W0_weights_only", "W0a_only_L23N981", "W0b_only_L22N1821", "F1_lora"):
    h = np.array(R1(name)["minimal_pairs"]["hit"]); g["conditions"][name] = {"all": float(h.mean()), "equal_length": float(h[m_s].mean())}
for name in ("B0", "W0a", "W0"):
    p = f"results/r2/qwen2.5-1.5b/test_{name}.json"
    if os.path.exists(p):
        h = np.array(json.load(open(p))["pairs_native"]["hit"]); g["conditions"]["r2_native_" + name] = {"all": float(h.mean()), "equal_length": float(h[m_n].mean())}
res["grammar_pairs_equal_token_length"] = g
L.append(f"\n## 3. Grammar pairs restricted to equal token counts\n\n{g['synthetic_equal_length']} of {g['synthetic_total']} synthetic pairs and {g['native_equal_length']} of {g['native_total']} native pairs have the same number of tokens in both sentences.\n")
L += ["| Condition | All pairs | Equal-length pairs |", "|---|---|---|"] + [f"| {k} | {100*v['all']:.1f}% | {100*v['equal_length']:.1f}% |" for k, v in g["conditions"].items()]

# ---------- 4. MKQA: factual correctness jointly with answer language ----------
_, mk = mkqa(0); gold = {}
with gzip.open(MKQA_PATH, "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line); gold[r["example_id"]] = r["answers"]
def answers(item):
    out = set()
    for lang in ("sv", "en"):
        for a in gold[item["id"]].get(lang, []):
            if a.get("type") in ("unanswerable", "long_answer", "binary"):
                continue
            for t in [a.get("text")] + (a.get("aliases") or []):
                if t and len(str(t)) >= 2:
                    out.add(str(t).lower())
    return out
G = [answers(x) for x in mk]; scorable = np.array([len(a) > 0 for a in G])
mkres = {"scorable_questions": int(scorable.sum()), "conditions": {}}
def score(outputs, hits):
    fact = np.array([any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", o.lower()) for a in gs) for o, gs in zip(outputs, G)])
    hits = np.array(hits)
    return {"fact": float(fact[scorable].mean()), "language": float(hits[scorable].mean()), "joint": float((fact & (hits == 1))[scorable].mean())}
for name in ("B0", "W0a", "W0", "P1"):
    p = f"results/r2/qwen2.5-1.5b/test_{name}.json"
    if os.path.exists(p):
        x = json.load(open(p))["MK_sv"]; mkres["conditions"][name] = score(x["outputs"], x["hit"])
p = f"{OUT}/E2_same_language_prompt.json"
if os.path.exists(p):
    x = json.load(open(p))["MK_sv"]; mkres["conditions"]["P2_same_language"] = score(x["outputs"], x["hit"])
res["mkqa_joint"] = mkres
L.append(f"\n## 4. MKQA Swedish questions: gold answer found, answer in Swedish, both\n\n{mkres['scorable_questions']} of 300 questions have a short gold answer (string match against the dataset's Swedish and English answers and aliases).\n")
L += ["| Condition | Gold answer found | Answer in Swedish | Both |", "|---|---|---|---|"] + [f"| {k} | {100*v['fact']:.1f}% | {100*v['language']:.1f}% | {100*v['joint']:.1f}% |" for k, v in mkres["conditions"].items()]

# ---------- 5. round-1 Danish result without the three flagged topic translations ----------
flag = {8, 9, 11}                                                 # genbrug, hjemmearbejde, ligestilling
keep = np.array([(i % 20) not in flag for i in range(160)])
dres = {}
for name in ("B0_baseline", "W0_weights_only", "W0a_only_L23N981", "F1_lora"):
    h = np.array(R1(name)["da_B"]["hit"]); dres[name] = {"all_160": float(h.mean()), "without_flagged_136": float(h[keep].mean())}
res["round1_danish_without_flagged_topics"] = dres
L.append("\n## 5. Round-1 Danish prompts answered in Danish, with and without three imprecise topic translations\n")
L += ["| Condition | All 160 | Without genbrug / hjemmearbejde / ligestilling (136) |", "|---|---|---|"] + [f"| {k} | {100*v['all_160']:.1f}% | {100*v['without_flagged_136']:.1f}% |" for k, v in dres.items()]

# ---------- 6. items for a blinded re-labelling ----------
T = test_sets(); items = []
def add(source, cond, text, label):
    items.append({"source": source, "condition": cond, "text": text, "detector": label})
for split in ("sv_A", "sv_B"):
    for i in range(160):
        lb, l0, la = (lang_of(x[split]["outputs"][i]) for x in (b, w0, w0a))
        if lb == "sv" and l0 != "sv":
            add("round1 W0 broken", "W0", w0[split]["outputs"][i], l0); add("round1 W0 broken", "B0", b[split]["outputs"][i], lb)
        if lb != "sv" and la == "sv" and i % 2 == 0:
            add("round1 W0a fixed", "W0a", w0a[split]["outputs"][i], la); add("round1 W0a fixed", "B0", b[split]["outputs"][i], lb)
p = "results/r2/qwen2.5-1.5b/test_B0.json"
if os.path.exists(p):
    x = json.load(open(p))["MK_sv"]; en = [i for i, o in enumerate(x["outputs"]) if lang_of(o) == "en"]; sv = [i for i, o in enumerate(x["outputs"]) if lang_of(o) == "sv"]
    r2 = random.Random(1)
    for i in r2.sample(en, 15):
        add("MKQA B0 labelled English", "B0", x["outputs"][i], "en")
    for i in r2.sample(sv, 8):
        add("MKQA B0 labelled Swedish", "B0", x["outputs"][i], "sv")
random.Random(2).shuffle(items)
json.dump(items, open(f"{OUT}/blind_review_key.json", "w"), ensure_ascii=False, indent=1)
json.dump([{"id": k, "text": it["text"]} for k, it in enumerate(items)], open(f"{OUT}/blind_review_items.json", "w"), ensure_ascii=False, indent=1)
res["blind_review_items"] = len(items)

json.dump(res, open(f"{OUT}/saved_output_analyses.json", "w"), ensure_ascii=False, indent=1)
open(f"{OUT}/saved_output_analyses.md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("\nblind review items:", len(items))
