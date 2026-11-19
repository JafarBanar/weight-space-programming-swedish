"""Exploratory round-2 extras on Qwen2.5-1.5B-Instruct. See PREREGISTRATION_ROUND2_ADDENDUM.md.
  python src/run_extras.py            -> results/r2/extras/*.json
"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, SEED, free_memory
from evaluate import response_language
from prompts2 import t1_prompts, mkqa
import edits

OUT = "results/r2/extras"
FACTS = [  # (Swedish question, Danish question, Swedish answer, Danish answer); answers aligned word by word
    ("Vilken är huvudstaden i Frankrike?", "Hvad er hovedstaden i Frankrig?", "Frankrikes huvudstad heter Paris", "Frankrigs hovedstad hedder Paris"),
    ("Hur många dagar har en vecka?", "Hvor mange dage har en uge?", "En vecka har sju dagar", "En uge har syv dage"),
    ("Vilken färg har himlen en klar dag?", "Hvilken farve har himlen på en klar dag?", "Himlen blir ofta blå när solen skiner", "Himlen bliver ofte blå når solen skinner"),
    ("Vilket djur ger oss mjölk?", "Hvilket dyr giver os mælk?", "Kon ger oss mjölk", "Koen giver os mælk"),
    ("Vilken årstid kommer efter vintern?", "Hvilken årstid kommer efter vinteren?", "Våren kommer efter vintern", "Foråret kommer efter vinteren"),
    ("Vad heter den största staden i Sverige?", "Hvad hedder den største by i Sverige?", "Den största staden i Sverige heter Stockholm", "Den største by i Sverige hedder Stockholm"),
    ("Vad gör man med en bok?", "Hvad gør man med en bog?", "Man läser en bok", "Man læser en bog"),
    ("Vad behöver växter för att växa?", "Hvad har planter brug for for at vokse?", "Växter behöver vatten och ljus", "Planter behøver vand og lys"),
    ("Vad äter man ofta till frukost?", "Hvad spiser man ofte til morgenmad?", "Många äter bröd till frukost", "Mange spiser brød til morgenmad"),
    ("Vad använder man för att skriva?", "Hvad bruger man til at skrive?", "Man använder en penna", "Man bruger en pen"),
    ("Vilken månad är ofta kallast?", "Hvilken måned er ofte koldest?", "Januari är ofta den kallaste månaden", "Januar er ofte den koldeste måned"),
    ("Vad gör en lärare?", "Hvad laver en lærer?", "En lärare undervisar elever i skolan", "En lærer underviser elever i skolen"),
    ("Var lever fiskar?", "Hvor lever fisk?", "Fiskar lever i vatten", "Fisk lever i vand"),
    ("Vad gör de flesta på natten?", "Hvad gør de fleste om natten?", "På natten sover de flesta människor", "Om natten sover de fleste mennesker"),
]
SAME_LANG = "You are a helpful assistant. Always answer in the same language as the user's message."
DA_SYS = "You are a helpful assistant. Svar altid på dansk."


def contexts():
    rows = []
    for fi, (qs, qd, s, d) in enumerate(FACTS):
        S, D = s.split(), d.split(); assert len(S) == len(D), (s, d)
        diff = [k for k in range(len(S)) if S[k] != D[k]]; i, j = diff[0], diff[-1]; assert j > i
        for pl, q in (("sv", qs), ("da", qd)):
            for ctx, prefix, k in (("neutral", S[:i], i), ("after_danish", D[:j], j), ("after_swedish", S[:j], j)):
                rows.append({"fact": fi, "prompt_lang": pl, "context": ctx, "q": q, "prefix": " ".join(prefix),
                             "sv_word": (" " if k else "") + S[k], "da_word": (" " if k else "") + D[k]})
    return rows


@torch.no_grad()
def word_logprob(model, tok, ctx_ids, word, ablate=None, scope=None, record=None):
    """log P(word | context), teacher-forced. ablate=(layer, neuron) zeroes that neuron's activation either only at
    the positions that predict the word ('position') or at every position ('everywhere')."""
    tgt = tok.encode(word, add_special_tokens=False)
    ids = torch.tensor([ctx_ids + tgt[:-1]], device=model.device)
    start = len(ctx_ids) - 1; hooks = []
    if ablate:
        l, j = ablate
        def pre(m, inp):
            x = inp[0].clone()
            x[:, (start if scope == "position" else 0):, j] = 0
            return (x,)
        hooks.append(edits.layers(model)[l].mlp.down_proj.register_forward_pre_hook(pre))
    if record is not None:
        for (l, j) in record:
            hooks.append(edits.layers(model)[l].mlp.down_proj.register_forward_pre_hook(
                lambda m, inp, l=l, j=j: record[(l, j)].append(float(inp[0][0, start, j]))))
    logp = torch.log_softmax(model(input_ids=ids).logits[0, start:].float(), -1)
    for h in hooks:
        h.remove()
    return float(sum(logp[t, tgt[t]] for t in range(len(tgt))))


@torch.no_grad()
def fixed_prefix(model, tok):
    rows = contexts(); W = edits.layers(model)[23].mlp.down_proj.weight.float()
    # control neuron: same layer, closest output-norm x RMS activation at the predicting positions
    acts = []
    store = []
    h = edits.layers(model)[23].mlp.down_proj.register_forward_pre_hook(lambda m, inp: store.append(inp[0][0, -1].float().cpu()))
    for r in rows:
        r["ctx_ids"] = tok(chat(tok, r["q"]) + r["prefix"])["input_ids"]
        model(input_ids=torch.tensor([r["ctx_ids"]], device=model.device))
    h.remove(); A = torch.stack(store)
    strength = (A.pow(2).mean(0).sqrt() * W.norm(dim=0).cpu())
    order = (strength - strength[981]).abs().argsort().tolist(); control = next(j for j in order if j != 981)
    neurons = {"N981": (23, 981), "N1821": (22, 1821), "control": (23, control)}
    for r in rows:
        rec = {(23, 981): [], (22, 1821): []}
        base = word_logprob(model, tok, r["ctx_ids"], r["sv_word"], record=rec) - word_logprob(model, tok, r["ctx_ids"], r["da_word"])
        r["margin_base"] = base; r["act_N981"] = rec[(23, 981)][0]; r["act_N1821"] = rec[(22, 1821)][0]
        for name, n in neurons.items():
            for scope in ("position", "everywhere"):
                m = word_logprob(model, tok, r["ctx_ids"], r["sv_word"], n, scope) - word_logprob(model, tok, r["ctx_ids"], r["da_word"], n, scope)
                r[f"delta_{name}_{scope}"] = m - base
        del r["ctx_ids"]
    g = np.random.default_rng(SEED); summ = {}
    for pl in ("sv", "da"):
        for ctx in ("neutral", "after_danish", "after_swedish"):
            sub = [r for r in rows if r["prompt_lang"] == pl and r["context"] == ctx]
            def stat(key):
                v = np.array([r[key] for r in sub]); bs = [v[g.integers(0, len(v), len(v))].mean() for _ in range(2000)]
                return {"mean": float(v.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            summ[f"{pl}_prompt/{ctx}"] = {k: stat(k) for k in sub[0] if k.startswith(("delta_", "margin_", "act_"))}
    return {"control_neuron": [23, control], "strength_N981": float(strength[981]), "strength_control": float(strength[control]),
            "summary": summ, "rows": rows}


def lang_eval(model, tok, system, t1, mk, sets=("T1_sv", "MK_sv", "MK_da", "MK_en")):
    out = {}
    if "T1_sv" in sets:
        out["T1_sv"] = response_language(model, tok, [p["sv"] for p in t1], "sv", system=system)
    if "MK_sv" in sets:
        out["MK_sv"] = response_language(model, tok, [r["sv"] for r in mk], "sv", system=system)
    if "MK_da" in sets:
        out["MK_da"] = response_language(model, tok, [r["da"] for r in mk[:150]], "da", system=system)
    if "MK_en" in sets:
        out["MK_en"] = response_language(model, tok, [r["en"] for r in mk[:150]], "en", system=system)
    return out


def main():
    os.makedirs(OUT, exist_ok=True); torch.manual_seed(SEED)
    model, tok = load(); t1 = t1_prompts(); _, mk = mkqa(SEED)
    def cached(name, fn):
        p = f"{OUT}/{name}.json"
        if not os.path.exists(p):
            json.dump(fn(), open(p, "w"), ensure_ascii=False, indent=1); free_memory()
        return json.load(open(p))
    fp = cached("E1_fixed_prefix", lambda: fixed_prefix(model, tok))
    print("control neuron", fp["control_neuron"], "strength", round(fp["strength_N981"], 2), round(fp["strength_control"], 2))
    for k, v in fp["summary"].items():
        print(f"{k:24s} base margin {v['margin_base']['mean']:+.2f} | act981 {v['act_N981']['mean']:+.1f} act1821 {v['act_N1821']['mean']:+.1f} | "
              + " | ".join(f"{n} {v[f'delta_{n}_position']['mean']:+.2f} [{v[f'delta_{n}_position']['ci'][0]:+.2f},{v[f'delta_{n}_position']['ci'][1]:+.2f}]" for n in ("N981", "N1821", "control"))
              + f" | N981 everywhere {v['delta_N981_everywhere']['mean']:+.2f}", flush=True)
    p2 = cached("E2_same_language_prompt", lambda: lang_eval(model, tok, SAME_LANG, t1, mk))
    print("E2 same-language prompt:", {k: (round(v["rate"], 3), v["langs"]) for k, v in p2.items()}, flush=True)
    def e3():
        out = {"B0_danish_instruction": lang_eval(model, tok, DA_SYS, t1, mk, sets=("MK_da",))}
        with edits.scale_neurons(model, [(23, 981)], 0):
            out["W0a_danish_instruction"] = lang_eval(model, tok, DA_SYS, t1, mk, sets=("MK_da",))
            out["W0a_same_language_prompt"] = lang_eval(model, tok, SAME_LANG, t1, mk, sets=("MK_da",))
        return out
    e = cached("E3_danish_recovery", e3)
    print("E3:", {k: (round(v["MK_da"]["rate"], 3), v["MK_da"]["langs"]) for k, v in e.items()}, flush=True)
    print("EXTRAS DONE", flush=True)


if __name__ == "__main__":
    main()
