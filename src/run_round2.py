"""Round 2: confirmation on fresh prompts, survey across models, and a larger steering search.
See PREREGISTRATION_ROUND2.md (committed before this script was run on any test data).

  python src/run_round2.py qwen2.5-1.5b      # confirmatory test of the single-neuron edit + recipe
  python src/run_round2.py qwen2.5-0.5b      # survey models: same frozen recipe
  python src/run_round2.py eurollm-1.7b
  python src/run_round2.py qwen3-1.7b
  python src/run_round2.py qwen2.5-1.5b steering   # larger steering search, then test the winner

Stages are cached in results/r2/<model>/.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate, lang_of, SEED, MODEL_ID, MODEL_REV
from prompts import dev_prompts
from prompts2 import t1_prompts, mkqa
from evaluate import response_language, perplexity, minimal_pair_accuracy
from minimal_pairs import build as build_pairs
from native_pairs import build as build_native
from download_models import SURVEY_MODELS
import edits, fetch_data

MODELS = {"qwen2.5-1.5b": (MODEL_ID, MODEL_REV),
          "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", SURVEY_MODELS["Qwen/Qwen2.5-0.5B-Instruct"]),
          "eurollm-1.7b": ("utter-project/EuroLLM-1.7B-Instruct", SURVEY_MODELS["utter-project/EuroLLM-1.7B-Instruct"]),
          "qwen3-1.7b": ("Qwen/Qwen3-1.7B", SURVEY_MODELS["Qwen/Qwen3-1.7B"])}
SV_SYS = "You are a helpful assistant. Svara alltid på svenska."
Z_STRONG = -6.0


def cached(path, fn):
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time(); out = fn()
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[{os.path.basename(path)}] done in {time.time() - t:.0f}s", flush=True)
    return out


@torch.no_grad()
def answer_activations(model, tok, prompts, neurons, n_new=32, bs=16):
    """Generate an answer to each prompt; return its language and the mean activation of each
    neuron (the input to mlp.down_proj) over the generated positions."""
    store, hooks, rows = {}, [], []
    for l, j in neurons:
        hooks.append(edits.layers(model)[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, inp, l=l, j=j: store.setdefault((l, j), []).append(inp[0][:, :, j].float().cpu())))
    for i in range(0, len(prompts), bs):
        store.clear()
        outs = generate(model, tok, prompts[i:i + bs], max_new_tokens=n_new, batch_size=bs)
        for b, o in enumerate(outs):
            row = {"lang": lang_of(o), "text": o}
            for n in neurons:
                row[f"L{n[0]}N{n[1]}"] = float(torch.stack([s[b, -1] for s in store[n][1:]]).mean())
            rows.append(row)
    for h in hooks:
        h.remove()
    return rows


def recipe(model, tok, fit, mk_dev, out_dir):
    """The frozen, mechanical recipe. Step 1 reads weights only; step 2 uses one batch of answers."""
    d = edits.weight_direction(model, tok)
    Z = edits.neuron_zscores(model, d)
    top = edits.top_danish_neurons(Z, 16)
    strong = [n for n in top if Z[n[0], n[1]] < Z_STRONG]
    cands = strong if strong else top[:2]
    read = {"tier": "strong" if strong else "weak", "candidates": cands, "z": [float(Z[l, j]) for l, j in cands],
            "top8": [(l, j, float(Z[l, j])) for l, j in top[:8]], "n_neurons": int(Z.numel()),
            "most_positive_z": float(Z.max())}
    json.dump(read, open(f"{out_dir}/reading.json", "w"), indent=1)
    print("reading:", read["tier"], cands, [round(z, 1) for z in read["z"]], flush=True)

    def sign():
        ps = [chat(tok, p["sv"]) for p in fit[:40]] + [chat(tok, r["sv"]) for r in mk_dev[:60]] + [chat(tok, r["da"]) for r in mk_dev[:60]]
        return answer_activations(model, tok, ps, [tuple(c) for c in cands])
    rows = cached(f"{out_dir}/sign_step.json", sign)
    dan = [r for r in rows if r["lang"] in ("da", "no")]; swe = [r for r in rows if r["lang"] == "sv"]
    decision, delete = [], []
    for l, j in cands:
        k = f"L{l}N{j}"
        a_da = sum(r[k] for r in dan) / max(1, len(dan)); a_sv = sum(r[k] for r in swe) / max(1, len(swe))
        dele = bool(dan) and bool(swe) and a_da > 0 and a_sv > -0.25 * a_da
        decision.append({"neuron": [l, j], "act_danish_answers": a_da, "act_swedish_answers": a_sv, "delete": dele})
        if dele:
            delete.append((l, j))
    res = {"n_danish_answers": len(dan), "n_swedish_answers": len(swe), "decision": decision, "delete": delete}
    json.dump(res, open(f"{out_dir}/recipe_decision.json", "w"), indent=1)
    print("recipe decision:", json.dumps(decision), flush=True)
    return read, delete


def evaluate_all(model, tok, D, system=None, light=False):
    out = {"T1_sv": response_language(model, tok, [p["sv"] for p in D["t1"]], "sv", system=system),
           "MK_sv": response_language(model, tok, [r["sv"] for r in D["mk"]], "sv", system=system)}
    if light:
        return out
    out["MK_da"] = response_language(model, tok, [r["da"] for r in D["mk"][:150]], "da", system=system)
    out["MK_en"] = response_language(model, tok, [r["en"] for r in D["mk"][:150]], "en", system=system)
    out["T1_en"] = response_language(model, tok, [p["en"] for p in D["t1"] if D["t1"].index(p) % 8 < 4], "en", system=system)
    out["ppl"] = {k: perplexity(model, tok, v) for k, v in D["ud"].items()}
    out["pairs_synthetic"] = minimal_pair_accuracy(model, tok, D["pairs"])
    out["pairs_native"] = minimal_pair_accuracy(model, tok, D["native"])
    return out


def show(name, r):
    def wl(x):
        return f"{x['rate']:.3f} {x['langs']}"
    print(f"{name:14s} T1 {wl(r['T1_sv'])} | MKQA-sv {wl(r['MK_sv'])}" +
          (f" | da->da {r['MK_da']['rate']:.3f} | en->en {r['MK_en']['rate']:.3f}/{r['T1_en']['rate']:.3f} | ppl_sv {r['ppl']['sv']:.1f} "
           f"| pairs {r['pairs_synthetic']['accuracy']:.3f}/{r['pairs_native']['accuracy']:.3f}" if "ppl" in r else ""), flush=True)


@torch.no_grad()
def answer_state_means(model, tok, prompts, layer_ids, n_new=32, bs=16):
    """For steering: per-answer mean residual over the generated positions, after each layer."""
    rows = []
    for i in range(0, len(prompts), bs):
        outs = generate(model, tok, prompts[i:i + bs], max_new_tokens=n_new, batch_size=bs)
        for p, o in zip(prompts[i:i + bs], outs):
            n_p = len(tok(p)["input_ids"])
            enc = tok(p + o, return_tensors="pt").to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            if enc["input_ids"].shape[1] > n_p:
                rows.append({"lang": lang_of(o), "h": {l: hs[l + 1][0, n_p:].float().mean(0).cpu() for l in layer_ids}})
    return rows


def steering_search(model, tok, fit, val, mk_dev, out_dir):
    """A larger search than round 1: three ways of building the direction x six layers x three strengths."""
    layer_ids = (6, 10, 14, 18, 22, 26)
    base = cached(f"{out_dir}/steer_dev_baseline.json", lambda: dev_eval(model, tok, val))

    def vectors():
        v = {}
        old = json.load(open("results/dev_steer_vectors.json"))                       # recipe A (round 1), layers 8/14/20
        v["A_instruction"] = {str(l): old[min(old, key=lambda k: abs(int(k) - l))] for l in layer_ids}
        sw = answer_state_means(model, tok, [chat(tok, p["sv"]) for p in fit] + [chat(tok, r["sv"]) for r in mk_dev[:60]], layer_ids)
        da = answer_state_means(model, tok, [chat(tok, r["da"]) for r in mk_dev[:60]], layer_ids)
        def mean(rows, l):
            return torch.stack([r["h"][l] for r in rows]).mean(0)
        sv_ok = [r for r in sw if r["lang"] == "sv"]; drift = [r for r in sw if r["lang"] in ("da", "no")]
        da_ok = [r for r in da if r["lang"] in ("da", "no")]
        v["B_language"] = {str(l): (mean(sv_ok, l) - mean(da_ok, l)).tolist() for l in layer_ids}    # Swedish answers minus Danish answers
        v["C_drift"] = {str(l): (mean(sv_ok, l) - mean(drift, l)).tolist() for l in layer_ids}       # Swedish answers minus drifting answers
        v["_counts"] = {"swedish": len(sv_ok), "danish": len(da_ok), "drift": len(drift)}
        return v
    vecs = cached(f"{out_dir}/steer_vectors.json", vectors)

    def search():
        grid = []
        for rec in ("A_instruction", "B_language", "C_drift"):
            for l in layer_ids:
                for a in (1, 2, 4):
                    with edits.steer(model, l, a * torch.tensor(vecs[rec][str(l)])):
                        g = dev_eval(model, tok, val)
                    g.update(recipe=rec, layer=l, strength=a); grid.append(g)
                    print("S2", {k: g[k] for k in ("recipe", "layer", "strength", "sv_rate", "en_rate", "degenerate")}, flush=True)
        return {"grid": grid, "dev_prompts_run": 280 + len(grid) * 120}
    s = cached(f"{out_dir}/steer_search.json", search)
    ok = [g for g in s["grid"] if g["en_rate"] >= base["en_rate"] - 0.025 and g["degenerate"] <= base["degenerate"] + 0.025]
    sel = sorted(ok, key=lambda g: (-g["sv_rate"], g["strength"], g["layer"]))[0] if ok else None
    json.dump({"baseline": base, "selected": sel, "configs": len(s["grid"]), "dev_prompts_run": s["dev_prompts_run"]},
              open(f"{out_dir}/steer_selected.json", "w"), indent=1)
    print("S2 selected:", sel, flush=True)
    return sel, vecs


def dev_eval(model, tok, val):
    sv = response_language(model, tok, [p["sv"] for p in val], "sv")
    en = response_language(model, tok, [p["en"] for p in val[:40]], "en")
    return {"sv_rate": sv["rate"], "sv_langs": sv["langs"], "en_rate": en["rate"], "degenerate": sv["degenerate"]}


def main():
    key = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "main"
    out_dir = f"results/r2/{key}"; os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED)
    model, tok = load(*MODELS[key])
    fit, val = dev_prompts(SEED)
    mk_dev, mk_test = mkqa(SEED)
    D = {"t1": t1_prompts(), "mk": mk_test, "ud": fetch_data.main(), "pairs": build_pairs(seed=SEED), "native": build_native()}
    n_mlp = edits.layers(model)[0].mlp.down_proj.weight.shape[1]

    if mode == "steering":
        sel, vecs = steering_search(model, tok, fit, val, mk_dev, out_dir)
        if sel:
            v = torch.tensor(vecs[sel["recipe"]][str(sel["layer"])])
            def run():
                with edits.steer(model, sel["layer"], sel["strength"] * v):
                    return evaluate_all(model, tok, D)
            show("S2_steering", cached(f"{out_dir}/test_S2_steering.json", run))
        return

    read, delete = recipe(model, tok, fit, mk_dev, out_dir)
    conds = {"B0": (edits.no_edit, None, False)}
    if key == "qwen2.5-1.5b":                                    # confirmatory: edits fixed in the pre-registration
        conds["W0a"] = (lambda: edits.scale_neurons(model, [(23, 981)], 0), None, False)
        conds["W0"] = (lambda: edits.scale_neurons(model, [(23, 981), (22, 1821)], 0), None, False)
        ref = [(23, 981)]
    else:
        ref = delete
    if delete and (key != "qwen2.5-1.5b" or sorted(delete) != [(23, 981)]):
        conds["RECIPE"] = (lambda: edits.scale_neurons(model, delete, 0), None, False)
    conds["P1"] = (edits.no_edit, SV_SYS, key != "qwen2.5-1.5b")
    for s in range(2):
        if ref:
            conds[f"R_seed{s}"] = (lambda s=s: edits.scale_neurons(model, edits.random_neurons(ref, n_mlp, 200 + s), 0), None, True)
    for name, (ctx, system, light) in conds.items():
        def run(ctx=ctx, system=system, light=light):
            with ctx():
                return evaluate_all(model, tok, D, system=system, light=light)
        show(name, cached(f"{out_dir}/test_{name}.json", run))
    print("DONE", key, flush=True)


if __name__ == "__main__":
    main()
