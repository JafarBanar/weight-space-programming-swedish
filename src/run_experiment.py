"""Main experiment. Stages are cached in results/, so the script can be re-run or resumed.

  python src/run_experiment.py            # everything except LoRA (see src/run_lora.py)

Stage 1  read weights (zero forward passes) -> direction, neuron z-scores, W0 neurons
Stage 2  development only: tune W1 and S1 on dev-val; check neuron activations on dev-fit
Stage 3  held-out test: B0, P1, W0, W1, S1 and random-neuron controls
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate, lang_of, SEED, MODEL_ID, MODEL_REV
from prompts import dev_prompts, test_sets
from evaluate import response_language, perplexity, minimal_pair_accuracy
from minimal_pairs import build as build_pairs
import edits, fetch_data

R = "results"
SV_SYS = "You are a helpful assistant. Svara alltid på svenska."
DA_SYS = "You are a helpful assistant. Svar altid på dansk."
CALLS = {"dev_prompts_run": 0}          # how much model feedback the tuned methods consumed


def cached(name, fn):
    path = f"{R}/{name}.json"
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time(); out = fn()
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[{name}] done in {time.time() - t:.0f}s", flush=True)
    return out


def full_eval(model, tok, T, ud, pairs, system=None, light=False):
    """Held-out evaluation. light=True runs only the Swedish target sets and Swedish perplexity."""
    out = {}
    for name, key, target in [("sv_A", "sv", "sv"), ("sv_B", "sv", "sv"), ("en_B", "en", "en"), ("da_B", "da", "da")]:
        if light and name in ("en_B", "da_B"):
            continue
        out[name] = response_language(model, tok, [p[key] for p in T[name]], target, system=system)
    out["ppl"] = {k: perplexity(model, tok, v) for k, v in ud.items() if not light or k == "sv"}
    if not light:
        out["minimal_pairs"] = minimal_pair_accuracy(model, tok, pairs)
    return out


def dev_eval(model, tok, val, system=None):
    sv = response_language(model, tok, [p["sv"] for p in val], "sv", system=system)
    en = response_language(model, tok, [p["en"] for p in val[:40]], "en", system=system)
    CALLS["dev_prompts_run"] += len(val) + 40
    return {"sv_rate": sv["rate"], "sv_langs": sv["langs"], "en_rate": en["rate"],
            "degenerate": sv["degenerate"], "marker_share": sv["marker_share"]}


def select(grid, base, size_key):
    """Pre-registered rule: best dev-val Swedish rate subject to the English and degeneracy
    constraints; ties go to the smaller intervention."""
    ok = [g for g in grid if g["en_rate"] >= base["en_rate"] - 0.025 and g["degenerate"] <= base["degenerate"] + 0.025]
    return sorted(ok, key=lambda g: (-g["sv_rate"], size_key(g)))[0] if ok else None


@torch.no_grad()
def last_token_residuals(model, tok, prompts, layer_ids, batch_size=16):
    """Mean residual stream at the final prompt position, after each layer in layer_ids."""
    sums = {l: 0 for l in layer_ids}
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        for l in layer_ids:
            sums[l] = sums[l] + hs[l + 1][:, -1].float().sum(0).cpu()
    return {l: s / len(prompts) for l, s in sums.items()}


@torch.no_grad()
def neuron_activation_check(model, tok, fit, neurons):
    """Post-hoc mechanism check on dev-fit: do the neurons picked from the weights fire more
    while the model is writing a Danish/Norwegian answer than while it writes a Swedish one?"""
    acts = {n: [] for n in neurons}; store = {}
    hooks = []
    for l, j in neurons:
        def pre(m, inp, l=l, j=j):
            store.setdefault((l, j), []).append(inp[0][:, :, j].float().cpu())
        hooks.append(edits.layers(model)[l].mlp.down_proj.register_forward_pre_hook(pre))
    rows = []
    for i in range(0, len(fit), 16):
        store.clear()
        prompts = [chat(tok, p["sv"]) for p in fit[i:i + 16]]
        outs = generate(model, tok, prompts, max_new_tokens=32, batch_size=16)
        for b, o in enumerate(outs):
            row = {"lang": lang_of(o)}
            for n in neurons:
                steps = store[n][1:]                       # generated positions only (first call = prompt)
                row[f"L{n[0]}N{n[1]}"] = float(torch.stack([s[b, -1] for s in steps]).mean())
            rows.append(row)
    for h in hooks:
        h.remove()
    return rows


def main():
    os.makedirs(R, exist_ok=True)
    torch.manual_seed(SEED)
    model, tok = load()
    fit, val = dev_prompts(SEED)
    T, ud, pairs = test_sets(), fetch_data.main(), build_pairs(seed=SEED)
    n_mlp = edits.layers(model)[0].mlp.down_proj.weight.shape[1]
    n_params = sum(p.numel() for p in model.parameters())

    # ---------------- Stage 1: read the weights, zero forward passes ----------------
    d = edits.weight_direction(model, tok)
    Z = edits.neuron_zscores(model, d)
    W0 = [n for n in edits.top_danish_neurons(Z, 16) if Z[n[0], n[1]] < -6]
    top = edits.top_danish_neurons(Z, 128)
    json.dump({"model": MODEL_ID, "revision": MODEL_REV, "seed": SEED, "n_params": n_params,
               "W0_neurons": W0, "W0_z": [float(Z[l, j]) for l, j in W0],
               "top128": [(l, j, float(Z[l, j])) for l, j in top],
               "z_abs_max_per_layer": Z.abs().max(1).values.tolist(),
               "z_flat_sample": Z.flatten()[torch.randperm(Z.numel(), generator=torch.Generator().manual_seed(SEED))[:20000]].tolist()},
              open(f"{R}/stage1_weight_reading.json", "w"), indent=1)
    print("W0 neurons (z < -6):", W0, flush=True)
    assert sorted(W0) == [(22, 1821), (23, 981)], "pre-registered W0 neurons not reproduced"

    # ---------------- Stage 2: development (dev-fit / dev-val only) ----------------
    base_dev = cached("dev_baseline", lambda: dev_eval(model, tok, val))
    print("dev baseline:", base_dev, flush=True)

    def tune_w1():
        grid = []
        for K in (2, 8, 32, 128):
            for c in (0, -1, -2, -4) + ((-8,) if K == 2 else ()):
                with edits.scale_neurons(model, top[:K], c):
                    g = dev_eval(model, tok, val)
                g.update(K=K, c=c); grid.append(g)
                print("W1", g, flush=True)
        return {"grid": grid, "dev_prompts_run": len(grid) * 120}
    w1 = cached("dev_tune_W1", tune_w1)
    w1_sel = select(w1["grid"], base_dev, lambda g: (g["K"], abs(g["c"])))
    print("W1 selected:", w1_sel, flush=True)

    steer_layers = (8, 14, 20)
    def steer_vectors():
        sv = last_token_residuals(model, tok, [chat(tok, p["sv"], system=SV_SYS) for p in fit], steer_layers)
        da = last_token_residuals(model, tok, [chat(tok, p["sv"], system=DA_SYS) for p in fit], steer_layers)
        return {str(l): (sv[l] - da[l]).tolist() for l in steer_layers}
    vecs = {int(k): torch.tensor(v) for k, v in cached("dev_steer_vectors", steer_vectors).items()}

    def tune_s1():
        grid = []
        for l in steer_layers:
            for a in (1, 2, 4):
                with edits.steer(model, l, a * vecs[l]):
                    g = dev_eval(model, tok, val)
                g.update(layer=l, strength=a); grid.append(g)
                print("S1", g, flush=True)
        return {"grid": grid, "dev_prompts_run": 160 + len(grid) * 120}
    s1 = cached("dev_tune_S1", tune_s1)
    s1_sel = select(s1["grid"], base_dev, lambda g: (g["strength"], g["layer"]))
    print("S1 selected:", s1_sel, flush=True)

    cached("dev_neuron_activations", lambda: neuron_activation_check(
        model, tok, fit, [tuple(n) for n in W0] + edits.random_neurons(W0, n_mlp, 123)))

    # ---------------- Stage 3: held-out test ----------------
    conds = {"B0_baseline": (edits.no_edit, {}, False),
             "P1_prompt": (edits.no_edit, {"system": SV_SYS}, False),
             "W0_weights_only": (lambda: edits.scale_neurons(model, W0, 0), {}, False)}
    w1_same = w1_sel is not None and w1_sel["K"] == 2 and w1_sel["c"] == 0
    if w1_sel is not None and not w1_same:
        conds["W1_tuned"] = (lambda: edits.scale_neurons(model, top[:w1_sel["K"]], w1_sel["c"]), {}, False)
    if s1_sel is not None:
        conds["S1_steering"] = (lambda: edits.steer(model, s1_sel["layer"], s1_sel["strength"] * vecs[s1_sel["layer"]]), {}, False)
    for s in range(5):
        conds[f"R0_random_seed{s}"] = (lambda s=s: edits.scale_neurons(model, edits.random_neurons(W0, n_mlp, s), 0), {}, True)
        if "W1_tuned" in conds:
            conds[f"R1_random_seed{s}"] = (lambda s=s: edits.scale_neurons(
                model, edits.random_neurons(top[:w1_sel["K"]], n_mlp, 100 + s), w1_sel["c"]), {}, True)

    for name, (ctx, kw, light) in conds.items():
        def run(ctx=ctx, kw=kw, light=light):
            with ctx():
                return full_eval(model, tok, T, ud, pairs, light=light, **kw)
        r = cached(f"test_{name}", run)
        print(f"{name:22s} sv_A={r['sv_A']['rate']:.3f} sv_B={r['sv_B']['rate']:.3f} "
              + (f"en={r['en_B']['rate']:.3f} da={r['da_B']['rate']:.3f} mp={r['minimal_pairs']['accuracy']:.3f} " if not light else "")
              + f"ppl_sv={r['ppl']['sv']:.1f}", flush=True)

    json.dump({"W0": {"neurons": W0, "c": 0, "weights_changed": len(W0) * d.numel(), "forward_passes_used": 0},
               "W1": {"selected": w1_sel, "same_as_W0": w1_same, "dev_prompts_run": w1["dev_prompts_run"],
                      "weights_changed": None if w1_sel is None else w1_sel["K"] * d.numel()},
               "S1": {"selected": s1_sel, "dev_prompts_run": s1["dev_prompts_run"]},
               "n_params": n_params, "device": str(model.device), "dtype": str(next(model.parameters()).dtype),
               "torch": torch.__version__}, open(f"{R}/selected_configs.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
