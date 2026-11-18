"""Post-hoc content check (not pre-registered): fact accuracy and answer language on the notice
questions, for the unedited model, the prompt control, the weights-only edit, random-neuron
controls and activation steering.   Output: results/notices_<condition>.json"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate, lang_of, SEED
import edits, notices
from run_experiment import SV_SYS, R

VARIANTS = {"sv_sv": "sv", "en_sv": "sv", "en_en": "en"}


def notice_eval(model, tok, system=None):
    items = notices.build(SEED); out = {}
    for v, target in VARIANTS.items():
        outs = generate(model, tok, [chat(tok, it[v], system=system) for it in items], max_new_tokens=40)
        fact = [int(notices.fact_correct(o, it)) for o, it in zip(outs, items)]
        langs = [lang_of(notices.strip_codes(o)) for o in outs]
        lang = [int(l == target) for l in langs]
        out[v] = {"fact": sum(fact) / len(items), "language": sum(lang) / len(items),
                  "joint": sum(f * l for f, l in zip(fact, lang)) / len(items),
                  "fact_hit": fact, "lang_hit": lang, "langs": langs, "outputs": outs}
    return out


def main():
    model, tok = load()
    n_mlp = edits.layers(model)[0].mlp.down_proj.weight.shape[1]
    W0 = [tuple(n) for n in json.load(open(f"{R}/stage1_weight_reading.json"))["W0_neurons"]]
    sel = json.load(open(f"{R}/selected_configs.json"))["S1"]["selected"]
    vecs = {int(k): torch.tensor(v) for k, v in json.load(open(f"{R}/dev_steer_vectors.json")).items()}
    conds = {"B0_baseline": (edits.no_edit, None), "P1_prompt": (edits.no_edit, SV_SYS),
             "W0_weights_only": (lambda: edits.scale_neurons(model, W0, 0), None),
             "W0a_only_L23N981": (lambda: edits.scale_neurons(model, [(23, 981)], 0), None)}
    if sel:
        conds["S1_steering"] = (lambda: edits.steer(model, sel["layer"], sel["strength"] * vecs[sel["layer"]]), None)
    for s in range(3):
        conds[f"R0_random_seed{s}"] = (lambda s=s: edits.scale_neurons(model, edits.random_neurons(W0, n_mlp, s), 0), None)
    for name, (ctx, system) in conds.items():
        path = f"{R}/notices_{name}.json"
        if not os.path.exists(path):
            with ctx():
                json.dump(notice_eval(model, tok, system), open(path, "w"), ensure_ascii=False, indent=1)
        r = json.load(open(path))
        print(f"{name:20s} " + "  ".join(f"{v}: fact {r[v]['fact']:.2f} lang {r[v]['language']:.2f} joint {r[v]['joint']:.2f}" for v in VARIANTS), flush=True)


if __name__ == "__main__":
    main()
