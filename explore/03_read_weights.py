"""Exploration with ZERO forward passes: read the weights only.
1) Build a Swedish-vs-Danish/Norwegian direction from the (tied) token embedding matrix.
2) Score every MLP neuron by how its output weights line up with that direction.
"""
import sys, json, torch
sys.path.insert(0, "src")
from common import load
model, tok = load()
E = model.get_input_embeddings().weight.detach().float().cpu()          # [vocab, d]; tied with the unembedding
gamma = model.model.norm.weight.detach().float().cpu()                   # final RMSNorm gain
print("tied:", model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())

SV = ["är", "och", "inte", "jag", "av", "från", "också", "att", "mycket", "något", "bara", "efter", "eller", "hur", "vad", "när", "några", "själv", "måste", "där"]
DN = ["er", "og", "ikke", "jeg", "af", "fra", "også", "at", "meget", "noget", "kun", "etter", "hvordan", "hvad", "når", "nogle", "selv", "må", "der", "å", "mye", "noe", "hva", "bare"]
def single(words):
    keep = {}
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            keep[w] = ids[0]
    return keep
sv, dn = single(SV), single(DN)
print("single-token sv:", list(sv)); print("single-token da/no:", list(dn))
d = (E[list(sv.values())].mean(0) - E[list(dn.values())].mean(0)) * gamma
d = d / d.norm()
# sanity: which tokens does the direction itself point to / away from?
proj = E @ (d / gamma)   # not exact logit, just for a look
top = proj.topk(25).indices.tolist(); bot = (-proj).topk(25).indices.tolist()
print("direction + :", [tok.decode([i]) for i in top]); print("direction - :", [tok.decode([i]) for i in bot])

L = len(model.model.layers)
rows = []
for l in range(L):
    W = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()   # [d, d_mlp]
    s = d @ W                                                               # signed write along d per neuron
    c = s / W.norm(dim=0)
    rows.append({"layer": l, "s_std": s.std().item(), "s_min": s.min().item(), "s_max": s.max().item(),
                 "cos_min": c.min().item(), "cos_max": c.max().item(),
                 "n_neg_6sd": int((s < -6 * s.std()).sum()), "n_pos_6sd": int((s > 6 * s.std()).sum())})
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
json.dump({"sv_tokens": sv, "dn_tokens": dn, "layers": rows}, open("explore/03_read_weights.json", "w"), ensure_ascii=False, indent=1)
