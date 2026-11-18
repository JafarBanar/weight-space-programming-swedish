"""Exploration, ZERO forward passes: refine the direction with translation pairs and
read which tokens the outlier neurons promote."""
import sys, json, torch
sys.path.insert(0, "src")
from common import load
model, tok = load()
E = model.get_input_embeddings().weight.detach().float().cpu()
gamma = model.model.norm.weight.detach().float().cpu()
PAIRS = [("är", "er"), ("och", "og"), ("inte", "ikke"), ("jag", "jeg"), ("av", "af"), ("från", "fra"),
         ("också", "også"), ("att", "at"), ("mycket", "meget"), ("hur", "hvordan"), ("vad", "hvad"),
         ("när", "når"), ("där", "der"), ("efter", "etter"), ("bara", "kun"), ("vad", "hva"), ("att", "å")]
ids = []
for a, b in PAIRS:
    ia, ib = tok.encode(" " + a, add_special_tokens=False), tok.encode(" " + b, add_special_tokens=False)
    if len(ia) == 1 and len(ib) == 1:
        ids.append((ia[0], ib[0]))
print(len(ids), "single-token pairs")
diff = torch.stack([E[a] - E[b] for a, b in ids])
d = diff.mean(0) * gamma; d = d / d.norm()
cos = torch.nn.functional.cosine_similarity(diff * gamma, d[None], dim=1)
print("cos(pair diff, mean dir):", [round(c, 2) for c in cos.tolist()])
z_all = []
for l in range(len(model.model.layers)):
    W = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
    s = d @ W
    z = (s - s.mean()) / s.std()
    z_all.append(z)
    top = z.abs().topk(3)
    print(l, "std", round(s.std().item(), 4), "top |z|:", [(i, round(z[i].item(), 1)) for i in top.indices.tolist()])
Z = torch.stack(z_all)
flat = Z.abs().flatten().topk(12)
print("\nGlobal top neurons by |z| (layer, neuron, z):")
for v, idx in zip(flat.values.tolist(), flat.indices.tolist()):
    l, j = divmod(idx, Z.shape[1])
    w = model.model.layers[l].mlp.down_proj.weight.detach().float().cpu()[:, j]
    logit = E @ (w * gamma)            # direct effect on output logits, up to the per-position RMS scale
    up = [tok.decode([i]) for i in logit.topk(14).indices.tolist()]
    down = [tok.decode([i]) for i in (-logit).topk(14).indices.tolist()]
    print(f"L{l} N{j} z={Z[l, j]:+.1f}\n   promotes : {up}\n   suppresses: {down}")
