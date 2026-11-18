"""Zero-forward-pass robustness check of the weight reading.
Does the choice of the 17 function-word pairs matter? Re-derive the direction from random
subsets of the pairs and record how often the two W0 neurons are still the two strongest
Danish/Norwegian-writing neurons in the whole model.  Output: results/robustness_weight_reading.json
"""
import json, os, random, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, SEED
import edits


@torch.no_grad()
def main(n_subsets=200, subset_size=8):
    model, tok = load()
    E = model.get_input_embeddings().weight.detach().float().cpu()
    gamma = model.model.norm.weight.detach().float().cpu()
    diffs, kept = [], []
    for a, b in edits.PAIRS:
        ia, ib = tok.encode(" " + a, add_special_tokens=False), tok.encode(" " + b, add_special_tokens=False)
        if len(ia) == 1 and len(ib) == 1:
            diffs.append(E[ia[0]] - E[ib[0]]); kept.append((a, b))
    D = torch.stack(diffs) * gamma                                           # [pairs, d]
    W = [blk.mlp.down_proj.weight.detach().float().cpu() for blk in edits.layers(model)]
    S = torch.stack([D @ w for w in W])                                      # [layers, pairs, neurons]
    target = {(22, 1821), (23, 981)}
    r = random.Random(SEED); both, ranks, zs = 0, [], []
    for _ in range(n_subsets):
        idx = r.sample(range(len(diffs)), subset_size)
        norm = D[idx].mean(0).norm()
        s = S[:, idx].mean(1) / norm
        Z = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True)
        order = Z.flatten().argsort().tolist()
        top2 = {divmod(i, Z.shape[1]) for i in order[:2]}
        both += top2 == target
        ranks.append([order.index(l * Z.shape[1] + j) + 1 for l, j in sorted(target)])
        zs.append([float(Z[l, j]) for l, j in sorted(target)])
    # single-pair directions: the weakest possible reading
    single = []
    for k, (a, b) in enumerate(kept):
        s = S[:, k] / D[k].norm(); Z = (s - s.mean(1, keepdim=True)) / s.std(1, keepdim=True)
        single.append({"pair": f"{a}/{b}", "z_L22N1821": float(Z[22, 1821]), "z_L23N981": float(Z[23, 981])})
    out = {"n_subsets": n_subsets, "subset_size": subset_size, "n_pairs": len(diffs),
           "both_W0_neurons_are_top2": both / n_subsets,
           "worst_rank": [max(x[0] for x in ranks), max(x[1] for x in ranks)],
           "z_mean": [sum(x[0] for x in zs) / n_subsets, sum(x[1] for x in zs) / n_subsets],
           "z_weakest": [max(x[0] for x in zs), max(x[1] for x in zs)], "single_pair": single}
    json.dump(out, open("results/robustness_weight_reading.json", "w"), indent=1, ensure_ascii=False)
    print({k: v for k, v in out.items() if k != "single_pair"})
    for s_ in single:
        print(s_)


if __name__ == "__main__":
    main()
