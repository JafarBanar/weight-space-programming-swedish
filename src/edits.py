"""Weight edits and their controls. Every edit is applied in place and can be undone exactly.

Edit family (after Experiment 8 of the article, which rewrote a few MLP output columns):
for a chosen set of MLP neurons, multiply the neuron's output column (a column of
`mlp.down_proj.weight`) by a constant c.  c = 0 deletes the neuron's contribution,
c < 0 reverses it. No bias terms are added and no other weights are touched.
"""
import contextlib
import torch
from peft import PeftModel  # noqa: F401  (imported so peft problems surface early)

# Swedish / Danish-Norwegian function-word translation pairs (chosen before any model run).
PAIRS = [("är", "er"), ("och", "og"), ("inte", "ikke"), ("jag", "jeg"), ("av", "af"), ("från", "fra"),
         ("också", "også"), ("att", "at"), ("mycket", "meget"), ("hur", "hvordan"), ("vad", "hvad"),
         ("när", "når"), ("där", "der"), ("efter", "etter"), ("bara", "kun"), ("vad", "hva"), ("att", "å")]


def layers(model):
    return model.model.layers


@torch.no_grad()
def weight_direction(model, tok):
    """Swedish-minus-Danish/Norwegian direction read from the tied embedding matrix alone
    (no forward pass): mean embedding difference over single-token translation pairs,
    mapped through the final RMSNorm gain, unit norm."""
    E = model.get_input_embeddings().weight.detach().float().cpu()
    gamma = model.model.norm.weight.detach().float().cpu()
    diffs = []
    for a, b in PAIRS:
        ia, ib = tok.encode(" " + a, add_special_tokens=False), tok.encode(" " + b, add_special_tokens=False)
        if len(ia) == 1 and len(ib) == 1:
            diffs.append(E[ia[0]] - E[ib[0]])
    d = torch.stack(diffs).mean(0) * gamma
    return d / d.norm()


@torch.no_grad()
def neuron_zscores(model, d):
    """z[l, j]: how strongly neuron j of layer l writes along d, standardised within its layer.
    Strongly negative = writes towards Danish/Norwegian and away from Swedish."""
    Z = []
    for blk in layers(model):
        s = d @ blk.mlp.down_proj.weight.detach().float().cpu()
        Z.append((s - s.mean()) / s.std())
    return torch.stack(Z)


def top_danish_neurons(Z, k):
    """The k most negative-z neurons in the whole model, as (layer, neuron) pairs."""
    idx = Z.flatten().topk(k, largest=False).indices.tolist()
    return [divmod(i, Z.shape[1]) for i in idx]


def random_neurons(like, n_mlp, seed):
    """Control: same layers and same count as `like`, but neurons drawn at random."""
    g = torch.Generator().manual_seed(seed)
    return [(l, int(torch.randint(0, n_mlp, (1,), generator=g))) for l, _ in like]


@contextlib.contextmanager
def scale_neurons(model, neurons, c):
    """Multiply the output column of each (layer, neuron) by c; restore on exit."""
    saved = []
    with torch.no_grad():
        for l, j in neurons:
            W = layers(model)[l].mlp.down_proj.weight
            saved.append((l, j, W[:, j].clone()))
            W[:, j] *= c
    try:
        yield
    finally:
        with torch.no_grad():
            for l, j, col in saved:
                layers(model)[l].mlp.down_proj.weight[:, j] = col


@contextlib.contextmanager
def steer(model, layer, vec):
    """Conventional activation steering (not a weight edit): add `vec` to the residual
    stream at the output of `layer`, at every position."""
    v = vec.to(model.device, dtype=next(model.parameters()).dtype)
    h = layers(model)[layer].register_forward_hook(
        lambda m, i, o: (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def no_edit():
    yield
