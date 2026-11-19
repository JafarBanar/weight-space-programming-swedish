"""Metrics. Everything here is forward-only and deterministic (greedy decoding)."""
import collections, re
import torch
from common import chat, generate, lang_of, degenerate, free_memory

SV_MARK = {"är", "och", "inte", "jag", "av", "från", "också", "att", "mycket", "något", "några",
           "hur", "vad", "när", "där", "här", "till", "för"}
DN_MARK = {"er", "og", "ikke", "jeg", "af", "fra", "også", "at", "å", "meget", "mye", "noget", "noe",
           "nogle", "noen", "hvordan", "hvad", "hva", "når", "der", "her", "til", "for"}


def marker_share(text):
    """Detector-independent check: share of Swedish function words among Swedish +
    Danish/Norwegian function words in the text (None if neither occurs)."""
    w = re.findall(r"[a-zåäöæø]+", text.lower())
    s, d = sum(x in SV_MARK for x in w), sum(x in DN_MARK for x in w)
    return None if s + d == 0 else s / (s + d)


def response_language(model, tok, user_msgs, target, system=None, max_new_tokens=48):
    outs = generate(model, tok, [chat(tok, u, system=system) for u in user_msgs], max_new_tokens=max_new_tokens)
    langs = [lang_of(o) for o in outs]
    ms = [marker_share(o) for o in outs]
    ms_valid = [m for m in ms if m is not None]
    return {"rate": sum(l == target for l in langs) / len(langs),
            "langs": dict(collections.Counter(langs)),
            "degenerate": sum(degenerate(o) for o in outs) / len(outs),
            "marker_share": sum(ms_valid) / max(1, len(ms_valid)),
            "hit": [int(l == target) for l in langs],
            "outputs": outs}


@torch.no_grad()
def sentence_logprobs(model, tok, sents, batch_size=16):
    """Total log-probability and token count of each raw sentence (no chat template).
    '<|endoftext|>' is prepended as a start token so the first real token is scored too."""
    tok.padding_side = "right"
    lps, ns = [], []
    for i in range(0, len(sents), batch_size):
        start = "" if tok.bos_token else "<|endoftext|>"          # models with a BOS token add it themselves
        enc = tok([start + s for s in sents[i:i + batch_size]], return_tensors="pt", padding=True).to(model.device)
        logits = model(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        tok_lp = -nll * mask
        lps += tok_lp.sum(1).tolist(); ns += mask.sum(1).tolist()
        free_memory()
    tok.padding_side = "left"
    return lps, ns


def perplexity(model, tok, sents):
    lps, ns = sentence_logprobs(model, tok, sents)
    return float(torch.exp(torch.tensor(-sum(lps) / sum(ns))))


def minimal_pair_accuracy(model, tok, pairs):
    good, _ = sentence_logprobs(model, tok, [p["good"] for p in pairs])
    bad, _ = sentence_logprobs(model, tok, [p["bad"] for p in pairs])
    hit = [int(g > b) for g, b in zip(good, bad)]
    by = collections.defaultdict(list)
    for p, h in zip(pairs, hit):
        by[p["phenomenon"]].append(h)
    return {"accuracy": sum(hit) / len(hit), "by_phenomenon": {k: sum(v) / len(v) for k, v in by.items()}, "hit": hit}
