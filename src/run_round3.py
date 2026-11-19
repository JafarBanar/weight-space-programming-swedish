"""Round 3: is there a fix that makes the model answer in the language of the question, without an
instruction at inference time and without damage? See PREREGISTRATION_ROUND3.md.

  python src/run_round3.py <model> prep            # training targets, natural answers, dev baseline
  python src/run_round3.py <model> p3              # choose the request-aware prompt on dev
  python src/run_round3.py <model> e1              # closed-form conditional edit: statistics, probes, dev grid
  python src/run_round3.py <model> train F2        # LoRA distillation (full precision, own process)
  python src/run_round3.py <model> train F3 <lr>   # rank-3 LoRA on the single matrix chosen for E1
  python src/run_round3.py <model> loradev         # evaluate LoRA checkpoints on dev
  python src/run_round3.py <model> freeze          # write frozen.json (everything chosen on dev)
  python src/run_round3.py <model> test            # run every frozen condition once on the test questions
  python src/run_round3.py <model> robust          # decoding and system-prompt checks for passing methods

<model> is qwen2.5-1.5b (primary) or qwen2.5-0.5b (replication, no retuning). Stages are cached in results/r3/<model>/.
"""
import contextlib, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate, lang_of, free_memory, SEED, MODEL_ID, MODEL_REV
from prompts2 import t1_prompts
from prompts3 import mkqa3, ud_train, LANGS4
from evaluate import response_language, perplexity, minimal_pair_accuracy
from minimal_pairs import build as build_pairs
from native_pairs import build as build_native
from download_models import SURVEY_MODELS
import edits, fetch_data

MODELS = {"qwen2.5-1.5b": (MODEL_ID, MODEL_REV),
          "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", SURVEY_MODELS["Qwen/Qwen2.5-0.5B-Instruct"])}
HELP = "You are a helpful assistant. "
SYS = {"sv": HELP + "Svara alltid på svenska.", "da": HELP + "Svar altid på dansk.", "no": HELP + "Svar alltid på norsk.", "en": None}
P2 = HELP + "Always answer in the same language as the user's message."
P3_CANDIDATES = [
    P2,
    HELP + "Reply in the language the user writes in. If the user writes in Swedish, reply in Swedish; in Danish, reply in Danish; in Norwegian, reply in Norwegian.",
    HELP + "Svara på samma språk som användaren skriver på.",
    HELP + "Detect the language of the user's message and write your entire answer in that language. Do not answer in English unless the user writes in English.",
    HELP + "Answer in the user's language. Svara på användarens språk. Svar på brugerens sprog. Svar på brukerens språk.",
    HELP + "The user may write in any language, including Swedish, Danish and Norwegian. Never translate your answer into English; respond in the user's own language.",
]
UNSEEN_SYS = "You are a concise assistant for a public library."
E1_LAYERS, E1_STRENGTHS = (6, 10, 14), (1, 2, 4)
PRIMARY_LAYERS = 28
PRIMARY = "qwen2.5-1.5b"


def cached(path, fn):
    if os.path.exists(path):
        return json.load(open(path))
    t = time.time(); out = fn()
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"[{os.path.basename(path)}] done in {time.time() - t:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- generation helpers
@torch.no_grad()
def generate_end(model, tok, prompts, max_new_tokens=64, bs=16):
    """Greedy generation that also reports whether the answer ended by itself (needed for training targets)."""
    end_id = tok.convert_tokens_to_ids("<|im_end|>"); out = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             top_k=None, repetition_penalty=1.0, pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:]
        for row, text in zip(gen, tok.batch_decode(gen, skip_special_tokens=True)):
            out.append((text, bool((row == end_id).any())))
        free_memory()
    return out


@torch.no_grad()
def generate_sampled(model, tok, prompts, seed, max_new_tokens=48, bs=16):
    torch.manual_seed(seed); outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, top_k=None,
                             repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        free_memory()
    return outs


# ---------------------------------------------------------------- weight deltas (every weight method ends up here)
@contextlib.contextmanager
def weight_deltas(model, deltas):
    """Add `deltas[module_path]` (float32) to that module's weight; restore the exact original on exit."""
    saved = {}
    with torch.no_grad():
        for name, d in deltas.items():
            W = model.get_submodule(name).weight
            saved[name] = W.detach().clone()
            W.copy_((W.float() + d.to(W.device)).to(W.dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for name, w in saved.items():
                model.get_submodule(name).weight.copy_(w)


def lora_deltas(path):
    """Turn a saved LoRA state {module: (A, B, scale)} into dense weight deltas."""
    return {name: scale * (B.float() @ A.float()) for name, (A, B, scale) in torch.load(path).items()}


def n_changed(deltas):
    return int(sum(d.numel() for d in deltas.values()))


# ---------------------------------------------------------------- evaluation
def dev_eval3(model, tok, dev, ud_dev_sv, system=None, with_ppl=True):
    out, deg = {}, []
    for lang in LANGS4:
        r = response_language(model, tok, [x[lang] for x in dev], lang, system=system)
        out[lang] = r["rate"]; out[lang + "_langs"] = r["langs"]; deg.append(r["degenerate"])
    out["mean"] = sum(out[l] for l in LANGS4) / 4
    out["degenerate"] = sum(deg) / 4
    if with_ppl:
        out["ppl_sv_dev"] = perplexity(model, tok, ud_dev_sv)
    return out


def select(cands, base):
    """Pre-registered dev rule. `order` ranks interventions from weak to strong."""
    ok = [c for c in cands if c["en"] >= base["en"] - 0.025 and c["degenerate"] <= base["degenerate"] + 0.025
          and c["ppl_sv_dev"] <= 1.05 * base["ppl_sv_dev"]]
    pool = ok or [c for c in cands if c["degenerate"] <= base["degenerate"] + 0.10] or cands
    best = dict(sorted(pool, key=lambda c: (-round(c["mean"], 6), c["order"]))[0])
    best["passed_dev_filter"] = bool(ok)
    return best


def evaluate3(model, tok, D, system=None, light=False, max_new_tokens=48):
    out = {f"MK_{l}": response_language(model, tok, [r[l] for r in D["test"]], l, system=system, max_new_tokens=max_new_tokens)
           for l in LANGS4}
    if light:
        return out
    out["T1_sv"] = response_language(model, tok, [p["sv"] for p in D["t1"]], "sv", system=system)
    out["ppl"] = {k: perplexity(model, tok, v) for k, v in D["ud"].items()}
    out["pairs_synthetic"] = minimal_pair_accuracy(model, tok, D["pairs"])
    out["pairs_native"] = minimal_pair_accuracy(model, tok, D["native"])
    return out


def show(name, r):
    s = " | ".join(f"{l}->{l} {r['MK_' + l]['rate']:.3f}" for l in LANGS4)
    if "ppl" in r:
        s += f" | T1 {r['T1_sv']['rate']:.3f} | ppl " + "/".join(f"{r['ppl'][k]:.1f}" for k in LANGS4) + \
             f" | pairs {r['pairs_synthetic']['accuracy']:.3f}/{r['pairs_native']['accuracy']:.3f}"
    print(f"{name:12s} {s}", flush=True)


# ---------------------------------------------------------------- stage: prep
def prep(model, tok, train, dev, ud, out):
    def targets():
        rows = []
        for lang in LANGS4:
            outs = generate_end(model, tok, [chat(tok, r[lang], system=SYS[lang]) for r in train])
            rows += [{"id": r["id"], "lang": lang, "q": r[lang], "target": t, "ended": e, "target_lang": lang_of(t)}
                     for r, (t, e) in zip(train, outs)]
        return rows
    tg = cached(f"{out}/targets.json", targets)
    print("targets kept per language:", {l: sum(r["lang"] == l and r["target_lang"] == l for r in tg) for l in LANGS4}, flush=True)

    def natural():
        rows = []
        for lang in ("sv", "da", "no"):
            outs = generate_end(model, tok, [chat(tok, r[lang]) for r in train])
            rows += [{"id": r["id"], "lang": lang, "q": r[lang], "answer": t, "answer_lang": lang_of(t)} for r, (t, _) in zip(train, outs)]
        return rows
    nat = cached(f"{out}/natural_answers.json", natural)
    print("natural answer languages:", {l: {a: sum(r["lang"] == l and r["answer_lang"] == a for r in nat) for a in ("sv", "da", "no", "en")}
                                        for l in ("sv", "da", "no")}, flush=True)
    base = cached(f"{out}/dev_B0.json", lambda: dev_eval3(model, tok, dev, ud["sv"]["dev"]))
    print("dev B0:", {k: round(base[k], 3) for k in (*LANGS4, "mean", "degenerate", "ppl_sv_dev")}, flush=True)
    return tg, nat, base


# ---------------------------------------------------------------- stage: p3
def p3(model, tok, dev, ud, out, base):
    def run():
        rows = []
        for i, s in enumerate(P3_CANDIDATES):
            r = dev_eval3(model, tok, dev, None, system=s, with_ppl=False)
            r.update(system=s, order=i, ppl_sv_dev=base["ppl_sv_dev"]); rows.append(r)
            print("P3", i, {k: round(r[k], 3) for k in (*LANGS4, "mean")}, flush=True)
        return rows
    rows = cached(f"{out}/dev_P3.json", run)
    sel = select(rows, base)
    json.dump(sel, open(f"{out}/selected_P3.json", "w"), ensure_ascii=False, indent=1)
    print("P3 selected:", sel["system"], round(sel["mean"], 3), flush=True)
    return sel


# ---------------------------------------------------------------- stage: e1 (closed form, no gradient)
def e1_layers(model):
    n = len(edits.layers(model))
    return [round(l * n / PRIMARY_LAYERS) for l in E1_LAYERS]


@torch.no_grad()
def e1_statistics(model, tok, tg, nat, layer_ids):
    """One forward pass per (question, answer) sequence. For each layer: the Gram matrix of the MLP hidden
    activations over all positions, their sums per class of position, and the mean residual after the block
    over the answer positions, split by prompt language and answer language."""
    seqs = [(r["lang"], r["target_lang"], r["q"], r["target"]) for r in tg if r["target_lang"] == r["lang"]]
    seqs += [(r["lang"], r["answer_lang"], r["q"], r["answer"]) for r in nat if r["answer_lang"] in ("sv", "da", "no", "en")]
    n_pre = len(tok(chat(tok, "§§").split("§§")[0])["input_ids"])      # tokens before the user's text
    cls = {"sv": 0, "da": 1, "no": 2, "en": 3}
    d = edits.layers(model)[0].mlp.down_proj.weight.shape[1]
    dev_ = model.device
    G = {l: torch.zeros(d, d, dtype=torch.float64) for l in layer_ids}
    Gb = {l: torch.zeros(d, d, device=dev_) for l in layer_ids}
    S = {l: torch.zeros(5, d, dtype=torch.float64) for l in layer_ids}        # sums over sv / da / no / en / pre-user positions
    N = torch.zeros(5, dtype=torch.float64)
    resid, cnt, store = {}, {}, {}
    hooks = [edits.layers(model)[l].mlp.down_proj.register_forward_pre_hook(
        lambda m, inp, l=l: store.__setitem__(l, inp[0][0].float())) for l in layer_ids]
    for i, (lang, alang, q, a) in enumerate(seqs):
        p = chat(tok, q); n_p = len(tok(p)["input_ids"])
        enc = tok(p + a, return_tensors="pt").to(dev_)
        hs = model(**enc, output_hidden_states=True).hidden_states
        T = enc["input_ids"].shape[1]
        N[4] += n_pre; N[cls[lang]] += T - n_pre
        for l in layer_ids:
            A = store[l]
            Gb[l] += A.T @ A
            S[l][4] += A[:n_pre].sum(0).cpu().double(); S[l][cls[lang]] += A[n_pre:].sum(0).cpu().double()
            if T > n_p:
                k = (l, lang, alang)
                resid[k] = resid.get(k, 0) + hs[l + 1][0, n_p:].float().mean(0).cpu().double(); cnt[k] = cnt.get(k, 0) + 1
        if i % 50 == 49 or i == len(seqs) - 1:
            for l in layer_ids:
                G[l] += Gb[l].cpu().double(); Gb[l].zero_()
            free_memory()
    for h in hooks:
        h.remove()
    return G, S, N, {k: v / cnt[k] for k, v in resid.items()}, cnt, len(seqs)


def e1_build(G, S, N, resid, cnt, layer):
    """Ridge probes p_k (does the user write in language k?) and write vectors v_k, for k = sv, da, no."""
    g = G[layer]; lam = 0.01 * float(g.diagonal().mean())
    P = torch.linalg.solve(g + lam * torch.eye(g.shape[0], dtype=torch.float64), S[layer][:3].T).T      # 3 x d
    V, src = [], []
    for k in ("sv", "da", "no"):
        ref = (layer, k, "en") if cnt.get((layer, k, "en"), 0) >= 20 else (layer, "en", "en")
        V.append(resid[(layer, k, k)] - resid[ref]); src.append(f"{ref[1]}->{ref[2]} (n={cnt[ref]})")
    V = torch.stack(V)                                                                                  # 3 x hidden
    fit = (P @ (S[layer] / N[:, None]).T).tolist()            # mean probe output: rows = probes, columns = sv/da/no/en/pre-user
    return P.float(), V.float(), {"ridge": lam, "write_reference": src, "mean_probe_output": fit,
                                  "write_norms": [float(v.norm()) for v in V]}


def e1_delta(model, layer, P, V, alpha):
    return {f"model.layers.{layer}.mlp.down_proj": alpha * (V.T @ P)}


def e1(model, tok, tg, nat, dev, ud, out, base, copy_from=None):
    layer_ids = e1_layers(model)
    path = f"{out}/e1_edit.pt"
    if not os.path.exists(path):
        t = time.time()
        G, S, N, resid, cnt, n_seq = e1_statistics(model, tok, tg, nat, layer_ids)
        built, info = {}, {"sequences": n_seq, "positions": N.tolist()}
        for l in layer_ids:
            P, V, d = e1_build(G, S, N, resid, cnt, l); built[l] = (P, V); info[str(l)] = d
        torch.save(built, path); json.dump(info, open(f"{out}/e1_build.json", "w"), indent=1)
        print(f"E1 statistics and probes built from {n_seq} sequences in {time.time() - t:.0f}s", flush=True)
    built = torch.load(path); info = json.load(open(f"{out}/e1_build.json"))
    for l in layer_ids:
        print(f"E1 layer {l}: mean probe output (rows sv/da/no probe; cols sv/da/no/en/pre-user positions)",
              [[round(x, 2) for x in row] for row in info[str(l)]["mean_probe_output"]], flush=True)

    if copy_from is not None:
        l = round(copy_from["layer"] * len(edits.layers(model)) / PRIMARY_LAYERS)
        sel = {"layer": l, "strength": copy_from["strength"], "copied_from": PRIMARY, "passed_dev_filter": None}
        json.dump(sel, open(f"{out}/selected_E1.json", "w"), indent=1)
        print("E1 copied from the primary model, no retuning:", sel, flush=True)
        return sel

    def grid():
        rows = []
        for l in layer_ids:
            P, V = built[l]
            for a in E1_STRENGTHS:
                with weight_deltas(model, e1_delta(model, l, P, V, a)):
                    r = dev_eval3(model, tok, dev, ud["sv"]["dev"])
                r.update(layer=l, strength=a, order=a * 100 + l); rows.append(r)
                print("E1", {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("layer", "strength", *LANGS4, "mean", "degenerate", "ppl_sv_dev")}, flush=True)
        return rows
    rows = cached(f"{out}/dev_E1.json", grid)
    sel = select(rows, base)
    json.dump(sel, open(f"{out}/selected_E1.json", "w"), indent=1)
    print("E1 selected:", {k: sel[k] for k in ("layer", "strength", "mean", "passed_dev_filter")}, flush=True)
    return sel


# ---------------------------------------------------------------- stage: train (LoRA, full precision, own process)
def train(key, out, tag, lr, layer=None, epochs=2, bs=8):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    model, tok = load(*MODELS[key], dtype=torch.float32)
    tg = json.load(open(f"{out}/targets.json")); ud = ud_train()
    data = [(chat(tok, r["q"]), r["target"] + ("<|im_end|>" if r["ended"] else "")) for r in tg if r["target_lang"] == r["lang"]]
    reg = [s for l in LANGS4 for s in ud[l]["reg"]]
    if tag.startswith("F3"):
        cfg = LoraConfig(r=3, lora_alpha=6, lora_dropout=0.0, target_modules=rf".*layers\.{layer}\.mlp\.down_proj", task_type="CAUSAL_LM")
    else:
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(SEED)
    tok.padding_side = "right"
    log, t0, step = [], time.time(), 0
    pm.train()
    for epoch in range(epochs):
        order = torch.randperm(len(data), generator=g).tolist()
        rorder = torch.randperm(len(reg), generator=g).tolist()
        for i in range(0, len(order), bs):
            # One update per `bs` examples. To fit a laptop, the batch is processed in chunks of 4 whose gradients are
            # accumulated, and each loss is back-propagated separately; the update equals that of one batch of `bs`.
            idx = order[i:i + bs]; ce_sum = kl_sum = 0.0; n_chunks = (len(idx) + 3) // 4
            for c in range(0, len(idx), 4):
                batch = [data[j] for j in idx[c:c + 4]]
                enc = tok([p + t for p, t in batch], return_tensors="pt", padding=True).to(model.device)
                labels = enc["input_ids"].clone(); labels[enc["attention_mask"] == 0] = -100
                for b, (p, _) in enumerate(batch):                  # loss on the answer tokens only
                    labels[b, :len(tok(p)["input_ids"])] = -100
                ce = pm(**enc, labels=labels).loss / n_chunks
                ce.backward(); ce_sum += ce.item(); del ce, enc, labels
                rb = [reg[rorder[(i + c + j) % len(reg)]] for j in range(4)]   # keep natural-text predictions close to the unedited model
                er = tok(["<|endoftext|>" + s for s in rb], return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
                with torch.no_grad(), pm.disable_adapter():
                    base_lp = torch.log_softmax(pm(**er).logits, -1)
                lp = torch.log_softmax(pm(**er).logits, -1)
                m = er["attention_mask"].float()
                kl = (torch.nn.functional.kl_div(lp, base_lp, log_target=True, reduction="none").sum(-1) * m).sum() / m.sum() / n_chunks
                kl.backward(); kl_sum += kl.item(); del kl, lp, base_lp, er
                free_memory()
            opt.step(); opt.zero_grad()
            log.append((ce_sum, kl_sum)); step += 1
            if step % 20 == 0:
                print(f"{tag} epoch {epoch} step {step} ce {sum(x for x, _ in log[-20:]) / 20:.3f} kl {sum(x for _, x in log[-20:]) / 20:.4f}", flush=True)
        state = {name.replace("base_model.model.", "", 1): (mod.lora_A["default"].weight.detach().cpu(),
                                                            mod.lora_B["default"].weight.detach().cpu(), float(mod.scaling["default"]))
                 for name, mod in pm.named_modules() if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {})}
        os.makedirs(f"{out}/adapters", exist_ok=True)
        torch.save(state, f"{out}/adapters/{tag}_ep{epoch + 1}.pt")
    json.dump({"tag": tag, "lr": lr, "layer": layer, "epochs": epochs, "batch_size": bs, "train_examples": len(data),
               "regulariser_sentences": len(reg), "steps": step, "trainable_params": sum(p.numel() for p in params),
               "ce_first20": sum(x for x, _ in log[:20]) / 20, "ce_last20": sum(x for x, _ in log[-20:]) / 20,
               "kl_last20": sum(x for _, x in log[-20:]) / 20, "train_seconds": time.time() - t0},
              open(f"{out}/train_{tag}.json", "w"), indent=1)
    print("TRAINED", tag, flush=True)


def loradev(model, tok, dev, ud, out, base):
    res = {}
    for fam in ("F2", "F3"):
        files = sorted(f for f in os.listdir(f"{out}/adapters") if f.startswith(fam))
        def run(files=files):
            rows = []
            for i, f in enumerate(files):
                with weight_deltas(model, lora_deltas(f"{out}/adapters/{f}")):
                    r = dev_eval3(model, tok, dev, ud["sv"]["dev"])
                r.update(adapter=f, order=i); rows.append(r)
                print(fam, f, {k: round(r[k], 3) for k in (*LANGS4, "mean", "degenerate", "ppl_sv_dev")}, flush=True)
            return rows
        rows = cached(f"{out}/dev_{fam}.json", run)
        res[fam] = select(rows, base)
        json.dump(res[fam], open(f"{out}/selected_{fam}.json", "w"), indent=1)
        print(fam, "selected:", res[fam]["adapter"], round(res[fam]["mean"], 3), res[fam]["passed_dev_filter"], flush=True)
    return res


# ---------------------------------------------------------------- stage: test
def conditions(model, tok, key, out, frozen):
    """name -> (context manager factory, system prompt, light evaluation?)"""
    c = {"B0": (edits.no_edit, None, False)}
    if key != PRIMARY:
        c["P3"] = (edits.no_edit, frozen["P3"]["system"], False)
    if key == PRIMARY:
        c["P1"] = (edits.no_edit, SYS["sv"], False)
        c["P2"] = (edits.no_edit, P2, False)
        c["P3"] = (edits.no_edit, frozen["P3"]["system"], False)
        c["W0a"] = (lambda: edits.scale_neurons(model, [(23, 981)], 0), None, False)
        s2 = json.load(open("results/r2/qwen2.5-1.5b/steer_selected.json"))["selected"]
        v = torch.tensor(json.load(open("results/r2/qwen2.5-1.5b/steer_vectors.json"))[s2["recipe"]][str(s2["layer"])])
        c["S2"] = (lambda: edits.steer(model, s2["layer"], s2["strength"] * v), None, False)
    if "E1" in frozen:
        built = torch.load(f"{out}/e1_edit.pt"); l, a = frozen["E1"]["layer"], frozen["E1"]["strength"]
        P, V = built[l]; delta = e1_delta(model, l, P, V, a)
        c["E1"] = (lambda: weight_deltas(model, delta), None, False)
        gen = torch.Generator().manual_seed(300)
        d = delta[f"model.layers.{l}.mlp.down_proj"]
        rnd = torch.randn(d.shape[0], 3, generator=gen) @ torch.randn(3, d.shape[1], generator=gen)
        c["R3"] = (lambda: weight_deltas(model, {f"model.layers.{l}.mlp.down_proj": rnd * (d.norm() / rnd.norm())}), None, False)
    for fam in ("F3", "F2"):
        if fam in frozen:
            c[fam] = (lambda fam=fam: weight_deltas(model, lora_deltas(f"{out}/adapters/{frozen[fam]['adapter']}")), None, False)
    return c


def test(model, tok, key, out, D):
    frozen = json.load(open(f"{out}/frozen.json"))
    for name, (ctx, system, light) in conditions(model, tok, key, out, frozen).items():
        def run(ctx=ctx, system=system, light=light):
            with ctx():
                return evaluate3(model, tok, D, system=system, light=light)
        show(name, cached(f"{out}/test_{name}.json", run))
    print("TEST DONE", key, flush=True)


def robust(model, tok, key, out, D, names):
    frozen = json.load(open(f"{out}/frozen.json")); conds = conditions(model, tok, key, out, frozen)
    for name in names:
        ctx, system, _ = conds[name]
        def run(ctx=ctx, system=system):
            res = {}
            with ctx():
                for l in ("sv", "da"):
                    qs = [chat(tok, r[l], system=system) for r in D["test"]]
                    res[f"sampled_{l}"] = []
                    for seed in (11, 12, 13):
                        outs = generate_sampled(model, tok, qs, seed)
                        res[f"sampled_{l}"].append({"seed": seed, "rate": sum(lang_of(o) == l for o in outs) / len(outs), "outputs": outs})
                    res[f"long_{l}"] = response_language(model, tok, [r[l] for r in D["test"]], l, system=system, max_new_tokens=160)
                if system is None:                                   # weight methods and B0: an English system prompt never seen in training
                    res["unseen_system"] = {l: response_language(model, tok, [r[l] for r in D["test"]], l, system=UNSEEN_SYS) for l in LANGS4}
            return res
        r = cached(f"{out}/robust_{name}.json", run)
        print(name, "sampled sv", [round(x["rate"], 3) for x in r["sampled_sv"]], "da", [round(x["rate"], 3) for x in r["sampled_da"]],
              "| 160 tokens sv", round(r["long_sv"]["rate"], 3), "da", round(r["long_da"]["rate"], 3),
              "| unseen system", {l: round(v["rate"], 3) for l, v in r.get("unseen_system", {}).items()}, flush=True)


def main():
    key, stage = sys.argv[1], sys.argv[2]
    out = f"results/r3/{key}"; os.makedirs(out, exist_ok=True)
    if stage == "train":
        tag = sys.argv[3]
        if tag == "F2":
            return train(key, out, "F2", 2e-4)
        layer = json.load(open(f"{out}/selected_E1.json"))["layer"]
        if key != PRIMARY:                                      # same learning rate as the adapter frozen on the primary model
            lr = float(json.load(open(f"results/r3/{PRIMARY}/frozen.json"))["F3"]["adapter"].split("_lr")[1].split("_ep")[0])
        else:
            lr = float(sys.argv[4])
        return train(key, out, f"F3_lr{lr:g}", lr, layer=layer)
    if stage == "freeze" and key != PRIMARY:                    # replication: copy every choice from the primary model
        pf = json.load(open(f"results/r3/{PRIMARY}/frozen.json"))
        frozen = {"P3": pf["P3"], "E1": json.load(open(f"{out}/selected_E1.json")),
                  "F3": {"adapter": pf["F3"]["adapter"]}, "F2": {"adapter": pf["F2"]["adapter"]}, "copied_from": PRIMARY}
        json.dump(frozen, open(f"{out}/frozen.json", "w"), ensure_ascii=False, indent=1)
        return print("frozen (copied from the primary model):", {k: v for k, v in frozen.items() if k != "P3"})
    if stage == "freeze":                                       # everything chosen on dev, written before the test stage
        frozen = {k: json.load(open(f"{out}/selected_{k}.json")) for k in ("P3", "E1", "F3", "F2") if os.path.exists(f"{out}/selected_{k}.json")}
        json.dump(frozen, open(f"{out}/frozen.json", "w"), ensure_ascii=False, indent=1)
        return print("frozen:", {k: {x: v[x] for x in ("system", "layer", "strength", "adapter", "passed_dev_filter") if x in v} for k, v in frozen.items()})
    torch.manual_seed(SEED)
    model, tok = load(*MODELS[key])
    train_q, dev, test_q = mkqa3()
    ud = ud_train()
    if stage == "prep":
        return prep(model, tok, train_q, dev, ud, out)
    base = json.load(open(f"{out}/dev_B0.json"))
    if stage == "p3":
        return p3(model, tok, dev, ud, out, base)
    if stage == "e1":
        tg = json.load(open(f"{out}/targets.json")); nat = json.load(open(f"{out}/natural_answers.json"))
        copy = None if key == PRIMARY else json.load(open(f"results/r3/{PRIMARY}/frozen.json"))["E1"]
        return e1(model, tok, tg, nat, dev, ud, out, base, copy_from=copy)
    if stage == "loradev":
        return loradev(model, tok, dev, ud, out, base)
    D = {"test": test_q, "t1": t1_prompts(), "ud": fetch_data.main(), "pairs": build_pairs(seed=SEED), "native": build_native()}
    if stage == "test":
        return test(model, tok, key, out, D)
    if stage == "robust":
        return robust(model, tok, key, out, D, sys.argv[3:])
    if stage == "robustlite":                                   # reduced robustness check (time limit): one sampling seed,
        frozen = json.load(open(f"{out}/frozen.json")); conds = conditions(model, tok, key, out, frozen)   # 160 tokens for Swedish only
        for name in sys.argv[3:]:
            ctx, system, _ = conds[name]
            def run(ctx=ctx, system=system):
                res = {}
                with ctx():
                    for l in ("sv", "da"):
                        outs = generate_sampled(model, tok, [chat(tok, r[l], system=system) for r in D["test"]], 11)
                        res[f"sampled_{l}"] = [{"seed": 11, "rate": sum(lang_of(o) == l for o in outs) / len(outs), "outputs": outs}]
                    res["unseen_system"] = {l: response_language(model, tok, [r[l] for r in D["test"]], l, system=UNSEEN_SYS) for l in LANGS4}
                    res["long_sv"] = response_language(model, tok, [r["sv"] for r in D["test"]], "sv", system=system, max_new_tokens=160)
                return res
            r = cached(f"{out}/robustlite_{name}.json", run)
            print(name, "sampled sv/da", r["sampled_sv"][0]["rate"], r["sampled_da"][0]["rate"], "| unseen system",
                  {l: round(v["rate"], 3) for l, v in r["unseen_system"].items()}, "| 160 tokens sv", round(r["long_sv"]["rate"], 3), flush=True)
        return
    if stage == "randoms":                                      # exploratory, added after the test: more random edits of E1's size
        frozen = json.load(open(f"{out}/frozen.json")); l = frozen["E1"]["layer"]
        P, V = torch.load(f"{out}/e1_edit.pt")[l]; d = e1_delta(model, l, P, V, frozen["E1"]["strength"])[f"model.layers.{l}.mlp.down_proj"]
        for seed in (301, 302, 303, 304):
            def run(seed=seed):
                gen = torch.Generator().manual_seed(seed)
                rnd = torch.randn(d.shape[0], 3, generator=gen) @ torch.randn(3, d.shape[1], generator=gen)
                with weight_deltas(model, {f"model.layers.{l}.mlp.down_proj": rnd * (d.norm() / rnd.norm())}):
                    return evaluate3(model, tok, D, light=True)
            show(f"R3_seed{seed}", cached(f"{out}/test_R3_seed{seed}.json", run))
        return
    raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
