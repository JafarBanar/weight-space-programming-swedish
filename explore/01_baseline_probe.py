"""Exploration (dev prompts only): under which realistic conditions does the unedited model
answer a Swedish user message in the wrong language?"""
import sys, json, collections, time
sys.path.insert(0, "src")
from common import load, chat, generate, lang_of
from prompts import dev_prompts

model, tok = load()
fit, val = dev_prompts()
P = fit[:48]
EN_HIST = [("Hi! Can you help me with a few questions today?", "Of course! I'd be happy to help. What would you like to know?")]
conds = {
    "default_system": dict(),
    "plain_en_system": dict(system="You are a helpful assistant."),
    "deploy_en_system": dict(system="You are a customer support assistant for an online electronics store. Be concise, accurate and polite."),
    "en_history": dict(history=EN_HIST),
    "deploy_en_system+en_history": dict(system="You are a customer support assistant for an online electronics store. Be concise, accurate and polite.", history=EN_HIST),
    "sv_instruction": dict(system="You are a helpful assistant. Svara alltid på svenska."),
}
res = {}
for name, kw in conds.items():
    t = time.time()
    outs = generate(model, tok, [chat(tok, p["sv"], **kw) for p in P])
    c = collections.Counter(lang_of(o) for o in outs)
    res[name] = {"langs": dict(c), "examples": outs[:4]}
    print(f"{name:32s} sv={c['sv']/len(P):.2f} {dict(c)}  ({time.time()-t:.0f}s)", flush=True)
json.dump(res, open("explore/01_baseline_probe.json", "w"), ensure_ascii=False, indent=1)
