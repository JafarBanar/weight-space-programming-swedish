"""Exploration: print the dev responses that the detector labels as non-Swedish."""
import sys, json
sys.path.insert(0, "src")
from common import load, chat, generate, lang_of
from prompts import dev_prompts
model, tok = load()
fit, val = dev_prompts()
outs = generate(model, tok, [chat(tok, p["sv"]) for p in fit], max_new_tokens=64)
rows = [{"prompt": p["sv"], "lang": lang_of(o), "response": o} for p, o in zip(fit, outs)]
json.dump(rows, open("explore/02_inspect_drift.json", "w"), ensure_ascii=False, indent=1)
import collections
print(collections.Counter(r["lang"] for r in rows))
for r in rows:
    if r["lang"] != "sv":
        print(f"[{r['lang']}] {r['prompt']}\n    -> {r['response'][:230]!r}")
print("--- some sv ones")
for r in [r for r in rows if r["lang"] == "sv"][:5]:
    print(f"[{r['lang']}] {r['prompt']}\n    -> {r['response'][:230]!r}")
