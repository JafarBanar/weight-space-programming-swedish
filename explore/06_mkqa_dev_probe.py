"""Round-2 exploration (MKQA DEV split only, unedited model): what do answers to short human-written
questions look like, and can their language be scored?"""
import sys, json, collections
sys.path.insert(0, "src")
from common import load, chat, generate, lang_of
from prompts2 import mkqa
model, tok = load()
dev, test = mkqa()
print(len(dev), len(test)); res = {}
for k in ("sv", "da", "en"):
    outs = generate(model, tok, [chat(tok, r[k]) for r in dev[:60]])
    langs = [lang_of(o) for o in outs]; nw = [len(o.split()) for o in outs]
    res[k] = [{"q": r[k], "a": o, "lang": l} for r, o, l in zip(dev, outs, langs)]
    print(k, dict(collections.Counter(langs)), "median words", sorted(nw)[len(nw)//2], "short(<6 words)", sum(n < 6 for n in nw), flush=True)
for x in res["sv"][:8] + [x for x in res["sv"] if x["lang"] != "sv"][:8]:
    print(f"[{x['lang']}] {x['q']}\n    -> {x['a'][:150]!r}")
json.dump(res, open("explore/06_mkqa_dev_probe.json", "w"), ensure_ascii=False, indent=1)
