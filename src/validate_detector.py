"""How reliable is the language detector on Scandinavian text? Run it on natural sentences of
known language (UD test sets). No model involved.  Output: results/detector_validation.json"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import lang_of
from evaluate import marker_share
import fetch_data

ud = fetch_data.main(); out = {}
for lang, sents in ud.items():
    # judge 3-sentence chunks, roughly the length of a 48-token answer
    chunks = [" ".join(sents[i:i + 3]) for i in range(0, len(sents) - 2, 3)]
    c = collections.Counter(lang_of(x) for x in chunks)
    ms = [m for m in (marker_share(x) for x in chunks) if m is not None]
    out[lang] = {"n": len(chunks), "labels": dict(c), "accuracy": c[lang] / len(chunks), "mean_swedish_marker_share": sum(ms) / len(ms)}
    print(lang, out[lang])
json.dump(out, open("results/detector_validation.json", "w"), indent=1)
