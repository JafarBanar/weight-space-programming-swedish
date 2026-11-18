"""Quick reproducibility check: recompute the unedited model and the weights-only edit on the
160 sv_A prompts from scratch and compare with the committed outputs, answer by answer."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate
from prompts import test_sets
import edits

model, tok = load()
prompts = [chat(tok, p["sv"]) for p in test_sets()["sv_A"]]
for name, ctx in [("B0_baseline", edits.no_edit), ("W0_weights_only", lambda: edits.scale_neurons(model, [(23, 981), (22, 1821)], 0))]:
    with ctx():
        outs = generate(model, tok, prompts)
    ref = json.load(open(f"results/test_{name}.json"))["sv_A"]["outputs"]
    same = sum(a == b for a, b in zip(outs, ref))
    print(f"{name}: {same}/{len(ref)} answers identical to the committed results", flush=True)
