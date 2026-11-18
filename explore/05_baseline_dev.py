"""Dev-only sanity check of the metric code on the UNEDITED model (no edit is run here)."""
import sys, json
sys.path.insert(0, "src")
from common import load
from prompts import dev_prompts
from evaluate import response_language, perplexity, minimal_pair_accuracy
from minimal_pairs import build
import fetch_data
model, tok = load()
fit, val = dev_prompts()
ud = fetch_data.main()
r = response_language(model, tok, [p["sv"] for p in val], "sv")
print("val sv:", r["rate"], r["langs"], "marker", round(r["marker_share"], 3), "degen", r["degenerate"])
r = response_language(model, tok, [p["en"] for p in val[:40]], "en")
print("val en:", r["rate"], r["langs"])
print("ppl:", {k: round(perplexity(model, tok, v), 1) for k, v in ud.items()})
mp = minimal_pair_accuracy(model, tok, build())
print("minimal pairs:", round(mp["accuracy"], 3), {k: round(v, 2) for k, v in mp["by_phenomenon"].items()})
