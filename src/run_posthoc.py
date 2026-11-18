"""Post-hoc diagnostics (NOT pre-registered; run after the main results were seen).

The activation check showed that the two W0 neurons behave differently: L23/N981 fires strongly
positive while the model writes Danish/Norwegian, whereas L22/N1821 fires strongly NEGATIVE while
it writes Swedish. So which of the two deletions actually removes the drift?
Conditions: delete only L23/N981; delete only L22/N1821.  Output: results/test_W0a_*.json, test_W0b_*.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import load, SEED
from prompts import test_sets
from minimal_pairs import build as build_pairs
from run_experiment import full_eval, R
import edits, fetch_data


def main():
    model, tok = load()
    T, ud, pairs = test_sets(), fetch_data.main(), build_pairs(seed=SEED)
    for name, neuron in [("W0a_only_L23N981", (23, 981)), ("W0b_only_L22N1821", (22, 1821))]:
        path = f"{R}/test_{name}.json"
        if not os.path.exists(path):
            with edits.scale_neurons(model, [neuron], 0):
                json.dump(full_eval(model, tok, T, ud, pairs), open(path, "w"), ensure_ascii=False, indent=1)
        r = json.load(open(path))
        print(f"{name:20s} sv_A={r['sv_A']['rate']:.3f} sv_B={r['sv_B']['rate']:.3f} en={r['en_B']['rate']:.3f} "
              f"da={r['da_B']['rate']:.3f} {r['da_B']['langs']} mp={r['minimal_pairs']['accuracy']:.3f} ppl={ {k: round(v, 1) for k, v in r['ppl'].items()} }", flush=True)


if __name__ == "__main__":
    main()
