"""Round-3 data, none of it used in rounds 1 or 2.

mkqa3    a fresh split of Apple MKQA (CC BY-SA 3.0): 300 train / 100 dev / 300 test questions, each in
         Swedish, Danish, Norwegian and English (professional translations). Questions used in round 2 are excluded.
ud_train sentences from the UD r2.14 TRAIN splits (CC BY-SA 4.0), used only for the regulariser and for the
         development perplexity check. The UD TEST sentences used for reported perplexity are not touched here.
"""
import gzip, json, os, random, re, urllib.request
from prompts2 import mkqa, MKQA_PATH

LANGS4 = ("sv", "da", "no", "en")


def mkqa3(seed=3, n_train=300, n_dev=100, n_test=300):
    cache = "data/mkqa3_selected.json"
    if os.path.exists(cache):
        d = json.load(open(cache)); return d["train"], d["dev"], d["test"]
    from common import lang_of
    dev2, test2 = mkqa(0)                                   # also downloads the file if needed
    used = {r["id"] for r in dev2 + test2}
    rows = []
    with gzip.open(MKQA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); q = r["queries"]
            if r["example_id"] not in used:
                rows.append({"id": r["example_id"], **{k: q[k].strip() for k in LANGS4}})
    random.Random(seed).shuffle(rows)
    keep = []
    for r in rows:
        if len(keep) >= n_train + n_dev + n_test:
            break
        if min(len(r[k].split()) for k in LANGS4) >= 5 and all(lang_of(r[k]) == k for k in ("sv", "da", "en")):
            keep.append(r)
    d = {"train": keep[:n_train], "dev": keep[n_train:n_train + n_dev], "test": keep[n_train + n_dev:]}
    json.dump(d, open(cache, "w"), ensure_ascii=False, indent=0)
    return d["train"], d["dev"], d["test"]


def gold_answers():
    """Short gold answers (Swedish and English strings and aliases) per MKQA question id."""
    gold = {}
    with gzip.open(MKQA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); out = set()
            for lang in ("sv", "en"):
                for a in r["answers"].get(lang, []):
                    if a.get("type") in ("unanswerable", "long_answer", "binary"):
                        continue
                    for t in [a.get("text")] + (a.get("aliases") or []):
                        if t and len(str(t)) >= 2:
                            out.add(str(t).lower())
            gold[r["example_id"]] = sorted(out)
    return gold


BASE = "https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.14/{f}"
UD_TRAIN = {"sv": ("UD_Swedish-Talbanken", "sv_talbanken-ud-train.conllu"),
            "en": ("UD_English-EWT", "en_ewt-ud-train.conllu"),
            "da": ("UD_Danish-DDT", "da_ddt-ud-train.conllu"),
            "no": ("UD_Norwegian-Bokmaal", "no_bokmaal-ud-train.conllu")}


def ud_train(n_reg=200, n_dev=100):
    """Per language: {'reg': n_reg sentences for the regulariser, 'dev': n_dev sentences for the development
    perplexity check}. Swedish sentences that also occur in the native grammar pairs are left out."""
    cache = "data/ud_train_text.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    native = set()
    if os.path.exists("data/native_pairs.json"):
        native = {p["good"] for p in json.load(open("data/native_pairs.json"))}
    data = {}
    for lang, (repo, f) in UD_TRAIN.items():
        path = f"data/external/ud/{f}"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(BASE.format(repo=repo, f=f), path)
        raw = open(path, encoding="utf-8").read()
        sents = [s for s in re.findall(r"^# text = (.+)$", raw, flags=re.M) if 8 <= len(s.split()) <= 40 and s not in native]
        random.Random(3).shuffle(sents)
        data[lang] = {"reg": sents[:n_reg], "dev": sents[n_reg:n_reg + n_dev]}
    json.dump(data, open(cache, "w"), ensure_ascii=False, indent=0)
    return data
