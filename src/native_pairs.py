"""Minimal pairs built from NATIVE Swedish sentences (UD Swedish-Talbanken r2.14, CC BY-SA 4.0).
The grammatical side is a sentence written by a native speaker; the ungrammatical side swaps one
indefinite article (en <-> ett) whose noun and gender are given by the treebank's gold annotation.
This replaces trust in an AI-written lexicon with trust in a hand-annotated treebank."""
import json, os, urllib.request

URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/r2.14/sv_talbanken-ud-{split}.conllu"


def _sentences(split):
    path = f"data/external/ud/sv_talbanken-ud-{split}.conllu"
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    sent = []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if sent:
                yield sent
            sent = []
        elif not line.startswith("#"):
            c = line.split("\t")
            if c[0].isdigit():
                sent.append(c)
    if sent:
        yield sent


def build(n=200):
    cache = "data/native_pairs.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    swap = {"en": "ett", "ett": "en", "En": "Ett", "Ett": "En"}
    want = {"en": "Gender=Com", "ett": "Gender=Neut"}
    out = []
    for split in ("test", "train"):
        for s in _sentences(split):
            if not 6 <= len(s) <= 30 or len(out) >= n:
                continue
            for i, c in enumerate(s):
                form, upos, feats, head, rel = c[1], c[3], c[5], c[6], c[7]
                if form in swap and upos == "DET" and rel == "det" and "Definite=Ind" in feats:
                    noun = s[int(head) - 1]
                    # adjacent noun only, gold gender must agree with the article that the native writer used
                    if int(head) == i + 2 and noun[3] == "NOUN" and want[form.lower()] in noun[5] and "Number=Sing" in noun[5]:
                        def text(forms):
                            return "".join(f + ("" if "SpaceAfter=No" in t[9] else " ") for f, t in zip(forms, s)).strip()
                        forms = [t[1] for t in s]; bad = list(forms); bad[i] = swap[form]
                        out.append({"phenomenon": "native_article_gender", "good": text(forms), "bad": text(bad),
                                    "article": form.lower(), "noun": noun[1], "split": split})
                        break
    json.dump(out, open(cache, "w"), ensure_ascii=False, indent=0)
    return out


if __name__ == "__main__":
    p = build(); print(len(p), "pairs;", sum(x["article"] == "en" for x in p), "en /", sum(x["article"] == "ett" for x in p), "ett")
    for x in p[:6]:
        print(" ", x["good"], "\n   ->", x["bad"])
