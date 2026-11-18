"""Download natural held-out text for perplexity checks: Universal Dependencies test sets
(release r2.14, CC BY-SA 4.0) for Swedish, English, Danish and Norwegian Bokmal.
Keeps the first 200 sentences with 8-40 words from each test file."""
import json, os, re, urllib.request

BASE = "https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.14/{f}"
FILES = {"sv": ("UD_Swedish-Talbanken", "sv_talbanken-ud-test.conllu"),
         "en": ("UD_English-EWT", "en_ewt-ud-test.conllu"),
         "da": ("UD_Danish-DDT", "da_ddt-ud-test.conllu"),
         "no": ("UD_Norwegian-Bokmaal", "no_bokmaal-ud-test.conllu")}


def main(out="data/ud_text.json", n=200):
    if os.path.exists(out):
        return json.load(open(out))
    data = {}
    for lang, (repo, f) in FILES.items():
        raw = urllib.request.urlopen(BASE.format(repo=repo, f=f), timeout=60).read().decode("utf-8")
        sents = [s for s in re.findall(r"^# text = (.+)$", raw, flags=re.M) if 8 <= len(s.split()) <= 40]
        data[lang] = sents[:n]
        print(lang, len(data[lang]), "sentences")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"), ensure_ascii=False, indent=0)
    return data


if __name__ == "__main__":
    main()
