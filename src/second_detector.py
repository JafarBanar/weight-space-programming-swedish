"""Cross-check of every language label with a second, independent identifier.

Primary: lingua (restricted to 11 candidate languages). Second: langdetect 1.0.9 (a port of Google's
language-detection, 55 languages, different model and training data). Every saved answer in results/ is
re-labelled; agreement is reported overall and per file, and the right-language rate of each round-2 and
round-3 test condition is recomputed under the second identifier.

  python src/second_detector.py        ->  results/second_detector.json, results/second_detector.md
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import lang_of
from langdetect import DetectorFactory, detect_langs
DetectorFactory.seed = 0


def lang2(text):
    text = re.sub(r"[`*#_>\-\d]+", " ", text).strip()
    if len(text) < 3:
        return "none"
    try:
        return detect_langs(text)[0].lang
    except Exception:
        return "none"


def blocks(x, path=""):
    """Yield (path, target-or-None, outputs) for every saved list of answers."""
    if isinstance(x, dict):
        if isinstance(x.get("outputs"), list) and x["outputs"] and isinstance(x["outputs"][0], str):
            yield path, x
        for k, v in x.items():
            if k != "outputs":
                yield from blocks(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            if isinstance(v, (dict, list)):
                yield from blocks(v, f"{path}[{i}]")


def target_of(path):
    m = re.search(r"(?:MK|T1|sampled|long)_(sv|da|no|en)", path) or re.search(r"/(sv|en|da)_[AB]\b", path)
    return m.group(1) if m else None


def main():
    files = sorted(glob.glob("results/**/*.json", recursive=True))
    tot = agree = 0; per_file, rates, confusion = [], [], {}
    for f in files:
        if any(s in f for s in ("second_detector", "reproduction_check", "blind_review", "language_checks")):
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        n = a = 0
        for path, b in blocks(data):
            l1 = [lang_of(o) for o in b["outputs"]]; l2 = [lang2(o) for o in b["outputs"]]
            n += len(l1); a += sum(x == y for x, y in zip(l1, l2))
            for x, y in zip(l1, l2):
                if x != y:
                    confusion[f"{x}->{y}"] = confusion.get(f"{x}->{y}", 0) + 1
            t = target_of(path)
            if t and ("/r2/" in f or "/r3/" in f) and os.path.basename(f).startswith("test_"):
                rates.append({"file": f.replace("results/", ""), "set": path.strip("/"), "target": t, "n": len(l1),
                              "primary": sum(x == t for x in l1) / len(l1), "second": sum(y == t for y in l2) / len(l2)})
        if n:
            per_file.append({"file": f.replace("results/", ""), "answers": n, "agreement": a / n}); tot += n; agree += a
    top = sorted(confusion.items(), key=lambda kv: -kv[1])[:12]
    diffs = [abs(r["primary"] - r["second"]) for r in rates]
    out = {"answers": tot, "agreement": agree / max(1, tot), "disagreements_primary_to_second": dict(top),
           "test_conditions": len(rates), "max_rate_difference": max(diffs) if diffs else None,
           "mean_rate_difference": sum(diffs) / len(diffs) if diffs else None, "rates": rates, "per_file": per_file}
    json.dump(out, open("results/second_detector.json", "w"), ensure_ascii=False, indent=1)
    L = ["# Second language identifier: agreement with the primary detector", "",
         f"Answers re-labelled: **{tot:,}**. Same label from both identifiers: **{100 * agree / max(1, tot):.1f}%**.", "",
         "Most common disagreements (primary → second): " + ", ".join(f"{k} ({v})" for k, v in top) + ".", "",
         f"Right-language rate of {len(rates)} round-2 and round-3 test sets under each identifier: mean absolute difference "
         f"{100 * out['mean_rate_difference']:.1f} points, largest {100 * out['max_rate_difference']:.1f} points." if rates else "", "",
         "| File | Set | Target | n | Primary | Second |", "|---|---|---|---|---|---|"]
    L += [f"| {r['file']} | {r['set']} | {r['target']} | {r['n']} | {100 * r['primary']:.1f}% | {100 * r['second']:.1f}% |" for r in rates]
    open("results/second_detector.md", "w").write("\n".join(L) + "\n")
    print(L[2]); print(L[4]); print(L[6])


if __name__ == "__main__":
    main()
