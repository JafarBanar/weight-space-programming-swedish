"""Compare a from-scratch rerun with the committed results, file by file.
  python src/compare_runs.py <committed_results_dir> <rerun_results_dir>
Every generated answer is compared as an exact string; every number as an absolute difference.
Output: results/reproduction_check.json and .md"""
import glob, json, os, sys


def walk(a, b, path, stats):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                walk(a[k], b[k], f"{path}/{k}", stats)
            else:
                stats["missing"].append(f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            stats["missing"].append(f"{path} (length {len(a)} vs {len(b)})")
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", stats)
    elif isinstance(a, str) and isinstance(b, str):
        stats["strings"] += 1; stats["strings_equal"] += a == b
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        stats["numbers"] += 1; d = abs(a - b); rel = d / max(1e-9, abs(a))
        stats["numbers_equal"] += d == 0
        if rel > stats["max_rel_diff"]:
            stats["max_rel_diff"], stats["max_rel_where"] = rel, path


def main(ref, new):
    rows, tot = [], {"strings": 0, "strings_equal": 0, "numbers": 0, "numbers_equal": 0}
    skip = ("summary", "notices_summary", "language_checks", "reproduction_check", "detector_validation", "examples")
    for f in sorted(glob.glob(f"{ref}/*.json")):
        name = os.path.basename(f)
        if name[:-5] in skip:
            continue
        g = f"{new}/{name}"
        if not os.path.exists(g):
            rows.append({"file": name, "status": "not produced by the rerun"}); continue
        st = {"strings": 0, "strings_equal": 0, "numbers": 0, "numbers_equal": 0, "max_rel_diff": 0.0, "max_rel_where": "", "missing": []}
        walk(json.load(open(f)), json.load(open(g)), "", st)
        rows.append({"file": name, **st})
        for k in tot:
            tot[k] += st[k]
    out = {"reference": ref, "rerun": new, "total": tot, "files": rows}
    json.dump(out, open("results/reproduction_check.json", "w"), indent=1)
    L = ["# Full from-scratch reproduction check\n",
         f"Generated answers and other strings identical: **{tot['strings_equal']:,} / {tot['strings']:,}**.  ",
         f"Numbers bit-identical: **{tot['numbers_equal']:,} / {tot['numbers']:,}**.\n",
         "| File | Answers identical | Numbers identical | Largest relative difference |", "|---|---|---|---|"]
    for r in rows:
        if "status" in r:
            L.append(f"| {r['file']} | {r['status']} | | |")
        else:
            L.append(f"| {r['file']} | {r['strings_equal']}/{r['strings']} | {r['numbers_equal']}/{r['numbers']} | {r['max_rel_diff']:.2e} {('at ' + r['max_rel_where']) if r['max_rel_diff'] > 0 else ''} |")
    open("results/reproduction_check.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4])); [print(r["file"], r.get("status", f"{r['strings_equal']}/{r['strings']} strings, {r['numbers_equal']}/{r['numbers']} numbers, max rel {r['max_rel_diff']:.1e}")) for r in rows]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
