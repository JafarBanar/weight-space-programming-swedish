"""Check the AI-written Swedish and Danish material against native-speaker sources on the internet.
No model is involved. Sources:
  * sv.wiktionary.org        noun gender, definite forms, adjective forms
  * Språkbanken Korp          counts in native Swedish text (Swedish Wikipedia corpus) of the
                              grammatical vs the ungrammatical variant of every minimal-pair pattern
  * sv/da.wikipedia.org       do the topic terms exist as article titles?
  * LanguageTool public API   spelling / style check of all Swedish and Danish prompts
Output: results/language_checks.json and results/language_checks.md
"""
import json, os, re, sys, time, urllib.parse, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
import minimal_pairs as mp
import prompts as pr

UA = {"User-Agent": "research-script/1.0"}


CACHE_PATH = "data/http_cache.json"
CACHE = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def get(url, params=None, data=None, tries=4):
    """GET/POST returning JSON. Successful answers are cached on disk, so a rerun only repeats failures."""
    if params:
        url += "?" + urllib.parse.urlencode(params)
    key = url + ("|" + json.dumps(data, sort_keys=True, ensure_ascii=False) if data else "")
    if key in CACHE:
        return CACHE[key]
    body = urllib.parse.urlencode(data).encode() if data else None
    for k in range(tries):
        try:
            if "wiki" in url:
                time.sleep(2.0)                                 # Wikimedia throttles fast anonymous clients
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=UA), timeout=90) as r:
                CACHE[key] = json.loads(r.read().decode("utf-8"))
            os.makedirs("data", exist_ok=True); json.dump(CACHE, open(CACHE_PATH, "w"), ensure_ascii=False)
            return CACHE[key]
        except Exception as e:                                  # noqa: BLE001
            err = str(e); time.sleep(5 + 10 * k)
    return {"ERROR": err}


def wiktionary(word):
    w = get("https://sv.wiktionary.org/w/api.php", {"action": "parse", "page": word, "prop": "wikitext|text",
                                                     "format": "json", "formatversion": 2})
    if "parse" not in w:
        return None, None
    wt = w["parse"]["wikitext"]
    sv = wt.split("==Svenska==")[1] if "==Svenska==" in wt else wt
    sv = re.split(r"\n==[^=]", sv)[0]
    return sv, w["parse"]["text"]


def korp(cqp):
    d = get("https://ws.spraakbanken.gu.se/ws/korp/v8/query", {"corpus": "WIKIPEDIA-SV", "cqp": cqp, "start": 0, "end": 0})
    time.sleep(0.15)
    return d.get("hits")


def w(x):
    return f'[word="{x}"]'


def main():
    out = {}
    # ---------- 1. nouns: gender and definite form (Wiktionary) ----------
    nouns = []
    for n in mp.EN + mp.ETT:
        mine = "en" if n in mp.EN else "ett"
        sv, html = wiktionary(n)
        tags = set(re.findall(r"\{\{sv-subst-([nt])-", sv or ""))
        listed = {"n": "en", "t": "ett"}
        genders = sorted(listed[t] for t in tags)
        row = {"noun": n, "mine": mine, "wiktionary": genders, "lookup_failed": sv is None,
               "ok": mine in genders, "both_genders_listed": len(genders) == 2}
        if n in mp.DEF and html is not None:
            row["definite"] = mp.DEF[n]; row["definite_ok"] = bool(re.search(r">\s*%s\s*<" % re.escape(mp.DEF[n]), html))
        nouns.append(row)
    out["nouns"] = nouns
    # ---------- 2. adjectives: neuter and plural/definite forms (Wiktionary) ----------
    adjs = []
    for base, neut, plur in mp.ADJ:
        _, html = wiktionary(base)
        adjs.append({"adj": base, "neuter": neut, "plural": plur, "lookup_failed": html is None,
                     "neuter_ok": bool(html and re.search(r">\s*%s\s*<" % neut, html)), "plural_ok": bool(html and re.search(r">\s*%s\s*<" % plur, html))})
    for a in adjs:                                               # tell a failed lookup apart from a real mismatch
        a["problem"] = (not a["lookup_failed"]) and not (a["neuter_ok"] and a["plural_ok"])
    out["adjectives"] = adjs
    # ---------- 3. corpus counts for every minimal-pair pattern (Korp, native Swedish text) ----------
    prev = json.load(open("results/language_checks.json")).get("corpus_counts", {}) if os.path.exists("results/language_checks.json") else {}
    counts = {}
    def cmp(key, good, bad):
        if key not in counts:
            old = prev.get(key)
            if old and old["good_cqp"] == good and old["bad_cqp"] == bad and old["good"] is not None and old["bad"] is not None:
                counts[key] = old
            else:
                counts[key] = {"good_cqp": good, "bad_cqp": bad, "good": korp(good), "bad": korp(bad)}
    pairs = mp.build(seed=0)
    for p in pairs:
        g, b = p["good"].rstrip(".").split(), p["bad"].rstrip(".").split()
        ph = p["phenomenon"]
        if ph == "article_gender":
            i = next(k for k in range(len(g)) if g[k] != b[k])
            cmp(f"{ph}: {g[i]} {g[i+1]}", w(g[i]) + " " + w(g[i+1]), w(b[i]) + " " + w(b[i+1]))
        elif ph == "double_definiteness":
            art, noun_def, noun_bare = g[2], g[4], b[4]
            cmp(f"{ph}: {art} ADJ {noun_def}", w(art) + ' [pos="JJ"] ' + w(noun_def), w(art) + ' [pos="JJ"] ' + w(noun_bare))
    for base, neut, _ in mp.ADJ:                                 # adjective agreement: "en stor NOUN" vs "en stort NOUN", and the reverse
        cmp(f"adjective_agreement: en {base}", w("en") + " " + w(base) + ' [pos="NN"]', w("en") + " " + w(neut) + ' [pos="NN"]')
        cmp(f"adjective_agreement: ett {neut}", w("ett") + " " + w(neut) + ' [pos="NN"]', w("ett") + " " + w(base) + ' [pos="NN"]')
    for adv in mp.ADV:                                           # verb-second after a fronted adverbial
        a = " ".join(w(x) for x in adv.split())
        cmp(f"verb_second: {adv}", "<sentence> " + a + ' [pos="VB"] [pos="PN"]', "<sentence> " + a + ' [pos="PN"] [pos="VB"]')
    for conj in ("om", "när"):                                   # negation before the finite verb in a subordinate clause
        cmp(f"subclause_negation: {conj}", w(conj) + ' [pos="PN"] ' + w("inte") + ' [pos="VB" & msd=".*PRS.*"]',
            w(conj) + ' [pos="PN"] [pos="VB" & msd=".*PRS.*"] ' + w("inte"))
    out["corpus_counts"] = counts
    # ---------- 4. topic terms exist as Wikipedia article titles ----------
    def wp(lang, titles):
        res = {}
        for i in range(0, len(titles), 20):
            d = get(f"https://{lang}.wikipedia.org/w/api.php", {"action": "query", "titles": "|".join(t[0].upper() + t[1:] for t in titles[i:i+20]),
                                                                 "redirects": 1, "format": "json", "formatversion": 2})
            back = {}
            for k in ("normalized", "redirects"):
                for x in d.get("query", {}).get(k, []):
                    back[x["to"]] = back.get(x["from"], x["from"])
            for pg in d.get("query", {}).get("pages", []):
                res[back.get(pg["title"], pg["title"])] = not pg.get("missing", False)
        return res
    import prompts2 as pr2
    sv_topics = [t[0] for t in pr.TOPICS] + [t[0] for t in pr2.TOPICS2]
    out["topics_sv_wikipedia"] = wp("sv", sv_topics)
    out["topics_da_wikipedia"] = wp("da", pr.DA_TEST_TOPICS)
    def usage(lang, terms):
        res = {}
        for t in terms:
            d = get(f"https://{lang}.wikipedia.org/w/api.php", {"action": "query", "list": "search", "srsearch": f'"{t}"', "srlimit": 1,
                                                                 "srinfo": "totalhits", "format": "json", "formatversion": 2})
            res[t] = d.get("query", {}).get("searchinfo", {}).get("totalhits")
        return res
    out["topic_usage_sv_wikipedia"] = usage("sv", sv_topics)
    out["topic_usage_da_wikipedia"] = usage("da", pr.DA_TEST_TOPICS)
    # ---------- 5. LanguageTool on prompts ----------
    def lt(lang, sents):
        found = []
        for i in range(0, len(sents), 150):
            chunk = sents[i:i+150]; text = "\n".join(chunk)
            d = get("https://api.languagetool.org/v2/check", data={"text": text, "language": lang})
            for m in d.get("matches", []):
                ctx = m["context"]
                found.append({"rule": m["rule"]["id"], "category": m["rule"]["category"]["id"],
                              "flagged": ctx["text"][ctx["offset"]:ctx["offset"] + ctx["length"]], "context": ctx["text"],
                              "suggestions": [r["value"] for r in m.get("replacements", [])[:3]]})
            time.sleep(4)
        return {"sentences": len(sents), "matches": found}
    sv_prompts = sorted({t[0].format(t=p[0]) for t in pr.TEMPLATES for p in pr.TOPICS} | {x["sv"] for x in pr2.t1_prompts()})
    da_prompts = sorted({t.format(t=p) for t in pr.DA_TEST_TEMPLATES for p in pr.DA_TEST_TOPICS})
    out["languagetool_sv_prompts"] = lt("sv", sv_prompts)
    out["languagetool_da_prompts"] = lt("da-DK", da_prompts)
    out["languagetool_sv_good_sentences"] = lt("sv", sorted({p["good"] for p in pairs}))
    json.dump(out, open("results/language_checks.json", "w"), ensure_ascii=False, indent=1)

    # ---------- summary ----------
    L = ["# Checks of the AI-written Swedish and Danish material against online native sources\n"]
    failed = [r["noun"] for r in nouns if r["lookup_failed"]]
    bad_n = [r for r in nouns if not r["ok"] and not r["lookup_failed"]]; both = [r["noun"] for r in nouns if r["both_genders_listed"]]
    bad_d = [r for r in nouns if r.get("definite_ok") is False]
    L.append(f"* **Noun gender (sv.wiktionary):** {sum(r['ok'] for r in nouns)}/{len(nouns)} agree. Disagree: {[r['noun'] for r in bad_n] or 'none'}. Lookup failed: {failed or 'none'}. Listed with both genders: {both or 'none'}.")
    L.append(f"* **Definite forms (sv.wiktionary):** {sum(1 for r in nouns if r.get('definite_ok'))}/{sum(1 for r in nouns if 'definite' in r)} found in the inflection table. Not found: {[r['definite'] for r in bad_d] or 'none'}.")
    L.append(f"* **Adjective forms (sv.wiktionary):** {sum(a['neuter_ok'] and a['plural_ok'] for a in adjs)}/{len(adjs)} agree. Mismatch: {[a['adj'] for a in adjs if a['problem']] or 'none'}. Lookup failed: {[a['adj'] for a in adjs if a['lookup_failed']] or 'none'}.")
    by = {}
    for k, c in counts.items():
        ph = k.split(":")[0]; ok = c["good"] is not None and c["bad"] is not None and c["good"] > c["bad"]
        by.setdefault(ph, []).append((k, c["good"], c["bad"], ok))
    for ph, rows in by.items():
        wrong = [(k, g, b) for k, g, b, ok in rows if not ok]
        L.append(f"* **Corpus counts, {ph} (Korp, Swedish Wikipedia):** grammatical variant more frequent in {sum(ok for *_, ok in rows)}/{len(rows)} patterns; "
                 f"total {sum(g or 0 for _, g, _, _ in rows):,} grammatical vs {sum(b or 0 for _, _, b, _ in rows):,} ungrammatical hits. Not confirmed: {wrong or 'none'}.")
    for lang, key in (("Swedish", "topics_sv_wikipedia"), ("Danish", "topics_da_wikipedia")):
        miss = [t for t, ok in out[key].items() if not ok]
        u = out[key.replace("topics_", "topic_usage_")]
        L.append(f"* **{lang} topic terms as Wikipedia titles:** {sum(out[key].values())}/{len(out[key])} exist. Missing as titles: {miss or 'none'}. "
                 f"Full-text hits in {lang} Wikipedia, lowest five: {sorted(u.items(), key=lambda x: x[1] if x[1] is not None else -1)[:5]}.")
    for name, key in (("Swedish prompts", "languagetool_sv_prompts"), ("Danish prompts", "languagetool_da_prompts"), ("Swedish grammatical sentences", "languagetool_sv_good_sentences")):
        m = out[key]["matches"]; uniq = sorted({(x["rule"], x["flagged"]) for x in m})
        L.append(f"* **LanguageTool, {name}:** {out[key]['sentences']} sentences, {len(m)} flags, {len(uniq)} distinct: {uniq[:25]}")
    open("results/language_checks.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
