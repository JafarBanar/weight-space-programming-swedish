"""Round-2 prompt sets, none of which overlaps with round 1.

T1   320 synthetic prompts: 40 NEW templates x 8 NEW topics (Swedish + English versions).
     Many templates with few topics each, because round 1 showed drift is driven by templates.
MKQA human-translated questions (Apple MKQA, CC BY-SA 3.0): the same Natural Questions queries
     translated by professional translators into Swedish, Danish and English. Used so that the
     conclusions no longer rest on Swedish and Danish text written by an AI.
"""
import gzip, json, os, random, urllib.request

TEMPLATES2 = [
    ("Vad innebär {t}?", "What does {t} involve?"),
    ("Ge en kort introduktion till {t}.", "Give a short introduction to {t}."),
    ("Vilken roll spelar {t} i vardagen?", "What role does {t} play in everyday life?"),
    ("Skriv en rubrik och två meningar om {t}.", "Write a headline and two sentences about {t}."),
    ("Vad skulle en expert säga om {t}?", "What would an expert say about {t}?"),
    ("Lista fem nyckelord som har med {t} att göra.", "List five keywords that have to do with {t}."),
    ("Hur kommer man igång med {t}?", "How does one get started with {t}?"),
    ("Varför är det svårt att förstå {t}?", "Why is it hard to understand {t}?"),
    ("Skriv ett kort stycke om {t} för en skoltidning.", "Write a short paragraph about {t} for a school newspaper."),
    ("Vilka frågor brukar folk ställa om {t}?", "What questions do people usually ask about {t}?"),
    ("Berätta något överraskande om {t}.", "Tell me something surprising about {t}."),
    ("Hur hänger {t} ihop med teknik?", "How is {t} connected to technology?"),
    ("Vad är skillnaden mellan teori och praktik när det gäller {t}?", "What is the difference between theory and practice when it comes to {t}?"),
    ("Skriv ett vänligt mejl till en kollega om {t}.", "Write a friendly email to a colleague about {t}."),
    ("Vilka yrken har med {t} att göra?", "Which professions have to do with {t}?"),
    ("Förklara {t} på ett sätt som en tonåring förstår.", "Explain {t} in a way a teenager understands."),
    ("Vad borde alla känna till om {t}?", "What should everyone know about {t}?"),
    ("Ge tre tips till den som är nyfiken på {t}.", "Give three tips to someone who is curious about {t}."),
    ("Hur ser framtiden ut för {t}?", "What does the future look like for {t}?"),
    ("Skriv en kort dialog mellan två vänner som pratar om {t}.", "Write a short dialogue between two friends talking about {t}."),
    ("Vilka är de största utmaningarna inom {t}?", "What are the biggest challenges in {t}?"),
    ("Vad har {t} för betydelse för Sverige?", "What significance does {t} have for Sweden?"),
    ("Sammanfatta {t} i en enda mening.", "Summarise {t} in a single sentence."),
    ("Vilka böcker eller källor rekommenderar du om {t}?", "Which books or sources do you recommend about {t}?"),
    ("Hur undervisar man barn i {t}?", "How does one teach children {t}?"),
    ("Skriv en faktaruta om {t}.", "Write a fact box about {t}."),
    ("Vad är det roligaste med {t}?", "What is the most fun thing about {t}?"),
    ("Finns det några risker med {t}?", "Are there any risks with {t}?"),
    ("Hur har forskningen om {t} utvecklats?", "How has research on {t} developed?"),
    ("Jämför {t} förr och nu.", "Compare {t} then and now."),
    ("Vilka begrepp måste man kunna för att förstå {t}?", "Which concepts must one know to understand {t}?"),
    ("Skriv en kort presentation av {t} för en bred publik.", "Write a short presentation of {t} for a general audience."),
    ("Vad kostar det att ägna sig åt {t}?", "What does it cost to pursue {t}?"),
    ("Varför väljer vissa människor att arbeta med {t}?", "Why do some people choose to work with {t}?"),
    ("Ge ett konkret exempel som visar vad {t} handlar om.", "Give a concrete example that shows what {t} is about."),
    ("Vilka myter om {t} stämmer inte?", "Which myths about {t} are not true?"),
    ("Hur påverkar {t} miljön?", "How does {t} affect the environment?"),
    ("Skriv tre frågor till ett quiz om {t}.", "Write three quiz questions about {t}."),
    ("Vad säger kritikerna om {t}?", "What do the critics say about {t}?"),
    ("Avsluta meningen: Det bästa med {t} är", "Complete the sentence: The best thing about {t} is"),
]
TOPICS2 = [("geologi", "geology"), ("arkitektur", "architecture"), ("näringslära", "nutrition"),
           ("solenergi", "solar energy"), ("teater", "theatre"), ("statistik", "statistics"),
           ("trädgårdsodling", "gardening"), ("källkritik", "source criticism")]


def t1_prompts():
    """320 prompts in template-major order (index = template * 8 + topic)."""
    return [{"sv": ts.format(t=ps), "en": te.format(t=pe), "template": ti, "topic": ps}
            for ti, (ts, te) in enumerate(TEMPLATES2) for ps, pe in TOPICS2]


MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"
MKQA_PATH = "data/external/mkqa/mkqa.jsonl.gz"


def mkqa(seed=0, n_dev=120, n_test=300):
    """Parallel human-translated questions. Keeps items whose Swedish, Danish and English versions are
    each identified as that language (so questions that are mostly an English title are dropped) and
    have at least five words. Returns (dev, test); each item has 'sv', 'da', 'en', 'id'."""
    from common import lang_of
    cache = "data/mkqa_selected.json"
    if os.path.exists(cache):
        d = json.load(open(cache)); return d["dev"], d["test"]
    if not os.path.exists(MKQA_PATH):
        os.makedirs(os.path.dirname(MKQA_PATH), exist_ok=True)
        urllib.request.urlretrieve(MKQA_URL, MKQA_PATH)
    rows = []
    with gzip.open(MKQA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); q = r["queries"]
            rows.append({"id": r["example_id"], "sv": q["sv"].strip(), "da": q["da"].strip(), "en": q["en"].strip()})
    random.Random(seed).shuffle(rows)
    keep = []
    for r in rows:
        if len(keep) >= n_dev + n_test:
            break
        if min(len(r[k].split()) for k in ("sv", "da", "en")) >= 5 and all(lang_of(r[k]) == k for k in ("sv", "da", "en")):
            keep.append(r)
    d = {"dev": keep[:n_dev], "test": keep[n_dev:n_dev + n_test]}
    json.dump(d, open(cache, "w"), ensure_ascii=False, indent=0)
    return d["dev"], d["test"]
