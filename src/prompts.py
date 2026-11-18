"""Synthetic, parallel Swedish/English prompt sets.

Every prompt is a (template, topic) pair. Dev and test use DISJOINT templates and
DISJOINT topics, so nothing used to derive or select an edit appears in the test set.
All text here was written for this project (no third-party data).
"""
import random

# (Swedish, English) instruction templates with a {t} topic slot.
TEMPLATES = [
    ("Förklara kort vad {t} är.", "Briefly explain what {t} is."),
    ("Skriv tre meningar om {t}.", "Write three sentences about {t}."),
    ("Vad är det viktigaste man bör veta om {t}?", "What is the most important thing to know about {t}?"),
    ("Kan du berätta lite om {t}?", "Can you tell me a little about {t}?"),
    ("Ge mig två intressanta fakta om {t}.", "Give me two interesting facts about {t}."),
    ("Varför spelar {t} roll i samhället?", "Why does {t} matter in society?"),
    ("Hur skulle du förklara {t} för ett barn?", "How would you explain {t} to a child?"),
    ("Sammanfatta kort historien bakom {t}.", "Briefly summarise the history behind {t}."),
    # ---- templates below are TEST-only ----
    ("Vilka är för- och nackdelarna med {t}?", "What are the pros and cons of {t}?"),
    ("Skriv en kort dikt om {t}.", "Write a short poem about {t}."),
    ("Vad tänker du på när du hör ordet {t}?", "What do you think of when you hear the word {t}?"),
    ("Beskriv {t} med enkla ord.", "Describe {t} in simple words."),
    ("Jag vill lära mig mer om {t}. Var ska jag börja?", "I want to learn more about {t}. Where should I start?"),
    ("Vilka vanliga missförstånd finns det om {t}?", "What common misconceptions are there about {t}?"),
    ("Hur har synen på {t} förändrats de senaste hundra åren?", "How has the view of {t} changed over the past hundred years?"),
    ("Nämn tre saker som hänger ihop med {t}.", "Name three things that are related to {t}."),
]
DEV_TEMPLATES, TEST_TEMPLATES = TEMPLATES[:8], TEMPLATES[8:]

# (Swedish, English) topics: singular or mass nouns that fit every template.
TOPICS = [
    ("fotosyntes", "photosynthesis"), ("inflation", "inflation"), ("demokrati", "democracy"),
    ("schack", "chess"), ("kaffe", "coffee"), ("kärnkraft", "nuclear power"),
    ("evolution", "evolution"), ("programmering", "programming"), ("jazz", "jazz"),
    ("filosofi", "philosophy"), ("vaccination", "vaccination"), ("kollektivtrafik", "public transport"),
    ("sömn", "sleep"), ("vänskap", "friendship"), ("yttrandefrihet", "freedom of speech"),
    ("biologisk mångfald", "biodiversity"), ("kryptografi", "cryptography"), ("brödbakning", "bread baking"),
    ("längdskidåkning", "cross-country skiing"), ("global uppvärmning", "global warming"),
    # ---- topics below are TEST-only ----
    ("gravitation", "gravity"), ("fotboll", "football"), ("vindkraft", "wind power"),
    ("artificiell intelligens", "artificial intelligence"), ("matematik", "mathematics"), ("opera", "opera"),
    ("poesi", "poetry"), ("astronomi", "astronomy"), ("återvinning", "recycling"),
    ("distansarbete", "remote work"), ("träning", "exercise"), ("jämställdhet", "gender equality"),
    ("havsförsurning", "ocean acidification"), ("kvantfysik", "quantum physics"), ("bokföring", "bookkeeping"),
    ("fågelskådning", "birdwatching"), ("surdeg", "sourdough"), ("inkomstskatt", "income tax"),
    ("yoga", "yoga"), ("stadsplanering", "urban planning"),
]
DEV_TOPICS, TEST_TOPICS = TOPICS[:20], TOPICS[20:]


def _grid(templates, topics):
    out = []
    for ti, (tsv, ten) in enumerate(templates):
        for pi, (psv, pen) in enumerate(topics):
            out.append({"sv": tsv.format(t=psv), "en": ten.format(t=pen), "template": ti, "topic": psv})
    return out


def dev_prompts(seed=0):
    """160 dev prompts, shuffled; first 80 = 'fit' (derive directions), last 80 = 'val' (select settings)."""
    g = _grid(DEV_TEMPLATES, DEV_TOPICS)
    random.Random(seed).shuffle(g)
    return g[:80], g[80:160]


def test_prompts():
    """All 160 combinations of the 8 test templates and 20 test topics."""
    return _grid(TEST_TEMPLATES, TEST_TOPICS)


# Danish versions of the TEST templates and topics (same order), used only for a regression
# check: does an edit that removes Danish drift also destroy the ability to answer in Danish?
DA_TEST_TEMPLATES = [
    "Hvad er fordelene og ulemperne ved {t}?",
    "Skriv et kort digt om {t}.",
    "Hvad tænker du på, når du hører ordet {t}?",
    "Beskriv {t} med enkle ord.",
    "Jeg vil gerne lære mere om {t}. Hvor skal jeg begynde?",
    "Hvilke almindelige misforståelser findes der om {t}?",
    "Hvordan har synet på {t} ændret sig de seneste hundrede år?",
    "Nævn tre ting, der hænger sammen med {t}.",
]
DA_TEST_TOPICS = ["tyngdekraft", "fodbold", "vindkraft", "kunstig intelligens", "matematik", "opera", "poesi",
                  "astronomi", "genbrug", "hjemmearbejde", "motion", "ligestilling", "havforsuring",
                  "kvantefysik", "bogføring", "fuglekiggeri", "surdej", "indkomstskat", "yoga", "byplanlægning"]


def test_sets():
    """Held-out evaluation prompts.
    sv_A: dev templates x TEST topics  (held-out topics; templates seen during development)
    sv_B: TEST templates x TEST topics (held-out templates AND topics)
    en_B / da_B: English / Danish versions of sv_B (regression checks)."""
    a = _grid(DEV_TEMPLATES, TEST_TOPICS)
    b = _grid(TEST_TEMPLATES, TEST_TOPICS)
    da = [{"da": t.format(t=p), "template": ti, "topic": TEST_TOPICS[pi][0]}
          for ti, t in enumerate(DA_TEST_TEMPLATES) for pi, p in enumerate(DA_TEST_TOPICS)]
    return {"sv_A": a, "sv_B": b, "en_B": b, "da_B": da}
