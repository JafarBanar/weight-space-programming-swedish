"""Synthetic Swedish minimal pairs (grammatical vs. ungrammatical) for five phenomena.
Scored without any chat template: the model 'passes' a pair if it gives the grammatical
sentence a higher total log-probability. This measures Swedish competence, not language choice."""
import random

EN = ["bil", "bok", "stol", "hund", "katt", "dörr", "lampa", "blomma", "flaska", "väska", "klocka",
      "cykel", "tidning", "dator", "telefon", "penna", "tröja", "jacka", "säng", "soffa", "kopp", "båt"]
ETT = ["hus", "bord", "äpple", "barn", "fönster", "träd", "brev", "glas", "rum", "tåg", "flygplan",
       "kök", "golv", "tak", "berg", "djur", "piano", "kort", "språk", "namn", "fotografi"]
DEF = {"bil": "bilen", "bok": "boken", "stol": "stolen", "hund": "hunden", "katt": "katten", "dörr": "dörren",
       "lampa": "lampan", "klocka": "klockan", "cykel": "cykeln", "båt": "båten", "hus": "huset",
       "bord": "bordet", "äpple": "äpplet", "barn": "barnet", "fönster": "fönstret", "träd": "trädet",
       "brev": "brevet", "glas": "glaset", "rum": "rummet", "tåg": "tåget", "piano": "pianot", "tak": "taket"}
ADJ = [("stor", "stort", "stora"), ("fin", "fint", "fina"), ("gammal", "gammalt", "gamla"),
       ("vacker", "vackert", "vackra"), ("dyr", "dyrt", "dyra"), ("billig", "billigt", "billiga"),
       ("ny", "nytt", "nya"), ("grön", "grönt", "gröna"), ("tung", "tungt", "tunga"), ("ren", "rent", "rena")]
FRAMES = ["Jag ser {}.", "Hon köpte {} igår.", "Vi har {} hemma.", "De hittade {} i går kväll.", "Han ritade {}."]
ADV = ["Igår", "Idag", "På morgonen", "Förra veckan", "I somras", "Ibland", "Sedan", "Nu"]
SUBJ = ["hon", "han", "vi", "de", "jag", "min bror"]
VPAST = ["köpte", "såg", "hittade", "sålde", "målade", "lånade"]
MATRIX = ["Vi blir oroliga om", "Jag blir förvånad när", "Det märks när", "Alla märker när", "Hon blir ledsen om"]
VPRES = ["ringer", "svarar", "kommer", "sover", "lyssnar", "skrattar", "arbetar", "skriver"]


def build(n_per=50, seed=0):
    r = random.Random(seed)
    P = []
    def noun():
        return (r.choice(EN), "en") if r.random() < 0.5 else (r.choice(ETT), "ett")
    for _ in range(n_per):                                  # 1. article gender
        n, g = noun(); bad = "ett" if g == "en" else "en"; f = r.choice(FRAMES)
        P.append(("article_gender", f.format(f"{g} {n}"), f.format(f"{bad} {n}")))
    for _ in range(n_per):                                  # 2. adjective agreement (indefinite)
        n, g = noun(); a = r.choice(ADJ); f = r.choice(FRAMES)
        good, bad = (a[0], a[1]) if g == "en" else (a[1], a[0])
        P.append(("adjective_agreement", f.format(f"{g} {good} {n}"), f.format(f"{g} {bad} {n}")))
    for _ in range(n_per):                                  # 3. verb-second word order
        adv, s, v = r.choice(ADV), r.choice(SUBJ), r.choice(VPAST); n, g = noun()
        P.append(("verb_second", f"{adv} {v} {s} {g} {n}.", f"{adv} {s} {v} {g} {n}."))
    for _ in range(n_per):                                  # 4. negation placement in subordinate clause
        m, s, v = r.choice(MATRIX), r.choice(SUBJ[:5]), r.choice(VPRES)
        P.append(("subclause_negation", f"{m} {s} inte {v}.", f"{m} {s} {v} inte."))
    for _ in range(n_per):                                  # 5. double definiteness
        n = r.choice(list(DEF)); art = "den" if n in EN else "det"; a = r.choice(ADJ)[2]
        P.append(("double_definiteness", f"Jag gillar {art} {a} {DEF[n]}.", f"Jag gillar {art} {a} {n}."))
    seen, out = set(), []
    for p in P:                                             # drop exact duplicates
        if p not in seen:
            seen.add(p); out.append({"phenomenon": p[0], "good": p[1], "bad": p[2]})
    return out
