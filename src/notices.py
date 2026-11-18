"""Synthetic 'notice' questions with one correct code and one plausible distractor.
Added after the pre-registration, following an external review suggestion: they check that an
edit which changes the answer LANGUAGE does not damage the answer CONTENT.

Three variants of the same 40 items (8 scenario families x 5 code draws):
  sv_sv  Swedish notice and question                          -> expect a correct Swedish answer
  en_sv  English notice and question + "Svara på svenska ..." -> expect a correct Swedish answer
  en_en  English notice and question                          -> expect a correct English answer
"""
import random, re

FAMILIES = [  # (sv target, sv distractor, sv question, en target, en distractor, en question)
    ("Reserverade böcker hämtas i skåp {a}.", "Återlämnade böcker läggs i skåp {b}.",
     "Vilket skåp ska en besökare använda för att hämta en reserverad bok?",
     "Reserved books are collected from locker {a}.", "Returned books go to locker {b}.",
     "Which locker should a visitor use to collect a reserved book?"),
    ("Blodprov tas i rum {a}.", "Vaccinationer ges i rum {b}.", "Vilket rum ska en patient gå till för ett blodprov?",
     "Blood tests are taken in room {a}.", "Vaccinations are given in room {b}.",
     "Which room should a patient go to for a blood test?"),
    ("Färjan till Gotland avgår från kaj {a}.", "Färjan till Åland avgår från kaj {b}.",
     "Från vilken kaj avgår färjan till Gotland?",
     "The ferry to Gotland departs from quay {a}.", "The ferry to Åland departs from quay {b}.",
     "Which quay does the ferry to Gotland depart from?"),
    ("Passansökningar hanteras vid disk {a}.", "Körkortsfrågor hanteras vid disk {b}.",
     "Vilken disk hanterar passansökningar?",
     "Passport applications are handled at counter {a}.", "Driving licence questions are handled at counter {b}.",
     "Which counter handles passport applications?"),
    ("Besökare parkerar i zon {a}.", "Personal parkerar i zon {b}.", "I vilken zon ska en besökare parkera?",
     "Visitors park in zone {a}.", "Staff park in zone {b}.", "In which zone should a visitor park?"),
    ("Matteprovet hålls i sal {a}.", "Historieprovet hålls i sal {b}.", "I vilken sal hålls matteprovet?",
     "The maths exam is held in classroom {a}.", "The history exam is held in classroom {b}.",
     "In which classroom is the maths exam held?"),
    ("Tåget till Uppsala avgår från spår {a}.", "Tåget till Malmö avgår från spår {b}.",
     "Från vilket spår avgår tåget till Uppsala?",
     "The train to Uppsala leaves from platform {a}.", "The train to Malmö leaves from platform {b}.",
     "Which platform does the train to Uppsala leave from?"),
    ("Glas lämnas i container {a}.", "Batterier lämnas i container {b}.", "I vilken container ska glas lämnas?",
     "Glass is left in container {a}.", "Batteries are left in container {b}.",
     "In which container should glass be left?"),
]


def build(seed=0, per_family=5):
    r = random.Random(seed); items = []
    for fi, (st, sd, sq, et, ed, eq) in enumerate(FAMILIES):
        for k in range(per_family):
            la, lb = r.sample("BCDFGHKLMNPRT", 2)
            a, b = f"{la}{r.randint(10, 99)}", f"{lb}{r.randint(10, 99)}"
            first = (k + fi) % 2 == 0                       # vary whether the target fact comes first
            sv = " ".join([st, sd] if first else [sd, st]).format(a=a, b=b)
            en = " ".join([et, ed] if first else [ed, et]).format(a=a, b=b)
            items.append({"family": fi, "target": a, "distractor": b,
                          "sv_sv": f"{sv}\n{sq} Svara med en hel mening.",
                          "en_sv": f"{en}\n{eq} Svara på svenska med en hel mening.",
                          "en_en": f"{en}\n{eq} Answer with a complete sentence."})
    return items


def fact_correct(text, item):
    """Correct = names the target code and does not name the distractor code."""
    return item["target"] in text and item["distractor"] not in text


def strip_codes(text):
    return re.sub(r"\b[A-Z]\d\d\b", " ", text)
