"""Shared helpers: model loading, chat formatting, batched greedy generation, language ID."""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lingua import Language, LanguageDetectorBuilder

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pinned Hugging Face commit
SEED = 0

LANGS = [Language.SWEDISH, Language.ENGLISH, Language.DANISH, Language.BOKMAL, Language.NYNORSK,
         Language.GERMAN, Language.DUTCH, Language.FINNISH, Language.FRENCH, Language.SPANISH,
         Language.CHINESE]
_detector = None


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_id=MODEL_ID, revision=MODEL_REV, dtype=torch.float16):
    """Half precision keeps the 1.5B model at ~3 GB, which matters on a laptop."""
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device()).eval()
    return model, tok


def chat(tok, user, system=None, history=None):
    """Format one chat prompt. system=None uses Qwen's built-in default system prompt."""
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    for u, a in (history or []):
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=48, batch_size=16):
    """Greedy (deterministic) batched generation. Returns the decoded continuations."""
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def lang_of(text):
    """ISO 639-1 code of the dominant language of `text` (lingua, 11 candidate languages).
    Bokmal and Nynorsk are merged to 'no'. Returns 'none' for empty/undetectable text."""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*LANGS).build()
    text = re.sub(r"[`*#_>\-\d]+", " ", text).strip()
    if len(text) < 3:
        return "none"
    lang = _detector.detect_language_of(text)
    if lang is None:
        return "none"
    code = lang.iso_code_639_1.name.lower()
    return "no" if code in ("nb", "nn") else code


def degenerate(text):
    """Crude degeneracy flag: empty output or a 3-word sequence repeated 3+ times."""
    w = text.split()
    if len(w) < 3:
        return True
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return max(tri.count(t) for t in set(tri)) >= 3
