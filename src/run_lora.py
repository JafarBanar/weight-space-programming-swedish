"""Conventional fine-tuning baseline (F1), configuration fixed in PREREGISTRATION.md, no tuning.

Self-distillation: the model's own Swedish answers (generated with the explicit
"Svara alltid på svenska." system prompt) on the 80 dev-fit prompts are used as targets for
the same prompts WITHOUT that instruction. LoRA rank 8 on q/v projections, 3 epochs, lr 2e-4.
"""
import json, os, sys, time
import torch
from peft import LoraConfig, get_peft_model
sys.path.insert(0, os.path.dirname(__file__))
from common import load, chat, generate, lang_of, SEED
from prompts import dev_prompts, test_sets
from minimal_pairs import build as build_pairs
from run_experiment import full_eval, SV_SYS, R
from run_notices import notice_eval
import fetch_data


def main():
    out_path = f"{R}/test_F1_lora.json"
    if os.path.exists(out_path):
        print("cached"); return
    torch.manual_seed(SEED)
    model, tok = load()
    fit, _ = dev_prompts(SEED)
    targets = generate(model, tok, [chat(tok, p["sv"], system=SV_SYS) for p in fit], max_new_tokens=96)
    data = [(chat(tok, p["sv"]), t + "<|im_end|>") for p, t in zip(fit, targets) if lang_of(t) == "sv"]
    print(len(data), "training examples (Swedish targets only)", flush=True)

    model = model.float()                                   # train in full precision
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
    tok.padding_side = "right"
    g = torch.Generator().manual_seed(SEED)
    losses, t0 = [], time.time()
    pm.train()
    for epoch in range(3):
        order = torch.randperm(len(data), generator=g).tolist()
        for i in range(0, len(order), 4):
            batch = [data[j] for j in order[i:i + 4]]
            enc = tok([p + t for p, t in batch], return_tensors="pt", padding=True).to(model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for b, (p, _) in enumerate(batch):              # loss on the answer tokens only
                labels[b, :len(tok(p)["input_ids"])] = -100
            loss = pm(**enc, labels=labels).loss
            loss.backward(); opt.step(); opt.zero_grad()
            losses.append(loss.item())
        print(f"epoch {epoch} mean loss {sum(losses[-len(order)//4:]) / (len(order)//4):.3f}", flush=True)
    tok.padding_side = "left"
    model = pm.merge_and_unload().half().eval()
    res = full_eval(model, tok, test_sets(), fetch_data.main(), build_pairs(seed=SEED))
    json.dump(notice_eval(model, tok), open(f"{R}/notices_F1_lora.json", "w"), ensure_ascii=False, indent=1)
    res["lora"] = {"trainable_params": n_train, "train_examples": len(data), "steps": len(losses),
                   "loss_first": losses[0], "loss_last": losses[-1], "train_seconds": time.time() - t0}
    json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
    print(f"F1 sv_A={res['sv_A']['rate']:.3f} sv_B={res['sv_B']['rate']:.3f} en={res['en_B']['rate']:.3f} "
          f"da={res['da_B']['rate']:.3f} mp={res['minimal_pairs']['accuracy']:.3f} ppl_sv={res['ppl']['sv']:.1f}", flush=True)


if __name__ == "__main__":
    main()
