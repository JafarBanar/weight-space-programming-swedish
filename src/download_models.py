"""Download the extra open-weight models used in the round-2 survey (pinned revisions, Apache-2.0)."""
from huggingface_hub import snapshot_download

SURVEY_MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "utter-project/EuroLLM-1.7B-Instruct": "a25c7fa65fc2a644e6270b8940dbe295b51da681",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}

if __name__ == "__main__":
    for repo, rev in SURVEY_MODELS.items():
        p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
        print("ok", repo, rev[:8], flush=True)
