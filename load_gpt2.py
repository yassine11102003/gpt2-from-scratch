"""
Download and load official GPT-2 weights from HuggingFace into GPTModel.
"""

import os
import requests
import torch

from config import GPT_BASE_CONFIG, GPT_MODEL_CONFIGS
from model import GPTModel


WEIGHTS_BASE_URL = "https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/"

FILE_NAMES = {
    "gpt2-small (124M)":  "gpt2-small-124M.pth",
    "gpt2-medium (355M)": "gpt2-medium-355M.pth",
    "gpt2-large (774M)":  "gpt2-large-774M.pth",
    "gpt2-xl (1558M)":    "gpt2-xl-1558M.pth",
}


def download_weights(model_name: str) -> str:
    file_name = FILE_NAMES[model_name]
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        response = requests.get(WEIGHTS_BASE_URL + file_name, timeout=300)
        response.raise_for_status()
        with open(file_name, "wb") as f:
            f.write(response.content)
        print("Done.")
    return file_name


def load_gpt2(model_name="gpt2-medium (355M)", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg   = {**GPT_BASE_CONFIG, **GPT_MODEL_CONFIGS[model_name]}
    model = GPTModel(cfg)

    weights_path = download_weights(model_name)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {model_name} on {device}")
    return model, cfg


if __name__ == "__main__":
    import tiktoken
    from generate import generate, text_to_ids, ids_to_text

    model, cfg = load_gpt2("gpt2-medium (355M)")
    device     = next(model.parameters()).device
    tokenizer  = tiktoken.get_encoding("gpt2")

    prompt     = "Every effort moves you"
    output_ids = generate(model, text_to_ids(prompt, tokenizer, device),
                          max_new_tokens=25, context_size=cfg["context_length"],
                          temperature=1.0, top_k=50)
    print(ids_to_text(output_ids, tokenizer))
