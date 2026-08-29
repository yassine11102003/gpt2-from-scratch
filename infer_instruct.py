"""
Run inference with the instruction-fine-tuned model (instruct_model.pth).
"""

import torch
import tiktoken

from config import GPT_BASE_CONFIG, GPT_MODEL_CONFIGS
from model import GPTModel
from dataset import format_instruction
from generate import generate, text_to_ids, ids_to_text


def load_instruct_model(model_name="gpt2-small (124M)", weights_path="instruct_model.pth", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg   = {**GPT_BASE_CONFIG, **GPT_MODEL_CONFIGS[model_name]}
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg


def generate_response(model, cfg, tokenizer, device, instruction, input_text=None,
                      max_new_tokens=100, temperature=0.0, top_k=None):
    entry  = {"instruction": instruction, "input": input_text}
    prompt = format_instruction(entry) + "\n\n### Response:\n"

    input_ids  = text_to_ids(prompt, tokenizer, device)
    output_ids = generate(model, input_ids, max_new_tokens=max_new_tokens,
                          context_size=cfg["context_length"],
                          temperature=temperature, top_k=top_k,
                          eos_id=tokenizer.eot_token)

    full_text = ids_to_text(output_ids, tokenizer)
    return full_text[len(prompt):].strip()


if __name__ == "__main__":
    model, cfg = load_instruct_model()
    tokenizer  = tiktoken.get_encoding("gpt2")
    device     = next(model.parameters()).device

    instruction = "Rewrite the sentence using a simile."
    input_text  = "The car is very fast."

    response = generate_response(model, cfg, tokenizer, device, instruction, input_text)
    print(response)
