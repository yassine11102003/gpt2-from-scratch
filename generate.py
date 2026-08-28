import torch
import tiktoken


def generate(model, idx, max_new_tokens, context_size,
             temperature=None, top_k=None):
    """
    Generate tokens autoregressively.

    Args:
        model:          GPTModel in eval mode.
        idx:            Starting token ids, shape (1, seq_len).
        max_new_tokens: Number of tokens to generate.
        context_size:   Maximum context the model accepts.
        temperature:    Softmax temperature (higher = more random).
                        None or 0 uses greedy decoding.
        top_k:          Keep only the top-k logits before sampling.
                        None disables top-k filtering.
    """
    for _ in range(max_new_tokens):
        idx_context = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_context)
        logits = logits[:, -1, :]  # last token logits

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits = torch.where(logits < top_values[:, [-1]],
                                 torch.tensor(float("-inf")), logits)

        if temperature and temperature != 0:
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_id), dim=-1)
    return idx


def text_to_ids(text: str, tokenizer, device="cpu") -> torch.Tensor:
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, device=device).unsqueeze(0)


def ids_to_text(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids.squeeze(0).tolist())


# Quick demo
if __name__ == "__main__":
    from model import GPTModel
    from config import GPT_TRAIN_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    model = GPTModel(GPT_TRAIN_CONFIG).to(device)
    model.eval()

    prompt = "Hello, I am"
    input_ids = text_to_ids(prompt, tokenizer, device)
    output_ids = generate(model, input_ids, max_new_tokens=20,
                          context_size=GPT_TRAIN_CONFIG["context_length"],
                          temperature=1.0, top_k=50)
    print(ids_to_text(output_ids, tokenizer))
