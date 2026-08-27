import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Pre-training dataset
# ---------------------------------------------------------------------------

class GPTDataset(Dataset):
    """Sliding-window dataset for next-token prediction."""

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        self.inputs, self.targets = [], []
        for i in range(0, len(ids) - max_length, stride):
            self.inputs.append(torch.tensor(ids[i: i + max_length]))
            self.targets.append(torch.tensor(ids[i + 1: i + max_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloader(txt, batch_size=8, max_length=4, stride=4,
                      shuffle=False, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Spam classification dataset
# ---------------------------------------------------------------------------

class SpamDataset(Dataset):
    """Binary spam/ham classification dataset."""

    def __init__(self, df, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        encoded = [tokenizer.encode(text) for text in df["Text"]]
        labels  = list(df["Label"])

        if max_length is None:
            max_length = max(len(t) for t in encoded)
        self.max_length = max_length

        pad_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        padded = [t[:max_length] + [pad_id] * (max_length - len(t[:max_length])) for t in encoded]

        self.inputs  = torch.tensor(padded)
        self.targets = torch.tensor(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Instruction fine-tuning dataset
# ---------------------------------------------------------------------------

def format_instruction(entry: dict) -> str:
    """Format a data entry as an instruction prompt."""
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    if entry.get("input"):
        prompt += f"\n\n### Input:\n{entry['input']}"
    return prompt


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning (input + expected response)."""

    def __init__(self, data: list[dict], tokenizer):
        self.encoded = []
        for entry in data:
            prompt   = format_instruction(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            self.encoded.append(tokenizer.encode(prompt + response))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def instruction_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                            allowed_max_length=None, device="cpu"):
    """
    Collate a batch of variable-length token sequences.
    Pads inputs and masks padding tokens in targets with ignore_index
    so they don't contribute to the loss.
    """
    max_len = max(len(item) + 1 for item in batch)
    inputs, targets = [], []

    for item in batch:
        padded = item + [pad_token_id] * (max_len - len(item))
        inp = torch.tensor(padded[:-1])
        tgt = torch.tensor(padded[1:])

        # Mask all padding positions after the first one
        pad_mask = tgt == pad_token_id
        pad_positions = torch.nonzero(pad_mask)
        if pad_positions.numel() > 1:
            tgt[pad_positions[1:]] = ignore_index

        if allowed_max_length is not None:
            inp = inp[:allowed_max_length]
            tgt = tgt[:allowed_max_length]

        inputs.append(inp)
        targets.append(tgt)

    return torch.stack(inputs).to(device), torch.stack(targets).to(device)
