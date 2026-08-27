"""
Instruction fine-tuning: teach the model to follow instructions
using a dataset of (instruction, input, output) triples.
"""

import json
import os
import requests
from functools import partial

import torch
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import GPT_BASE_CONFIG, GPT_MODEL_CONFIGS
from model import GPTModel
from dataset import InstructionDataset, instruction_collate_fn


DATA_URL  = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
DATA_FILE = "instruction_data.json"


def download_data():
    if not os.path.exists(DATA_FILE):
        response = requests.get(DATA_URL, timeout=60)
        with open(DATA_FILE, "w") as f:
            json.dump(response.json(), f)
    with open(DATA_FILE) as f:
        return json.load(f)


def build_loaders(data, tokenizer, device, batch_size=8):
    n_train = int(0.85 * len(data))
    n_val   = int(0.10 * len(data))

    train_data = data[:n_train]
    val_data   = data[n_train: n_train + n_val]

    collate = partial(instruction_collate_fn, device=device, allowed_max_length=1024)

    train_ds = InstructionDataset(train_data, tokenizer)
    val_ds   = InstructionDataset(val_data,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def train(model_name="gpt2-medium (355M)", epochs=20, lr=1e-4, batch_size=8):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    cfg   = {**GPT_BASE_CONFIG, **GPT_MODEL_CONFIGS[model_name]}
    model = GPTModel(cfg)

    weights_file = f"{model_name.split()[0]}.pth"
    if os.path.exists(weights_file):
        model.load_state_dict(torch.load(weights_file, weights_only=True))
    else:
        print(f"Warning: {weights_file} not found. Using random weights.")

    model.to(device)

    data = download_data()
    train_loader, val_loader = build_loaders(data, tokenizer, device, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=-100)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        t_loss, n = 0.0, 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs).flatten(0, 1)
            loss   = loss_fn(logits, targets.flatten())
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n      += 1

        model.eval()
        v_loss, vn = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                logits  = model(inputs).flatten(0, 1)
                v_loss += loss_fn(logits, targets.flatten()).item()
                vn     += 1

        train_losses.append(t_loss / n)
        val_losses.append(v_loss / vn)
        print(f"Epoch {epoch:3d} | train={train_losses[-1]:.4f} | val={val_losses[-1]:.4f}")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("instruct_loss.png")
    plt.show()

    torch.save(model.state_dict(), "instruct_model.pth")
    print("Model saved to instruct_model.pth")


if __name__ == "__main__":
    train()
