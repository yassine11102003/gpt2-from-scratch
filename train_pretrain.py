"""
Pre-train GPT from scratch on a small text corpus (the_verdict.txt).
"""

import torch
import tiktoken
import matplotlib.pyplot as plt

from config import GPT_TRAIN_CONFIG
from model import GPTModel
from dataset import create_dataloader
from generate import generate, text_to_ids, ids_to_text


DATA_FILE = "data_verdict.txt"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return f.read()


def train(cfg, epochs=10, batch_size=2, lr=1e-4):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    text      = load_data()

    split       = int(0.9 * len(text))
    train_text  = text[:split]
    val_text    = text[split:]

    train_loader = create_dataloader(train_text, batch_size=batch_size,
                                     max_length=cfg["context_length"],
                                     stride=cfg["context_length"], shuffle=True)
    val_loader   = create_dataloader(val_text, batch_size=batch_size,
                                     max_length=cfg["context_length"],
                                     stride=cfg["context_length"])

    torch.manual_seed(123)
    model     = GPTModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    sample_prompt = "Hello, I am"

    for epoch in range(epochs):
        # --- training ---
        print('hi')
        model.train()
        epoch_loss, n = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs).flatten(0, 1)
            loss   = torch.nn.functional.cross_entropy(logits, targets.flatten())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        train_losses.append(epoch_loss / n)

        # --- validation ---
        print('eval')
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs).flatten(0, 1)
                val_loss += torch.nn.functional.cross_entropy(logits, targets.flatten()).item()
                n += 1
            sample_ids = generate(model, text_to_ids(sample_prompt, tokenizer, device),
                                   max_new_tokens=10, context_size=cfg["context_length"])
        val_losses.append(val_loss / n)

        print(f"Epoch {epoch:3d} | train={train_losses[-1]:.4f} | val={val_losses[-1]:.4f}")
        print(f"  Sample: {ids_to_text(sample_ids, tokenizer)}")

    # Plot
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("pretrain_loss.png")
    plt.show()

    torch.save(model.state_dict(), "pretrained_model.pth")
    print("Model saved to pretrained_model.pth")


if __name__ == "__main__":
    train(GPT_TRAIN_CONFIG)
