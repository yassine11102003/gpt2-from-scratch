"""
Fine-tune a pre-trained GPT-2 for binary spam/ham classification.

The last token's representation is fed into a 2-class linear head.
Only the head, the final LayerNorm, and the last transformer block
are trained; all other weights are frozen.
"""

import zipfile
import requests
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import tiktoken
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from load_gpt2 import load_gpt2
from dataset import SpamDataset


DATA_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
DATA_FILE = Path("spam.tsv")


def download_spam_data():
    if DATA_FILE.exists():
        return
    zip_path = Path("spam.zip")
    response = requests.get(DATA_URL, timeout=60)
    zip_path.write_bytes(response.content)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("SMSSpamCollection") as src, open(DATA_FILE, "wb") as dst:
            dst.write(src.read())
    zip_path.unlink()


def load_balanced_dataframe():
    df = pd.read_csv(DATA_FILE, sep="\t", header=None, names=["Label", "Text"])
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})

    # Balance: keep as many ham as spam
    n_spam = (df["Label"] == 1).sum()
    ham    = df[df["Label"] == 0].sample(n=n_spam, random_state=123)
    spam   = df[df["Label"] == 1]
    df     = pd.concat([ham, spam]).sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def build_loaders(df, tokenizer, batch_size=8):
    n        = len(df)
    train_df = df.iloc[:int(0.7 * n)]
    val_df   = df.iloc[int(0.7 * n): int(0.9 * n)]
    test_df  = df.iloc[int(0.9 * n):]

    train_ds = SpamDataset(train_df, tokenizer)
    val_ds   = SpamDataset(val_df,   tokenizer, max_length=train_ds.max_length)
    test_ds  = SpamDataset(test_df,  tokenizer, max_length=train_ds.max_length)

    kwargs = dict(batch_size=batch_size, num_workers=0)
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, drop_last=False, **kwargs),
    )


def train(model_name="gpt2-medium (355M)", epochs=5, lr=1e-4, batch_size=8):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load pre-trained weights (downloads them if missing)
    model, cfg = load_gpt2(model_name, device=device)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Replace head with 2-class classifier and unfreeze key layers
    model.out_head = nn.Linear(cfg["emb_dim"], 2)
    for p in model.out_head.parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True

    model.to(device)

    download_spam_data()
    df = load_balanced_dataframe()
    train_loader, val_loader, test_loader = build_loaders(df, tokenizer, batch_size)

    optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn    = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(epochs):
        model.train()
        t_loss, t_correct, t_total, n = 0.0, 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)[:, -1, :]
            loss   = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            t_correct += (logits.argmax(-1) == targets).sum().item()
            t_total   += targets.size(0)
            n         += 1

        model.eval()
        v_loss, v_correct, v_total, vn = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits  = model(inputs)[:, -1, :]
                v_loss += loss_fn(logits, targets).item()
                v_correct += (logits.argmax(-1) == targets).sum().item()
                v_total   += targets.size(0)
                vn        += 1

        train_losses.append(t_loss / n)
        val_losses.append(v_loss / vn)
        train_accs.append(t_correct / t_total)
        val_accs.append(v_correct / v_total)
        print(f"Epoch {epoch} | train loss={train_losses[-1]:.4f} acc={train_accs[-1]:.3f}"
              f" | val loss={val_losses[-1]:.4f} acc={val_accs[-1]:.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="train"); ax1.plot(val_losses, label="val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(train_accs, label="train"); ax2.plot(val_accs, label="val")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.savefig("spam_training.png")
    plt.show()

    torch.save(model.state_dict(), "spam_model.pth")
    print("Model saved to spam_model.pth")


if __name__ == "__main__":
    train(model_name="gpt2-small (124M)", epochs=5)
