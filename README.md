# LLM From Scratch

Implémentation d'un LLM de type GPT en PyTorch, codé entièrement from scratch.
Le projet couvre toute la chaîne : tokenisation → architecture → entraînement → génération → fine-tuning.

---

## Structure du projet

```
llm-from-scratch/
├── config.py            # Configurations GPT-2 (toutes les tailles)
├── model.py             # Architecture complète du transformer
├── tokenizer.py         # Tokenizers faits à la main (V1, V2)
├── dataset.py           # Datasets PyTorch pour chaque tâche
├── generate.py          # Génération de texte (greedy, temperature, top-k)
├── train_pretrain.py    # Entraînement from scratch
├── train_spam.py        # Fine-tuning classification spam/ham
├── train_instruct.py    # Fine-tuning par instructions
├── load_gpt2.py         # Chargement des vrais poids GPT-2
└── requirements.txt     # Dépendances
```

---

## Concepts couverts

### 1. Tokenisation (`tokenizer.py`)
Deux tokenizers écrits from scratch :
- **TokenizerV1** : tokenizer basique, lève une erreur si un mot est inconnu
- **TokenizerV2** : même chose mais avec un token `<|unk|>` pour les mots inconnus et `<|endoftext|>` pour marquer la fin d'un document

En pratique, on utilise **tiktoken** (le tokenizer de GPT-2, BPE) pour tout le reste.

### 2. Architecture du modèle (`model.py`)

Le modèle suit l'architecture GPT-2 :

```
Tokens  →  Embedding (token + position)
        →  N × TransformerBlock
               ├── LayerNorm
               ├── MultiHeadAttention (causal / masqué)
               ├── Résidu
               ├── LayerNorm
               ├── FeedForward (Linear → GELU → Linear)
               └── Résidu
        →  LayerNorm finale
        →  Linear head  →  Logits (vocab_size)
```

**MultiHeadAttention** : l'attention est masquée (causal) — chaque token ne peut voir que les tokens précédents. On divise les projections Q/K/V en plusieurs têtes pour capturer différentes relations.

**GELU** : fonction d'activation utilisée dans GPT-2 (plus douce que ReLU).

**LayerNorm** : normalisation appliquée avant chaque sous-bloc (pre-norm), avec des paramètres appris `scale` et `shift`.

### 3. Dataset et DataLoader (`dataset.py`)
- **GPTDataset** : fenêtre glissante sur le texte. Pour chaque position `i`, l'entrée est `tokens[i:i+L]` et la cible est `tokens[i+1:i+L+1]`.
- **SpamDataset** : encode chaque SMS, les tronque/padde à la même longueur.
- **InstructionDataset** : formate chaque exemple en `Instruction + Input + Response` et l'encode en tokens.

### 4. Génération de texte (`generate.py`)
```python
generate(model, idx, max_new_tokens, context_size, temperature, top_k)
```
- **Greedy** (`temperature=None`) : prend toujours le token le plus probable.
- **Temperature** : divise les logits avant le softmax. Plus élevée = plus aléatoire.
- **Top-k** : ne garde que les `k` tokens les plus probables avant d'échantillonner.

### 5. Pré-entraînement (`train_pretrain.py`)
Entraîne le modèle à prédire le prochain token sur un texte brut (*The Verdict* de Henry James).
- Loss : cross-entropie
- Optimiseur : Adam

### 6. Fine-tuning spam (`train_spam.py`)
Adapte GPT-2 pré-entraîné pour classer les SMS en spam ou ham.
- On gèle tous les paramètres sauf la dernière couche transformer, le LayerNorm final, et une nouvelle tête linéaire à 2 sorties.
- On utilise la représentation du **dernier token** pour la classification.

### 7. Fine-tuning par instructions (`train_instruct.py`)
Apprend au modèle à suivre des instructions au format :
```
Below is an instruction that describes a task...

### Instruction:
<instruction>

### Input:
<input optionnel>

### Response:
<réponse attendue>
```
La loss est calculée uniquement sur les tokens de la réponse (les tokens de padding sont ignorés avec `ignore_index=-100`).

### 8. Chargement des poids GPT-2 (`load_gpt2.py`)
Télécharge les poids officiels GPT-2 depuis HuggingFace et les charge dans `GPTModel`.

---

## Installation

```bash
pip install -r requirements.txt
```

Ou avec uv :
```bash
uv pip install -r requirements.txt
```

---

## Utilisation

### Entraîner from scratch

```bash
python train_pretrain.py
```

Télécharge automatiquement `the_verdict.txt`, entraîne le modèle pendant 50 epochs et sauvegarde `pretrained_model.pth`.

---

### Charger GPT-2 et générer du texte

```python
from load_gpt2 import load_gpt2
from generate import generate, text_to_ids, ids_to_text
import tiktoken

model, cfg = load_gpt2("gpt2-medium (355M)")
tokenizer  = tiktoken.get_encoding("gpt2")
device     = next(model.parameters()).device

output = generate(
    model,
    text_to_ids("Every effort moves you", tokenizer, device),
    max_new_tokens=50,
    context_size=cfg["context_length"],
    temperature=1.0,
    top_k=50,
)
print(ids_to_text(output, tokenizer))
```

---

### Fine-tuner sur la classification spam

```bash
python train_spam.py
```

Télécharge le dataset SMS Spam, entraîne pendant 5 epochs et sauvegarde `spam_model.pth`.

---

### Fine-tuner par instructions

```bash
python train_instruct.py
```

Télécharge le dataset d'instructions, entraîne pendant 20 epochs et sauvegarde `instruct_model.pth`.

---

### Utiliser uniquement le modèle

```python
from model import GPTModel
from config import GPT_TRAIN_CONFIG
import torch

model = GPTModel(GPT_TRAIN_CONFIG)
x     = torch.randint(0, 50257, (1, 10))  # batch=1, seq_len=10
logits = model(x)  # shape: (1, 10, 50257)
```

---

## Configurations disponibles

| Modèle | Paramètres | emb_dim | Couches | Têtes |
|---|---|---|---|---|
| gpt2-small | 124M | 768 | 12 | 12 |
| gpt2-medium | 355M | 1024 | 24 | 16 |
| gpt2-large | 774M | 1280 | 36 | 20 |
| gpt2-xl | 1558M | 1600 | 48 | 25 |

Pour l'entraînement from scratch, `GPT_TRAIN_CONFIG` utilise la taille small avec `context_length=256` pour réduire la mémoire nécessaire.

---

## Références

- *Build a Large Language Model (From Scratch)* — Sebastian Raschka
- Architecture originale GPT-2 : [Language Models are Unsupervised Multitask Learners](https://openai.com/research/language-unsupervised)
