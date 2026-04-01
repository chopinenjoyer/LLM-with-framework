# LLM-with-framework

Pipeline PyTorch pour construire un petit LLM from scratch.

Le depot couvre maintenant quatre etapes:

- preparation des donnees
- preentrainement auto-regressif
- fine-tuning supervise
- evaluation et inference

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Preparer les donnees

```bash
python prepare_data.py
```

Les fichiers prepares seront ecrits dans `artifacts/datasets`.

## 2. Preentrainer le modele

```bash
python pretrain.py --data-dir artifacts/datasets --output artifacts/pretrained_model.pt
```

## 3. Fine-tuner le modele

```bash
python finetune.py --data-dir artifacts/datasets --checkpoint artifacts/pretrained_model.pt --output artifacts/sft_model.pt
```

## 4. Evaluer

```bash
python evaluate.py --data-dir artifacts/datasets --checkpoint artifacts/sft_model.pt
```

## 5. Discuter avec le modele

```bash
python chat.py --checkpoint artifacts/sft_model.pt
```

Question unique:

```bash
python chat.py --checkpoint artifacts/sft_model.pt --question "Quel jour vient apres jeudi ?"
```

## Fichiers principaux

- `tokenizer.py`: tokenizer byte-level
- `model.py`: modele Transformer causal decoder-only
- `prepare_data.py`: preparation du corpus et du SFT
- `pretrain.py`: preentrainement next-token prediction
- `finetune.py`: fine-tuning supervise
- `evaluate.py`: perplexite et exact match
- `chat.py`: inference interactive

## Formats de donnees

Corpus brut:

```text
Une ligne ou plusieurs paragraphes de texte libre.
```

Instruction tuning:

```json
{"instruction":"Explique la gravite.","response":"La gravite est la force..."}
```

## Point important

Le pipeline est generaliste dans sa structure, pas dans ses resultats actuels.
Pour obtenir un modele vraiment utile, il faut:

- un corpus massif et propre
- beaucoup plus de compute
- un modele plus grand
- une evaluation plus riche

Ce depot fournit la base logicielle pour ces etapes, pas un equivalent de ChatGPT local.
