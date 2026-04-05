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

## 1. Normaliser les donnees

Corpus brut:

```bash
python normalize_data.py --input data/raw_corpus.txt --output data/raw_corpus.normalized.txt --format raw
```

SFT JSONL:

```bash
python normalize_data.py --input data/instructions_train.jsonl --output data/instructions_train.normalized.jsonl --format jsonl --dedupe
```

## 2. Preparer les donnees

```bash
python prepare_data.py
```

Les fichiers prepares seront ecrits dans `artifacts/datasets`.
Le pretraining est maintenant ecrit sous forme de shards `npy` avec un manifeste `pretrain_manifest.json`.

## 3. Preentrainer le modele

```bash
python pretrain.py --data-dir artifacts/datasets --output artifacts/pretrained_model.pt
```

## 4. Fine-tuner le modele

```bash
python finetune.py --data-dir artifacts/datasets --checkpoint artifacts/pretrained_model.pt --output artifacts/sft_model.pt
```

## 5. Evaluer

```bash
python evaluate.py --data-dir artifacts/datasets --checkpoint artifacts/sft_model.pt
```

Par defaut, l'evaluation lit `data/instructions_eval.jsonl` et compare trois modes:

- `model`: generation seule
- `retrieval`: recherche seule dans les exemples SFT
- `hybrid`: retrieval puis generation

## 6. Discuter avec le modele

```bash
python chat.py --checkpoint artifacts/sft_model.pt
```

Question unique:

```bash
python chat.py --checkpoint artifacts/sft_model.pt --question "Quel jour vient apres jeudi ?"
```

Forcer un mode d'inference:

```bash
python chat.py --checkpoint artifacts/sft_model.pt --mode model
python chat.py --checkpoint artifacts/sft_model.pt --mode hybrid
```

## Fichiers principaux

- `tokenizer.py`: tokenizer byte-level
- `model.py`: modele Transformer causal decoder-only
- `normalize_data.py`: nettoyage du texte brut et du JSONL
- `prepare_data.py`: preparation du corpus et du SFT
- `pretrain.py`: preentrainement next-token prediction
- `finetune.py`: fine-tuning supervise
- `evaluate.py`: perplexite et exact match
- `chat.py`: inference interactive
- `data/instructions_eval.jsonl`: evaluation hors train plus difficile

## Formats de donnees

Corpus brut:

```text
Une ligne ou plusieurs paragraphes de texte libre.
```

Pour un corpus plus gros, `prepare_data.py` segmente le pretraining en shards `npy` pour eviter un seul fichier monolithique.

Instruction tuning:

```json
{"instruction":"Explique la gravite.","response":"La gravite est la force..."}
```

Evaluation dediee:

```json
{"instruction":"Quelle ville est la capitale du Japon ?","response":"La capitale du Japon est Tokyo."}
```

## Point important

Le pipeline est generaliste dans sa structure, pas dans ses resultats actuels.
Pour obtenir un modele vraiment utile, il faut:

- un corpus massif et propre
- beaucoup plus de compute
- un modele plus grand
- une evaluation plus riche

Ce depot fournit la base logicielle pour ces etapes, pas un equivalent de ChatGPT local.
