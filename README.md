# LLM-with-framework

Mini projet PyTorch pour repondre a des questions simples en francais.

Ce projet ne construit pas un vrai LLM generaliste. Il entraine un petit modele Transformer de classification qui associe une question a une reponse connue. C'est une bonne premiere etape pour comprendre:

- la preparation d'un dataset
- la tokenisation simple
- l'entrainement d'un modele PyTorch
- l'inference en ligne de commande

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` force l'installation de la version CPU de PyTorch, suffisante pour ce mini projet.

## Entrainer le modele

```bash
python train.py
```

Le modele entraine est sauve dans `artifacts/qa_model.pt`.

## Poser une question

```bash
python chat.py --question "Quelle est la capitale de la France ?"
```

Mode interactif:

```bash
python chat.py
```

## Structure

- `data/qa_dataset.json`: questions/reponses d'exemple
- `model.py`: vocabulaire, normalisation et modele Transformer
- `train.py`: entrainement
- `chat.py`: inference

## Limites

- Le modele ne sait repondre qu'aux themes presents dans le dataset.
- Si tu veux un vrai assistant plus general, il faudra beaucoup plus de donnees et un modele bien plus gros.
- Pour progresser ensuite, la prochaine etape logique serait de remplacer la classification par un modele sequence-to-sequence ou d'utiliser un modele pre-entraine.
