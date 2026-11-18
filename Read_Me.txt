# KWS: LSTM vs Keyword Transformer (SynTTS-Commands)

## Setup
    pip install -r requirements.txt

## 1. Pipeline testen (nepdata, geen download nodig)
    python train.py --demo

## 2. Echte data
Pak `dataset_samples.zip` uit (uit de GitHub repo) of download de volledige dataset.

    python train.py --data PAD/NAAR/DATA --inspect
Dit toont voorbeeldpaden en labels. Kies daarna `--label_from parent|grandparent|prefix`
zodat de gevonden labels de commando's zijn (en niet bv. speaker-ids).

    python train.py --data PAD/NAAR/DATA --classes play pause --label_from parent

Opties: `--epochs 15 --max_per_class 500 --bs 32`.
Output in `results/`: `results.json` + `confusion_LSTM.png` / `confusion_KWT.png`.