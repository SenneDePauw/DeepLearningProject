"""Live test: neem 2 seconden op met je micro (of geef een .wav) en classificeer."""
import argparse

import numpy as np
import soundfile as sf
import torch

from data import SR, load_features
from models import KWT, LSTMKWS

p = argparse.ArgumentParser()
p.add_argument("--wav", help="optioneel: bestaand .wav-bestand i.p.v. micro")
p.add_argument("--device", type=int, default=None, help="micro-nummer")
p.add_argument("--threshold", type=float, default=0.8, help="onder deze zekerheid: 'onzeker'")
args = p.parse_args()

norm = torch.load("results/norm.pt", weights_only=False)
classes = norm["classes"]

models = {"LSTM": LSTMKWS(n_classes=len(classes)), "KWT": KWT(n_classes=len(classes))}
for name, m in models.items():
    m.load_state_dict(torch.load(f"results/{name}.pt"))
    m.eval()


def classify(path):
    x = load_features(path)
    x = ((x - norm["mean"]) / norm["std"]).unsqueeze(0)
    with torch.no_grad():
        for name, m in models.items():
            probs = torch.softmax(m(x), dim=1)[0]
            k = probs.argmax().item()
            tag = classes[k] if probs[k] >= args.threshold else f"onzeker ({classes[k]}?)"
            print(f"  {name:5s} -> {tag:26s} ({probs[k]:.0%})")


if args.wav:
    classify(args.wav)
else:
    import sounddevice as sd
    print(f"Klassen: {classes}")
    while True:
        input("\nDruk Enter en zeg daarna je commando (je hebt 2 seconden)... ")
        print("Opnemen...")
        rec = sd.rec(2 * SR, samplerate=SR, channels=1, dtype="float32", device=args.device)
        sd.wait()
        peak = float(np.abs(rec).max())
        if peak < 0.02:
            print(f"  Bijna niets gehoord (piek {peak:.3f}). Spreek luider of kies een andere micro.")
            continue
        sf.write("results/_mic.wav", rec, SR)
        classify("results/_mic.wav")