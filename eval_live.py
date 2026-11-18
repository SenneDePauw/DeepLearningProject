"""Live evaluatie: neem N keer elk commando op en tel automatisch hoeveel juist zijn."""
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from data import SR, load_features
from models import KWT, LSTMKWS

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=20, help="aantal opnames per commando")
p.add_argument("--device", type=int, default=None, help="micro-nummer")
args = p.parse_args()

norm = torch.load("results/norm.pt", weights_only=False)
classes = norm["classes"]
n_cls = len(classes)

models = {"LSTM": LSTMKWS(n_classes=n_cls), "KWT": KWT(n_classes=n_cls)}
for name, m in models.items():
    m.load_state_dict(torch.load(f"results/{name}.pt"))
    m.eval()

out_dir = Path("results/live_test")
out_dir.mkdir(parents=True, exist_ok=True)

cms = {name: np.zeros((n_cls, n_cls), dtype=int) for name in models}

for true_idx, c in enumerate(classes):
    spoken = c.replace("_", " ")
    print(f"\n=== Nu {args.n}x '{spoken}' ===")
    i = 0
    while i < args.n:
        input(f"[{spoken} {i + 1}/{args.n}] Druk Enter en zeg meteen '{spoken}'... ")
        rec = sd.rec(2 * SR, samplerate=SR, channels=1, dtype="float32", device=args.device)
        sd.wait()
        if float(np.abs(rec).max()) < 0.02:
            print("  Te zacht of niets gehoord, opnieuw.")
            continue
        path = out_dir / f"{c}_{i:02d}_{int(time.time())}.wav"
        sf.write(path, rec, SR)

        x = load_features(path)
        x = ((x - norm["mean"]) / norm["std"]).unsqueeze(0)
        parts = []
        with torch.no_grad():
            for name, m in models.items():
                probs = torch.softmax(m(x), dim=1)[0]
                k = int(probs.argmax())
                cms[name][true_idx, k] += 1
                mark = "OK" if k == true_idx else "FOUT"
                parts.append(f"{name}: {classes[k]} ({probs[k]:.0%}) {mark}")
        print("  " + " | ".join(parts))
        i += 1

print("\n===== SAMENVATTING =====")
for name, cm in cms.items():
    total = int(cm.sum())
    correct = int(np.trace(cm))
    print(f"\n{name}: {correct}/{total} juist = {correct / total:.0%}")
    for t, c in enumerate(classes):
        row = dict(zip(classes, cm[t].tolist()))
        print(f"   echt {c:12s}: {cm[t, t]}/{cm[t].sum()} juist   voorspeld als {row}")