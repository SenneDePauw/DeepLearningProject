"""Controle: (1) test op datasetbestanden, (2) live test met opgegeven woord, playback en opslag."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from data import SR, get_label, load_features, norm_label
from models import KWT, LSTMKWS

p = argparse.ArgumentParser()
p.add_argument("--dataset", help="map met datasetbestanden, bv. C:\\kws_data\\extracted")
p.add_argument("--n", type=int, default=50, help="aantal bestanden per klasse (dataset-test)")
p.add_argument("--live", action="store_true", help="live test met je micro")
p.add_argument("--playback", action="store_true", help="speel elke opname terug")
p.add_argument("--device", type=int, default=None, help="micro-nummer")
args = p.parse_args()

if not args.dataset and not args.live:
    raise SystemExit("Gebruik --dataset PAD en/of --live")

norm = torch.load("results/norm.pt", weights_only=False)
classes = norm["classes"]
n_cls = len(classes)

models = {"LSTM": LSTMKWS(n_classes=n_cls), "KWT": KWT(n_classes=n_cls)}
for name, m in models.items():
    m.load_state_dict(torch.load(f"results/{name}.pt"))
    m.eval()

for name in models:
    t = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime(Path(f"results/{name}.pt").stat().st_mtime))
    print(f"Model {name} gebruikt: results/{name}.pt (laatst gewijzigd {t})")
print(f"Klassen: {classes}\n")


def predict_path(path):
    x = load_features(path)
    x = ((x - norm["mean"]) / norm["std"]).unsqueeze(0)
    out = {}
    with torch.no_grad():
        for name, m in models.items():
            probs = torch.softmax(m(x), dim=1)[0]
            k = int(probs.argmax())
            out[name] = (k, float(probs[k]))
    return out


# ---------- 1. Dataset-test ----------
if args.dataset:
    print("=== Test op datasetbestanden ===")
    wavs = sorted(Path(args.dataset).rglob("*.wav"))
    by_label = {}
    for w in wavs:
        by_label.setdefault(norm_label(get_label(w, "parent")), []).append(w)
    correct = {name: 0 for name in models}
    total = 0
    for idx, c in enumerate(classes):
        files = by_label.get(norm_label(c), [])
        random.shuffle(files)
        for f in files[:args.n]:
            res = predict_path(f)
            for name, (k, _) in res.items():
                correct[name] += int(k == idx)
            total += 1
    for name in models:
        print(f"  {name}: {correct[name]}/{total} juist = {correct[name] / total:.0%}")
    print("  (Dit kunnen ook bestanden zijn waarop getraind is, dus dit controleert vooral of")
    print("   de modellen en de verwerking correct werken. Verwacht ongeveer 98 tot 100%.)\n")

# ---------- 2. Live-test ----------
if args.live:
    import sounddevice as sd

    out_dir = Path("results/check")
    out_dir.mkdir(parents=True, exist_ok=True)
    tally = {name: [0, 0] for name in models}   # [juist, totaal]
    print("=== Live test ===")
    print("Het script zegt welk woord je moet zeggen. Stoppen: Ctrl+C (dan zie je de samenvatting).\n")
    try:
        while True:
            true_idx = random.randrange(n_cls)
            spoken = classes[true_idx].replace("_", " ")
            input(f"Zeg straks: '{spoken.upper()}'  ->  druk Enter en spreek meteen... ")
            rec = sd.rec(2 * SR, samplerate=SR, channels=1, dtype="float32", device=args.device)
            sd.wait()
            peak = float(np.abs(rec).max())
            if peak < 0.02:
                print("  Te zacht of niets gehoord, opnieuw.\n")
                continue
            tmp = out_dir / "_tmp.wav"
            sf.write(tmp, rec, SR)
            res = predict_path(tmp)

            parts, tags = [], []
            for name, (k, pr) in res.items():
                ok = k == true_idx
                tally[name][0] += int(ok)
                tally[name][1] += 1
                parts.append(f"{name}: {classes[k]} ({pr:.0%}) {'OK' if ok else 'FOUT'}")
                tags.append(f"{name}-{classes[k]}")
            print("  " + " | ".join(parts))

            fname = f"{classes[true_idx]}__{'__'.join(tags)}__{int(time.time())}.wav"
            tmp.rename(out_dir / fname)

            if args.playback:
                print("  (opname wordt afgespeeld...)")
                sd.play(rec, SR)
                sd.wait()
            print()
    except KeyboardInterrupt:
        pass

    print("\n===== SAMENVATTING LIVE =====")
    for name, (ok, tot) in tally.items():
        if tot:
            print(f"  {name}: {ok}/{tot} juist = {ok / tot:.0%}")
    print(f"Opnames staan in {out_dir} (bestandsnaam = wat je zei en wat elk model antwoordde)")