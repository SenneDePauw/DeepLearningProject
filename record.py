"""Neem eigen opnames op voor training. Voorbeeld:
python record.py --label volume_up --n 40 --out C:\\kws_data\\own
"""
import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000

p = argparse.ArgumentParser()
p.add_argument("--label", help="bv. volume_up")
p.add_argument("--n", type=int, default=40, help="aantal opnames")
p.add_argument("--out", default="own_data", help="hoofdmap voor eigen opnames")
p.add_argument("--device", type=int, default=None, help="micro-nummer (zie --list)")
p.add_argument("--list", action="store_true", help="toon beschikbare audioapparaten")
args = p.parse_args()

if args.list:
    print(sd.query_devices())
    raise SystemExit
if not args.label:
    raise SystemExit("Geef --label mee, bv. --label volume_up")

folder = Path(args.out) / args.label
folder.mkdir(parents=True, exist_ok=True)
start_idx = len(list(folder.glob("*.wav")))

print(f"Je neemt {args.n}x '{args.label.replace('_', ' ')}' op. Varieer een beetje: "
      f"snelheid, afstand tot de micro, toonhoogte.")
i = 0
while i < args.n:
    input(f"\n[{i + 1}/{args.n}] Druk Enter en zeg '{args.label.replace('_', ' ')}'... ")
    rec = sd.rec(2 * SR, samplerate=SR, channels=1, dtype="float32", device=args.device)
    sd.wait()
    rec = rec[:, 0]
    peak = float(np.abs(rec).max())
    if peak < 0.02:
        print(f"  Te zacht of niets gehoord (piek {peak:.3f}). Opnieuw proberen.")
        continue
    sf.write(folder / f"{args.label}_{start_idx + i:04d}_{int(time.time())}.wav", rec, SR)
    print(f"  Opgeslagen (piek {peak:.2f})")
    i += 1

print(f"\nKlaar. {args.n} opnames in {folder}")