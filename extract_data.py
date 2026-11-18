"""Pakt enkel de gekozen commando's uit.
Gebruik: python extract_data.py Volume_up Volume_down [--max 600]
Resultaat: C:\\kws_data\\extracted\\<Commando>\\*.wav"""
import argparse, re, zipfile
from pathlib import Path

SRC = Path(r"C:\kws_data\Free_ST_English")
DST = Path(r"C:\kws_data\extracted")


def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())


ap = argparse.ArgumentParser()
ap.add_argument("classes", nargs="*")
ap.add_argument("--max", type=int, default=600, help="max wav's per commando")
args = ap.parse_args()

zips = {norm(z.stem): z for z in sorted(SRC.glob("*.zip"))}
if not zips:
    raise SystemExit(f"Geen .zip's gevonden in {SRC}. Check met: dir {SRC}")
if not args.classes:
    raise SystemExit("Beschikbaar: " + ", ".join(z.stem for z in zips.values()))

for c in args.classes:
    z = zips.get(norm(c))
    if z is None:
        raise SystemExit(f"'{c}' niet gevonden. Beschikbaar: {[x.stem for x in zips.values()]}")
    out = DST / z.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        wavs = [m for m in zf.namelist() if m.lower().endswith(".wav")]
        print(f"{z.stem}: {len(wavs)} wav's in zip, pak {min(len(wavs), args.max)} uit")
        for m in wavs[: args.max]:
            (out / Path(m).name).write_bytes(zf.read(m))
print("Klaar ->", DST)
