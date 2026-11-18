"""Data: .wav's laden -> centreren + normaliseren -> log-Mel features -> stratified split.
Ondersteunt augmentatie en eigen opnames (--own)."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SR = 16000          # sample rate
N_SAMPLES = SR      # 1 seconde per commando
N_MELS = 40

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=480, hop_length=160, n_mels=N_MELS)
_db = torchaudio.transforms.AmplitudeToDB(top_db=80)


def norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def get_label(path: Path, mode: str) -> str:
    if mode == "parent":
        return path.parent.name
    if mode == "grandparent":
        return path.parent.parent.name
    if mode == "prefix":            # bv. play_0001.wav -> "play"
        return path.stem.split("_")[0]
    raise ValueError(mode)


def inspect(root: str):
    wavs = sorted(Path(root).rglob("*.wav"))
    print(f"{len(wavs)} .wav bestanden gevonden in {root}\n")
    for w in wavs[:8]:
        print("  ", w.relative_to(root))
    for mode in ("parent", "grandparent", "prefix"):
        c = Counter(get_label(w, mode) for w in wavs)
        print(f"\n--label_from {mode}: {len(c)} labels, top 15:")
        print("  ", c.most_common(15))


def load_wave(path) -> torch.Tensor:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def center_word(wav: torch.Tensor) -> torch.Tensor:
    """Peak-normaliseer en centreer het woord op het zwaartepunt van de energie."""
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak
    frame = 160
    n = wav.numel() // frame
    noise_floor = 1e-4 * torch.randn(N_SAMPLES)
    if n < 5:
        return noise_floor
    env = wav[: n * frame].view(n, frame).pow(2).mean(1).sqrt()
    env = F.avg_pool1d(env.view(1, 1, -1), kernel_size=5, stride=1, padding=2).view(-1)
    if env.max() <= 1e-8:
        return noise_floor
    e2 = env.pow(2) * (env > 0.1 * env.max()).float()
    if e2.sum() <= 0:
        return noise_floor
    centroid = (e2 * torch.arange(n, dtype=torch.float32)).sum() / e2.sum()
    center = int((centroid.item() + 0.5) * frame)
    start = center - N_SAMPLES // 2
    out = torch.zeros(N_SAMPLES)
    s0, e0 = max(start, 0), min(start + N_SAMPLES, wav.numel())
    if e0 > s0:
        out[s0 - start: e0 - start] = wav[s0:e0]
    return out + noise_floor


def features(wav: torch.Tensor) -> torch.Tensor:
    return _db(_mel(center_word(wav))).T             # (T=101, 40)


def load_features(path) -> torch.Tensor:
    return features(load_wave(path))


def augment_wave(wav: torch.Tensor, rng) -> torch.Tensor:
    """Snelheid, galm, micro-filters en achtergrondruis (allemaal willekeurig)."""
    # 1. snelheid (snelle lineaire interpolatie)
    f = rng.uniform(0.85, 1.15)
    new_len = max(int(wav.numel() / f), 16)
    wav = F.interpolate(wav.view(1, 1, -1), size=new_len, mode="linear",
                        align_corners=False).view(-1)

    # 2. galm: synthetische kamerimpulsrespons (exponentieel wegstervende ruis)
    if rng.random() < 0.5:
        rt = rng.uniform(0.05, 0.3)                      # galmduur in seconden
        L = int(rt * SR)
        t = torch.arange(L, dtype=torch.float32) / SR
        rir = torch.randn(L) * torch.exp(-6.9 * t / rt)
        rir[0] = 1.0                                     # direct geluid
        rir = rir / rir.abs().sum()
        wav = torchaudio.functional.fftconvolve(wav, rir)[: wav.numel()]

    # 3. micro-nabootsing: laagdoorlaat en/of hoogdoorlaat
    if rng.random() < 0.4:
        wav = torchaudio.functional.lowpass_biquad(wav, SR, float(rng.uniform(2500, 7000)))
    if rng.random() < 0.4:
        wav = torchaudio.functional.highpass_biquad(wav, SR, float(rng.uniform(100, 400)))

    # 4. achtergrondruis
    snr_db = rng.uniform(0, 30)
    sig_pow = wav.pow(2).mean().clamp_min(1e-10)
    noise = torch.randn_like(wav) * (sig_pow / 10 ** (snr_db / 10)).sqrt()
    return wav + noise


def make_demo_data(root: str, n=150, seed=0):
    """Nepdata: twee 'woorden' = lage vs hoge toon met ruis. Enkel om de pipeline te testen."""
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    for name, f0 in (("low", 300), ("high", 1200)):
        d = Path(root) / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            f = f0 * rng.uniform(0.8, 1.2)
            x = 0.3 * np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(SR)
            sf.write(d / f"{name}_{i:04d}.wav", x.astype("float32"), SR)


def build_splits(root, classes, label_from="parent", max_per_class=500, seed=42,
                 own=None, n_aug=2, n_own_aug=8):
    print("Stap 1/4: bestanden zoeken...")
    wavs = sorted(Path(root).rglob("*.wav"))
    by_label = {}
    for w in wavs:
        by_label.setdefault(norm_label(get_label(w, label_from)), []).append(w)

    wanted = [norm_label(c) for c in classes]
    missing = [c for c in wanted if c not in by_label]
    if missing:
        raise SystemExit(f"Klassen {missing} niet gevonden. Beschikbaar: "
                         f"{sorted(by_label)[:30]}\nProbeer --inspect of een andere --label_from.")

    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for idx, c in enumerate(wanted):
        files = list(by_label[c])
        rng.shuffle(files)
        files = files[:max_per_class]
        paths += files
        labels += [idx] * len(files)
    print("Samples per klasse:", {classes[i]: labels.count(i) for i in range(len(classes))})

    print("Stap 2/4: audio laden...")
    waves = [load_wave(p) for p in tqdm(paths, desc="Audio laden", unit="bestand")]
    y_all = np.array(labels)

    idx = np.arange(len(y_all))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y_all[tmp], random_state=seed)

    print(f"Stap 3/4: features maken (train x{1 + n_aug} door augmentatie)...")
    Xtr, ytr = [], []
    for i in tqdm(tr, desc="Train features", unit="sample"):
        Xtr.append(features(waves[i]))
        ytr.append(y_all[i])
        for _ in range(n_aug):
            Xtr.append(features(augment_wave(waves[i], rng)))
            ytr.append(y_all[i])
    Xva = [features(waves[i]) for i in tqdm(va, desc="Val features", unit="sample")]
    Xte = [features(waves[i]) for i in tqdm(te, desc="Test features", unit="sample")]
    yva, yte = list(y_all[va]), list(y_all[te])

    # eigen opnames: 70% train (sterk geaugmenteerd), 30% eigen test
    Xown, yown = [], []
    if own:
        print("Eigen opnames verwerken...")
        own_tr = 0
        for idx_c, c in enumerate(classes):
            files = sorted((Path(own) / c).glob("*.wav"))
            if not files:
                continue
            files = list(files)
            rng.shuffle(files)
            n_test = max(1, int(0.3 * len(files))) if len(files) >= 5 else 0
            for f in tqdm(files[n_test:], desc=f"Eigen train {c}", unit="opname"):
                w = load_wave(f)
                Xtr.append(features(w))
                ytr.append(idx_c)
                for _ in range(n_own_aug):
                    Xtr.append(features(augment_wave(w, rng)))
                    ytr.append(idx_c)
                own_tr += 1
            for f in files[:n_test]:
                Xown.append(features(load_wave(f)))
                yown.append(idx_c)
        print(f"Eigen opnames: {own_tr} in train, {len(Xown)} als eigen test")

    print("Stap 4/4: normaliseren...")
    Xtr = torch.stack(Xtr)
    mean, std = Xtr.mean((0, 1)), Xtr.std((0, 1)) + 1e-5

    def prep(X, y):
        return (torch.stack(X) - mean) / std, torch.tensor(np.array(y))

    out = {
        "train": ((Xtr - mean) / std, torch.tensor(np.array(ytr))),
        "val": prep(Xva, yva),
        "test": prep(Xte, yte),
    }
    if Xown:
        out["own_test"] = prep(Xown, yown)

    Path("results").mkdir(exist_ok=True)
    torch.save({"mean": mean, "std": std, "classes": list(classes)}, "results/norm.pt")
    print(f"Klaar: train {len(out['train'][1])}, val {len(out['val'][1])}, "
          f"test {len(out['test'][1])} samples\n")
    return out