"""Train en vergelijk LSTM en KWT. Slaat modellen, resultaten en confusion matrices op in results/."""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data import build_splits, inspect, make_demo_data
from models import KWT, LSTMKWS

p = argparse.ArgumentParser()
p.add_argument("--demo", action="store_true", help="nepdata gebruiken")
p.add_argument("--data", help="map met .wav bestanden")
p.add_argument("--inspect", action="store_true", help="toon paden en labels en stop")
p.add_argument("--classes", nargs="+", help="bv. volume_up volume_down")
p.add_argument("--label_from", default="parent", choices=["parent", "grandparent", "prefix"])
p.add_argument("--own", help="map met eigen opnames (submappen per klasse)")
p.add_argument("--epochs", type=int, default=25)
p.add_argument("--max_per_class", type=int, default=500)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--n_aug", type=int, default=2, help="geaugmenteerde kopieen per train-sample")
args = p.parse_args()

if args.inspect:
    if not args.data:
        raise SystemExit("Geef --data PAD mee bij --inspect")
    inspect(args.data)
    raise SystemExit

if args.demo:
    make_demo_data("demo_data")
    args.data, args.classes, args.label_from = "demo_data", ["low", "high"], "parent"

if not args.data or not args.classes:
    raise SystemExit("Gebruik --demo, of --data PAD --classes a b ...")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
Path("results").mkdir(exist_ok=True)
print(f"Device: {device}\n")

splits = build_splits(args.data, args.classes, args.label_from, args.max_per_class,
                      args.seed, own=args.own, n_aug=args.n_aug)
n_classes = len(args.classes)


def loader(split, shuffle):
    X, y = splits[split]
    return DataLoader(TensorDataset(X, y), batch_size=args.bs, shuffle=shuffle)


train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
own_dl = loader("own_test", False) if "own_test" in splits else None


def spec_augment(x):
    """Willekeurige tijd- en frequentiemaskers (op genormaliseerde features: 0 = gemiddelde)."""
    x = x.clone()
    B, T, M = x.shape
    for b in range(B):
        if np.random.rand() < 0.5:
            w = np.random.randint(0, 10)
            s = np.random.randint(0, T - w)
            x[b, s:s + w] = 0
        if np.random.rand() < 0.5:
            w = np.random.randint(0, 6)
            s = np.random.randint(0, M - w)
            x[b, :, s:s + w] = 0
    return x


@torch.no_grad()
def predict(model, dl):
    model.eval()
    preds, ys = [], []
    for xb, yb in dl:
        preds.append(model(xb.to(device)).argmax(1).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def train_model(name, model, lr):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = -1.0, None
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{name} totaal", unit="epoch")
    for ep in epoch_bar:
        model.train()
        total, n = 0.0, 0
        batch_bar = tqdm(train_dl, desc=f"  {name} epoch {ep}/{args.epochs}",
                         leave=False, unit="batch")
        for xb, yb in batch_bar:
            xb, yb = spec_augment(xb).to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
            batch_bar.set_postfix(loss=f"{total / n:.3f}")
        y, pr = predict(model, val_dl)
        val_acc = accuracy_score(y, pr)
        tqdm.write(f"[{name}] epoch {ep:02d}  loss {total / n:.3f}  val_acc {val_acc:.3f}")
        epoch_bar.set_postfix(val_acc=f"{val_acc:.3f}")
        if val_acc >= best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def ms_per_pred(model, n=100):
    """Gemiddelde inferentietijd per sample op CPU (batch van 1)."""
    model = model.to("cpu").eval()
    X = splits["test"][0][:n]
    with torch.no_grad():
        for x in X[:10]:
            model(x.unsqueeze(0))
        t0 = time.perf_counter()
        for x in X:
            model(x.unsqueeze(0))
    return (time.perf_counter() - t0) / len(X) * 1000


def save_confusion(cm, name):
    fig, ax = plt.subplots(figsize=(4 + 0.4 * n_classes, 4 + 0.4 * n_classes))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(args.classes, rotation=45, ha="right")
    ax.set_yticklabels(args.classes)
    ax.set_xlabel("Voorspeld")
    ax.set_ylabel("Echt")
    ax.set_title(f"Confusion matrix {name}")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(f"results/confusion_{name}.png", dpi=120)
    plt.close(fig)


configs = {
    "LSTM": (LSTMKWS(n_classes=n_classes), 1e-3),
    "KWT": (KWT(n_classes=n_classes), 5e-4),
}

results = {}
for name, (model, lr) in configs.items():
    print(f"\n=== Training {name} ===")
    model = train_model(name, model, lr)
    y, pr = predict(model, test_dl)
    cm = confusion_matrix(y, pr, labels=range(n_classes))
    save_confusion(cm, name)
    torch.save(model.state_dict(), f"results/{name}.pt")
    results[name] = {
        "acc": float(accuracy_score(y, pr)),
        "macro_f1": float(f1_score(y, pr, average="macro")),
        "params": sum(p.numel() for p in model.parameters()),
        "ms_per_pred": ms_per_pred(model),
    }
    if own_dl is not None:
        yo, po = predict(model.to(device), own_dl)
        results[name]["own_acc"] = float(accuracy_score(yo, po))
        results[name]["own_n"] = int(len(yo))

print(f"\n{'Model':8s} {'Acc':>6s} {'MacroF1':>8s} {'Params':>10s} {'ms/pred':>8s} {'EigenAcc':>9s}")
for name, r in results.items():
    own = f"{r['own_acc']:9.3f}" if "own_acc" in r else f"{'-':>9s}"
    print(f"{name:8s} {r['acc']:6.3f} {r['macro_f1']:8.3f} {r['params']:10,d} {r['ms_per_pred']:8.2f} {own}")
if own_dl is not None:
    print(f"(EigenAcc = nauwkeurigheid op {results['LSTM']['own_n']} eigen opnames die NIET in de training zaten)")

with open("results/results.json", "w") as f:
    json.dump({"args": vars(args), "results": results}, f, indent=2)
print("\nModellen, norm.pt, results.json en confusion matrices staan in results/")