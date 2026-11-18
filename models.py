import torch
import torch.nn as nn


class LSTMKWS(nn.Module):
    """Baseline: 2-laags LSTM over de tijdsframes, mean-pooling, lineaire classifier."""
    def __init__(self, n_mels=40, hidden=64, layers=2, n_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(n_mels, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):                      # x: (B, T, n_mels)
        out, _ = self.lstm(x)
        return self.fc(out.mean(dim=1))


class KWT(nn.Module):
    """Vereenvoudigde Keyword Transformer: elk tijdsframe = 1 token, CLS-token + positional emb."""
    def __init__(self, n_mels=40, T=101, dim=64, depth=3, heads=4, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(n_mels, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):                      # x: (B, T, n_mels)
        x = self.proj(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.norm(self.enc(x))
        return self.fc(x[:, 0])