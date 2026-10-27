"""Small CNN. The torch version is for training; the numpy version is what the app runs,
so the deployed app needs no torch."""
import numpy as np

CHANNELS = (16, 32, 64)


def build_torch_model(n_classes=4):
    import torch.nn as nn
    layers, c_in = [], 1
    for c in CHANNELS:
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
        c_in = c
    return nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(CHANNELS[-1] * 8 * 8, 64), nn.ReLU(), nn.Linear(64, n_classes))


def export_weights(model):
    """Fold BatchNorm into conv weights and return a dict of numpy arrays."""
    import torch.nn as nn
    model.eval()
    out, convs = {}, [m for m in model if isinstance(m, nn.Conv2d)]
    bns = [m for m in model if isinstance(m, nn.BatchNorm2d)]
    for i, (conv, bn) in enumerate(zip(convs, bns)):
        scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().numpy()
        out[f"conv{i}_w"] = conv.weight.detach().numpy() * scale[:, None, None, None]
        out[f"conv{i}_b"] = (conv.bias.detach().numpy() - bn.running_mean.numpy()) * scale + bn.bias.detach().numpy()
    lins = [m for m in model if isinstance(m, nn.Linear)]
    for i, lin in enumerate(lins):
        out[f"fc{i}_w"] = lin.weight.detach().numpy()
        out[f"fc{i}_b"] = lin.bias.detach().numpy()
    return {k: v.astype(np.float32) for k, v in out.items()}


def _conv3x3(x, w, b):
    # x: (B, C, H, W), w: (O, C, 3, 3) -> (B, O, H, W), padding 1
    B, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # B,C,H,W,3,3
    return np.einsum("bchwij,ocij->bohw", cols, w, optimize=True) + b[None, :, None, None]


def _maxpool2(x):
    B, C, H, W = x.shape
    return x.reshape(B, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def numpy_features(weights, specs):
    """specs: (B, 64, 64) or (64, 64). Returns penultimate features (B, 64)."""
    x = np.asarray(specs, np.float32)
    if x.ndim == 2:
        x = x[None]
    x = x[:, None]
    for i in range(len(CHANNELS)):
        x = _maxpool2(np.maximum(_conv3x3(x, weights[f"conv{i}_w"], weights[f"conv{i}_b"]), 0))
    x = x.reshape(len(x), -1)
    return np.maximum(x @ weights["fc0_w"].T + weights["fc0_b"], 0)


def head(weights, feats):
    return feats @ weights["fc1_w"].T + weights["fc1_b"]


def numpy_logits(weights, specs):
    return head(weights, numpy_features(weights, specs))


def fit_mahalanobis(feats, labels, n_classes):
    """Class means + shared precision matrix, fit on in-distribution training features only."""
    mu = np.stack([feats[labels == c].mean(0) for c in range(n_classes)])
    centered = feats - mu[labels]
    cov = centered.T @ centered / len(feats) + 1e-3 * np.eye(feats.shape[1])
    return mu.astype(np.float32), np.linalg.inv(cov).astype(np.float32)


def mahalanobis(feats, mu, prec):
    """Negative min class distance: higher = more familiar."""
    d = feats[:, None, :] - mu[None]
    return -np.einsum("bcd,de,bce->bc", d, prec, d).min(1)


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def energy(z, T=1.0):
    """Energy score -T*logsumexp(z/T). Lower = more familiar. We negate it so higher = more familiar."""
    z = np.asarray(z, np.float64) / T
    m = z.max(axis=-1)
    return T * (m + np.log(np.exp(z - m[:, None]).sum(axis=-1)))  # = -energy
