"""Training smoke test: the built dataset drives a standard LeRobot-style training loop as-is.

* LeRobotDataset + delta_timestamps (action chunks, F/T history) - no conversion step
* real + sim mixed in every batch with a controllable sim fraction (WeightedRandomSampler on is_sim)
* a small multi-camera RGB(-D) + proprio + F/T behaviour-cloning policy, normalised with meta/stats.json
* validation loss reported separately for real and sim episodes of the held-out split
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

CHUNK = 10


class Encoder(nn.Module):
    def __init__(self, cin: int, dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, dim, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).flatten(1)


class Policy(nn.Module):
    def __init__(self, cams: list[str], depth: list[str], low_dim: int, act_dim: int):
        super().__init__()
        self.cams, self.depth = cams, depth
        self.enc = nn.ModuleDict({k.replace(".", "_"): Encoder(3) for k in cams})
        self.denc = nn.ModuleDict({k.replace(".", "_"): Encoder(1) for k in depth})
        d = 64 * (len(cams) + len(depth)) + low_dim
        self.head = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, CHUNK * act_dim))
        self.act_dim = act_dim

    def forward(self, imgs: dict, depth: dict, low: torch.Tensor):
        f = [self.enc[k.replace(".", "_")](imgs[k]) for k in self.cams]
        f += [self.denc[k.replace(".", "_")](depth[k]) for k in self.depth]
        return self.head(torch.cat(f + [low], 1)).view(-1, CHUNK, self.act_dim)


def _episode_ranges(info: dict, meta) -> dict[str, list[int]]:
    out = {}
    for split, rng in info.get("splits", {}).items():
        a, b = map(int, rng.split(":"))
        out[split] = list(range(a, b))
    return out


def run_smoke(root: str | Path, steps: int = 200, batch_size: int = 32, sim_weight: float = 0.5,
              device: str = "auto", num_workers: int = 6, lr: float = 3e-4) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    feats = info["features"]
    cams = [k for k, f in feats.items() if f["dtype"] == "video" and not (f.get("info") or {}).get("is_depth_map")]
    depth = [k for k, f in feats.items() if f["dtype"] == "video" and (f.get("info") or {}).get("is_depth_map")]
    low_keys = [k for k in ("observation.state", "observation.force_torque", "observation.cartesian_state",
                            "observation.gripper") if k in feats]
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ds = LeRobotDataset(info.get("repo_id", f"robodata/{root.name}"), root=root,
                        delta_timestamps={"action": [i / fps for i in range(CHUNK)]})
    stats = ds.meta.stats

    def norm(k, x):
        m = torch.as_tensor(np.asarray(stats[k]["mean"]), dtype=torch.float32, device=x.device)
        s = torch.as_tensor(np.asarray(stats[k]["std"]), dtype=torch.float32, device=x.device)
        return (x - m) / (s + 1e-6)

    ep_split = _episode_ranges(info, ds.meta)
    ep_of = np.asarray(ds.hf_dataset["episode_index"]).ravel()
    is_sim = np.asarray(ds.hf_dataset["is_sim"]).ravel().astype(bool) if "is_sim" in feats else np.zeros(len(ep_of), bool)
    train_idx = np.where(np.isin(ep_of, ep_split.get("train", list(range(ds.num_episodes)))))[0]
    val_idx = np.where(np.isin(ep_of, ep_split.get("val", [])))[0]
    # sampler: reach the requested sim fraction regardless of the raw real/sim ratio
    tr_sim = is_sim[train_idx]
    n_sim, n_real = tr_sim.sum(), (~tr_sim).sum()
    if n_sim and n_real:
        w = np.where(tr_sim, sim_weight / n_sim, (1 - sim_weight) / n_real)
    else:
        w = np.ones(len(train_idx))
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=steps * batch_size, replacement=True)
    dl = DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                    pin_memory=dev.type == "cuda", persistent_workers=num_workers > 0)
    low_dim = sum(feats[k]["shape"][0] for k in low_keys)
    act_dim = feats["action"]["shape"][0]
    model = Policy(cams, depth, low_dim, act_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def prep(b):
        imgs = {k: b[k].to(dev, non_blocking=True) for k in cams}
        dmax = 3000.0
        dep = {k: (b[k].to(dev, non_blocking=True) / dmax).clamp(0, 1) for k in depth}
        low = torch.cat([norm(k, b[k].to(dev).float().reshape(len(b[k]), -1)) for k in low_keys], 1)
        tgt = norm("action", b["action"].to(dev).float())
        return imgs, dep, low, tgt

    losses, sim_frac, t0, n_seen = [], [], time.time(), 0
    model.train()
    for step, b in enumerate(dl):
        imgs, dep, low, tgt = prep(b)
        loss = nn.functional.smooth_l1_loss(model(imgs, dep, low), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        if "is_sim" in b:
            sim_frac.append(float(b["is_sim"].float().mean()))
        n_seen += len(tgt)
        if step + 1 >= steps:
            break
    train_time = time.time() - t0

    model.eval()
    val = {"real": [], "sim": []}
    if len(val_idx):
        sel = np.random.default_rng(0).choice(val_idx, min(len(val_idx), 1024), replace=False)
        vdl = DataLoader(Subset(ds, sel.tolist()), batch_size=64, num_workers=num_workers)
        with torch.no_grad():
            for b in vdl:
                imgs, dep, low, tgt = prep(b)
                err = nn.functional.smooth_l1_loss(model(imgs, dep, low), tgt, reduction="none").mean((1, 2)).cpu().numpy()
                sim = b["is_sim"].numpy().ravel().astype(bool) if "is_sim" in b else np.zeros(len(err), bool)
                val["sim"] += err[sim].tolist()
                val["real"] += err[~sim].tolist()
    k = max(1, len(losses) // 10)
    return {
        "dataset": root.name, "device": str(dev), "steps": len(losses), "batch_size": batch_size,
        "cameras": cams, "depth": depth, "low_dim_inputs": low_keys, "action_chunk": [CHUNK, act_dim],
        "train_loss_first10pct": float(np.mean(losses[:k])), "train_loss_last10pct": float(np.mean(losses[-k:])),
        "batch_sim_fraction_mean": float(np.mean(sim_frac)) if sim_frac else None, "requested_sim_fraction": sim_weight,
        "val_loss_real": float(np.mean(val["real"])) if val["real"] else None,
        "val_loss_sim": float(np.mean(val["sim"])) if val["sim"] else None,
        "samples_per_s": n_seen / train_time, "train_episodes": len(ep_split.get("train", [])),
        "val_samples": int(len(val_idx)),
    }
