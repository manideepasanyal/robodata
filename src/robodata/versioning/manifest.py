"""Content-addressed dataset versions.

A version = sha256 of every file + a Merkle-style root hash + the lineage needed to
rebuild it (pinned source revisions, config hashes, code version). Experiments record the
version id; ``robodata verify`` proves the bytes on disk are the ones that were trained on.
Works standalone; DVC (``dvc add data/datasets/<name>``) can track the same directory for
remote storage - see docs/VERSIONING.md.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from robodata.config import CONFIG_DIR, DATA_ROOT, REPO_ROOT


def sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            out[str(p.relative_to(root))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def root_hash(files: dict[str, dict]) -> str:
    h = hashlib.sha256()
    for k in sorted(files):
        h.update(f"{k}\0{files[k]['sha256']}\n".encode())
    return h.hexdigest()


def _code_version() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5)
        if rev.returncode == 0:
            return {"git_commit": rev.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except Exception:
        pass
    src = REPO_ROOT / "src" / "robodata"
    h = hashlib.sha256()
    for p in sorted(src.rglob("*.py")):
        h.update(p.read_bytes())
    return {"source_tree_sha256": h.hexdigest()}


def _config_hashes() -> dict:
    return {str(p.relative_to(CONFIG_DIR)): sha256_file(p)[:16] for p in sorted(CONFIG_DIR.rglob("*.yaml"))}


def create_version(dataset_dir: str | Path, versions_dir: str | Path, note: str = "") -> dict:
    dataset_dir = Path(dataset_dir)
    files = file_manifest(dataset_dir)
    rh = root_hash(files)
    lock = DATA_ROOT / "sources" / "lock.json"
    manifest = {
        "dataset": dataset_dir.name, "version": rh[:16], "root_hash": rh,
        "created": datetime.now(timezone.utc).isoformat(), "note": note,
        "n_files": len(files), "total_bytes": sum(f["bytes"] for f in files.values()),
        "lineage": {"code": _code_version(), "configs": _config_hashes(),
                    "sources": json.loads(lock.read_text()) if lock.exists() else {}},
        "files": files,
    }
    out = Path(versions_dir) / dataset_dir.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rh[:16]}.json").write_text(json.dumps(manifest, indent=1))
    (out / "LATEST").write_text(rh[:16])
    (dataset_dir / "meta" / "robodata").mkdir(parents=True, exist_ok=True)
    return {k: v for k, v in manifest.items() if k != "files"}


def verify(dataset_dir: str | Path, versions_dir: str | Path, version: str | None = None) -> dict:
    dataset_dir = Path(dataset_dir)
    vdir = Path(versions_dir) / dataset_dir.name
    version = version or (vdir / "LATEST").read_text().strip()
    manifest = json.loads((vdir / f"{version}.json").read_text())
    now = file_manifest(dataset_dir)
    missing = sorted(set(manifest["files"]) - set(now))
    extra = sorted(set(now) - set(manifest["files"]))
    changed = sorted(k for k in set(now) & set(manifest["files"]) if now[k]["sha256"] != manifest["files"][k]["sha256"])
    return {"version": version, "ok": not (missing or extra or changed), "missing": missing, "extra": extra,
            "changed": changed, "root_hash_now": root_hash(now), "root_hash_expected": manifest["root_hash"]}
