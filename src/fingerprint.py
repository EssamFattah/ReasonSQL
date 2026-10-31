"""Experiment fingerprints: resumability without silently mixing configurations.

Every stage (teacher generation, dataset build, training, evaluation, forgetting) writes a
`*.spec.json` next to its outputs containing the settings that determine those outputs and a
hash of them. On re-run the stage recomputes the spec:
  * same fingerprint      -> safe to skip / resume
  * different fingerprint -> refuse, and list the settings that differ

Upstream fingerprints are part of downstream specs (dataset <- traces, adapter <- dataset,
eval <- adapter), so changing anything upstream invalidates everything that depends on it.
Environment details (package versions, hardware) are recorded but deliberately not hashed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.common import read_json, write_json


class IncompatibleRun(SystemExit):
    pass


def fingerprint(spec: dict) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:12]


def file_sha(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_fingerprint(spec_path) -> str | None:
    s = read_json(spec_path)
    return s["fingerprint"] if s else None


def _diff(old: dict, new: dict, prefix="") -> list[str]:
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff(a, b, f"{prefix}{k}.")
        elif a != b:
            out.append(f"{prefix}{k}: {a!r} -> {b!r}")
    return out


def check(spec_path, spec: dict, what: str) -> tuple[str, bool]:
    """Returns (fingerprint, existed). Raises IncompatibleRun if outputs exist with another spec."""
    fp = fingerprint(spec)
    old = read_json(spec_path)
    if old is None:
        return fp, False
    if old["fingerprint"] != fp:
        diffs = "\n  ".join(_diff(old["spec"], spec)) or "(spec structure changed)"
        raise IncompatibleRun(
            f"\n[{what}] {Path(spec_path).parent} was produced with different settings:\n  {diffs}\n"
            "Refusing to reuse or overwrite it. Use a new output directory (e.g. --runs_dir runs_v2) "
            "or delete the old one if you really mean to replace it.")
    return fp, True


# Same file patterns mlx_lm downloads, so resolving a revision never fetches anything extra.
_HF_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jsonl"]


_SNAPSHOTS: dict[str, tuple[str | None, str | None]] = {}


def split_revision(model: str) -> tuple[str, str | None]:
    """'org/name@<commit-or-tag>' pins a revision explicitly; plain 'org/name' means the current main."""
    repo, _, rev = model.partition("@")
    return repo, (rev or None)


def resolve_hf_snapshot(model: str) -> tuple[str | None, str | None]:
    """(local snapshot path, commit hash) for a Hugging Face model, resolved once per process.

    Cache first, download (with mlx_lm's file patterns) if needed. The snapshot directory name
    *is* the commit hash, so the path we load from and the revision we record can't disagree.
    """
    if model in _SNAPSHOTS:
        return _SNAPSHOTS[model]
    repo, rev = split_revision(model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None, None
    path = None
    for local_only in (True, False):
        try:
            path = snapshot_download(repo, revision=rev, allow_patterns=_HF_PATTERNS, local_files_only=local_only)
            break
        except Exception:  # noqa: BLE001
            continue
    _SNAPSHOTS[model] = (path, Path(path).name if path else None)
    return _SNAPSHOTS[model]


def resolve_hf_revision(model: str) -> str | None:
    return resolve_hf_snapshot(model)[1]


def resolve_model_path(model: str) -> str:
    """What every loader should receive: a local dir as-is, or the exact pinned HF snapshot path."""
    if Path(model).exists():
        return model
    path, _ = resolve_hf_snapshot(model)
    if path is None:
        raise RuntimeError(f"could not resolve Hugging Face model {model!r} to a local snapshot "
                           "(no network and not cached?)")
    return path


def weights_sha(path: Path) -> str:
    return fingerprint({f.name: file_sha(f) for f in sorted(Path(path).glob("*.safetensors"))})


def model_identity(model: str) -> dict:
    """Local dir -> hash of its weights; Hugging Face repo -> repo + the commit that resolve_model_path loads."""
    p = Path(model)
    if p.exists():
        return {"path": str(p), "weights_sha": weights_sha(p)}
    repo, requested = split_revision(model)
    return {"repo": repo, "requested_revision": requested, "revision": resolve_hf_snapshot(model)[1]}


def adapter_identity(adapter_path: str | None) -> dict | None:
    """Training-spec fingerprint AND a hash of the actual weights: either changing invalidates evals."""
    if not adapter_path:
        return None
    a = Path(adapter_path)
    return {"spec": read_fingerprint(a / "spec.json"), "weights_sha": file_sha(a / "adapters.safetensors"),
            "config_sha": file_sha(a / "adapter_config.json")}


def sqlite_dir_identity(root, db_ids=None) -> dict | None:
    """Content hashes of the SQLite files under root/<db_id>/ (only the given db_ids, if provided)."""
    if root is None:
        return None
    root = Path(root)
    dirs = sorted(root / d for d in set(db_ids)) if db_ids is not None else sorted(p for p in root.iterdir() if p.is_dir())
    return {"path": str(root),
            "content_sha": fingerprint({f"{d.name}/{f.name}": file_sha(f) for d in dirs for f in sorted(d.glob("*.sqlite"))})}


def save(spec_path, spec: dict) -> str:
    fp = fingerprint(spec)
    write_json(spec_path, {"fingerprint": fp, "spec": spec})
    return fp
