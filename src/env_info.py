"""Record the software/hardware environment so every result is traceable to what produced it."""
import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import datetime, timezone

PACKAGES = ["mlx", "mlx-lm", "mlx-metal", "transformers", "datasets", "sqlglot", "numpy"]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def env_info() -> dict:
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    mem = _run(["sysctl", "-n", "hw.memsize"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "ram_gb": round(int(mem) / 1e9, 1) if mem and mem.isdigit() else None,
        "packages": pkgs,
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(env_info(), indent=2))
