from pathlib import Path
import os

CANONICAL_ROOT = Path("/home/ubuntu/chad_finale")

def get_root() -> Path:
    env = os.environ.get("CHAD_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "chad").exists():
            return p
    return CANONICAL_ROOT

def runtime_dir() -> Path:
    return get_root() / "runtime"
