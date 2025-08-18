#!/usr/bin/env python3
import os, shutil
from pathlib import Path

RAW = Path("./eegverse_raw/openneuro/ds000117")
DST = Path("./eegverse_data/openneuro/ds000117")
DST.mkdir(parents=True, exist_ok=True)

def copy_tree(src: Path, dst: Path, exts=None):
    for p in src.rglob("*"):
        rel = p.relative_to(src); tgt = dst / rel
        if p.is_dir():
            tgt.mkdir(parents=True, exist_ok=True)
        else:
            if exts and p.suffix.lower() not in exts: continue
            try:
                if tgt.exists(): continue
                os.symlink(p.resolve(), tgt)
            except Exception:
                shutil.copy2(p, tgt)

if not RAW.exists():
    raise SystemExit("Raw OpenNeuro path not found. Run fetch script first.")

copy_tree(RAW, DST, exts=None)
print("Mirrored to", DST)
