#!/usr/bin/env python3
import os
from pathlib import Path
OUT = Path("./eegverse_raw/openneuro/ds000117"); OUT.mkdir(parents=True, exist_ok=True)
print("Downloading OpenNeuro ds000117 subset (sub-01)...")
try:
    from openneuro import download
except Exception as e:
    raise SystemExit("Please `pip install openneuro-py` first.") from e
download(dataset="ds000117", target_dir=str(OUT), include=["sub-01"])
print("Done:", OUT)
