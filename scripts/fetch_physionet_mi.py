#!/usr/bin/env python3
import os, pooch
BASE = "https://physionet.org/static/published-projects/eegmmidb/1.0.0/"
OUT = "./eegverse_raw/physionet_mi"; os.makedirs(OUT, exist_ok=True)
FILES = ["S001/S001R01.edf","S001/S001R02.edf","S001/S001R03.edf","S001/S001R04.edf"]
for rel in FILES:
    local = pooch.retrieve(url=BASE+rel, known_hash=None, path=OUT); print("✔", local)
print("Done.")
