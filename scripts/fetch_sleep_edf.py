#!/usr/bin/env python3
import os, pooch
BASE = "https://physionet.org/static/published-projects/sleep-edfx/1.0.0/"
OUT = "./eegverse_raw/sleep_edf"; os.makedirs(OUT, exist_ok=True)
FILES = ["sleep-cassette/SC4001E0-PSG.edf", "sleep-cassette/SC4001EC-Hypnogram.edf"]
for rel in FILES:
    local = pooch.retrieve(url=BASE+rel, known_hash=None, path=OUT); print("✔", local)
print("Done.")
