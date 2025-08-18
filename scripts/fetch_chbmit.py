#!/usr/bin/env python3
import os, pooch
BASE = "https://physionet.org/static/published-projects/chbmit/1.0.0/"
OUT = "./eegverse_raw/chbmit"; os.makedirs(OUT, exist_ok=True)
FILES = ["chb01/chb01_03.edf", "chb01/chb01_04.edf"]
for rel in FILES:
    local = pooch.retrieve(url=BASE+rel, known_hash=None, path=OUT); print("✔", local)
print("Done. (Note: annotations for seizures may need manual CSV; see DATASETS/CHBMIT_CARD.md)")
