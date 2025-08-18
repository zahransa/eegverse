#!/usr/bin/env python3
import os, re, mne
from mne_bids import BIDSPath, write_raw_bids

RAW = "./eegverse_raw/physionet_mi"
BIDS = "./eegverse_data/physionet_mi"
os.makedirs(BIDS, exist_ok=True)

def subject_id(fn):  # e.g., S001R03.edf -> 001
    m = re.search(r"S(\d{3})", fn); return m.group(1) if m else "000"

for root,_,files in os.walk(RAW):
    for f in files:
        if f.lower().endswith(".edf"):
            p = os.path.join(root,f)
            sub = subject_id(f)
            raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")
            bids = BIDSPath(subject=sub, task="motor", run=None, root=BIDS, datatype="eeg", suffix="eeg")
            write_raw_bids(raw, bids, overwrite=True, verbose="ERROR")
            print("✔", f"sub-{sub}", f)
print("BIDS at", BIDS)
