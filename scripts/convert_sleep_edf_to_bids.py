#!/usr/bin/env python3
import os, re, mne
from mne_bids import BIDSPath, write_raw_bids

RAW = "./eegverse_raw/sleep_edf"
BIDS = "./eegverse_data/sleep_edf"
os.makedirs(BIDS, exist_ok=True)

def sub_from(fn):
    m = re.search(r"(\d{4})", fn); return m.group(1) if m else "0000"

pairs = [("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf")]

for psg, hyp in pairs:
    psg_path = os.path.join(RAW, psg); hyp_path = os.path.join(RAW, hyp)
    sub = sub_from(psg)
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    ann = mne.read_annotations(hyp_path, verbose="ERROR"); raw.set_annotations(ann, emit_warning=False)
    bids = BIDSPath(subject=sub, task="sleep", root=BIDS, datatype="eeg", suffix="eeg")
    write_raw_bids(raw, bids, overwrite=True, verbose="ERROR")
    print("✔", f"sub-{sub}")
print("BIDS at", BIDS)
