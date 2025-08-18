#!/usr/bin/env python3
import os, re, csv, mne
from mne_bids import BIDSPath, write_raw_bids
from pathlib import Path

RAW = Path("./eegverse_raw/chbmit")
BIDS = Path("./eegverse_data/chbmit"); BIDS.mkdir(parents=True, exist_ok=True)

CSV = RAW / "seizures.csv"  # optional: user-supplied CSV

def subject_from_path(p: Path):
    m = re.search(r"chb(\d{2})", str(p)); return m.group(1) if m else "00"

ann_map = {}
if CSV.exists():
    import pandas as pd
    df = pd.read_csv(CSV)
    for _, row in df.iterrows():
        rec = row["record"]; onset = float(row["onset_sec"]); dur = float(row["duration_sec"])
        ann_map.setdefault(rec, []).append((onset, dur))

for p in RAW.rglob("*.edf"):
    sub = subject_from_path(p)
    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
    bids = BIDSPath(subject=sub, task="monitoring", root=str(BIDS), datatype="eeg", suffix="eeg")
    write_raw_bids(raw, bids, overwrite=True, verbose="ERROR")
    if p.name in ann_map:
        import pandas as pd
        ev = [{"onset": o, "duration": d, "trial_type": "seizure"} for o,d in ann_map[p.name]]
        ev_path = BIDS / f"sub-{sub}" / "eeg" / p.name.replace(".edf", "_events.tsv")
        pd.DataFrame(ev).to_csv(ev_path, sep="\t", index=False)
        print("events:", ev_path)
    print("✔ sub-", sub, p.name)
print("BIDS at", BIDS)
