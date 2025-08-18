# CHB-MIT — Dataset Card

**Domain:** Clinical (Seizure Detection)  
**Source:** PhysioNet (CHB-MIT Scalp EEG Database)  
**License:** PhysioNet credentialed

## Notes
- CHB-MIT seizure spans are not embedded in EDF; they are provided in separate annotation files.
- `scripts/convert_chbmit_to_bids.py` will create `*_events.tsv` if it finds annotations (or a user-provided CSV).
- Without events, `eegverse/bench/baseline_seizure_cnn.py` may skip folds (needs both classes).
