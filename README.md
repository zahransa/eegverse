[![PyPI version](https://img.shields.io/pypi/v/eegverse.svg)](https://pypi.org/project/eegverse/)
[![Python](https://img.shields.io/pypi/pyversions/eegverse.svg)](https://pypi.org/project/eegverse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# 🧠 EEGVERSE
*A Universe of Annotated EEG Data for Benchmarks & Reproducible Research*

## Quickstart
```bash
pip install -e .

# Motor Imagery
python scripts/fetch_physionet_mi.py
python scripts/convert_physionet_mi_to_bids.py
python eegverse/bench/mi_baseline.py

# Sleep (Sleep-EDF)
python scripts/fetch_sleep_edf.py
python scripts/convert_sleep_edf_to_bids.py
python eegverse/bench/sleep_baseline_cnn.py

# ERP (OpenNeuro ds000117 example)
python scripts/fetch_openneuro_ds000117.py
python scripts/convert_openneuro_to_bids.py
python eegverse/bench/erp_baseline_logreg.py
```

Load data in Python:
```python
import eegverse as ev
ds = ev.load("mi", root="./eegverse_data")
X, y, info = ds.get_data()
print(X.shape, y.shape)
```
