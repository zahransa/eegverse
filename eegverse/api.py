from .datasets.physionet_mi import PhysioNetMI
from .datasets.chbmit import CHBMIT
from .datasets.sleep_edf import SleepEDF
from .datasets.openneuro_erp import OpenNeuroERP

def load(dataset: str, root: str = "./eegverse_data", split: str = "train", **kwargs):
    name = dataset.lower()
    if name in ("mi", "physionet_mi", "eegmmidb"):
        return PhysioNetMI(root=root, split=split, **kwargs)
    if name in ("chbmit", "seizure"):
        return CHBMIT(root=root, **kwargs)
    if name in ("sleep_edf", "sleep", "sleep-edfx"):
        return SleepEDF(root=root, split=split, **kwargs)
    if name.startswith("openneuro") or name in ("erp", "openneuro_erp", "ds000117"):
        return OpenNeuroERP(root=root, split=split, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset!r}. Try 'mi', 'chbmit', 'sleep_edf', or 'openneuro_erp'.")
