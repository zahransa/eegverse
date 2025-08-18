\
import os, re
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import mne

RUN_MAP = {
    "R01": ("rest_eyes_open", 0),
    "R02": ("rest_eyes_closed", 0),
    "R03": ("motor_left_fist", 1),
    "R04": ("motor_right_fist", 2),
    "R05": ("motor_both_fists", 3),
    "R06": ("motor_both_feet", 4),
    "R07": ("motor_mixed", 5),
    "R08": ("imag_left_fist", 1),
    "R09": ("imag_right_fist", 2),
    "R10": ("imag_both_fists", 3),
    "R11": ("imag_both_feet", 4),
    "R12": ("imag_mixed", 5),
    "R13": ("rest_alt1", 0),
    "R14": ("rest_alt2", 0),
}

def _extract_run_id_from_path(edf_path: str) -> str:
    m = re.search(r'R(\d{2})', edf_path, flags=re.IGNORECASE)
    if m: return f"R{m.group(1)}"
    m = re.search(r'task-(?:run)?(\d+)_', edf_path, flags=re.IGNORECASE)
    if m: return f"R{int(m.group(1)):02d}"
    return ""

@dataclass
class PhysioNetMI:
    root: str = "./eegverse_data"
    split: str = "train"
    subjects: Optional[List[str]] = None
    preload: bool = True
    epoch_sec: float = 2.0

    def _bids_root(self) -> str:
        return os.path.join(self.root, "physionet_mi")

    def subjects_list(self) -> List[str]:
        if self.subjects: return sorted(self.subjects)
        base = self._bids_root()
        subs = [d.replace("sub-","") for d in os.listdir(base) if d.startswith("sub-")] if os.path.isdir(base) else []
        subs.sort()
        if not subs: raise RuntimeError(f"No subjects found under {base}.")
        return subs

    def _eeg_files(self, sub: str) -> List[str]:
        eeg_dir = os.path.join(self._bids_root(), f"sub-{sub}", "eeg")
        if not os.path.isdir(eeg_dir): return []
        return sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith("_eeg.edf")])

    def _epochize(self, raw: mne.io.BaseRaw, edf_path: str):
        run_id = _extract_run_id_from_path(edf_path)
        trial_type, label_id = RUN_MAP.get(run_id, ("unknown", -1))
        sf = raw.info["sfreq"]; dur = raw.times[-1]
        tmin, tmax = 0.0, min(self.epoch_sec, dur)
        events = np.array([[0, 0, max(label_id,0)]])
        event_id = {trial_type: max(label_id,0)}
        picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, picks=picks,
                            baseline=None, preload=True, verbose="ERROR")
        X = epochs.get_data(); y = epochs.events[:, -1]
        return X, y

    def _split_subjects(self, subs: List[str]) -> List[str]:
        rng = np.random.default_rng(42); idx = rng.permutation(len(subs))
        n = len(subs); n_tr = int(0.7*n); n_val = int(0.15*n)
        if self.split == "train": return [subs[i] for i in idx[:n_tr]]
        if self.split in ("val","valid","validation"): return [subs[i] for i in idx[n_tr:n_tr+n_val]]
        if self.split == "test": return [subs[i] for i in idx[n_tr+n_val:]]
        return subs

    def get_data(self):
        subs = self._split_subjects(self.subjects_list())
        Xs, ys, who = [], [], []
        for sub in subs:
            for edf in self._eeg_files(sub):
                raw = mne.io.read_raw_edf(edf, preload=self.preload, verbose="ERROR")
                X, y = self._epochize(raw, edf)
                if len(y)==0: continue
                Xs.append(X); ys.append(y); who += [sub]*len(y)
        X = np.concatenate(Xs, axis=0) if Xs else np.empty((0,0,0))
        y = np.concatenate(ys, axis=0) if ys else np.empty((0,))
        info = {"subjects": who}
        return X, y, info
