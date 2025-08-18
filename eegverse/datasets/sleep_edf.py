\
import os
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import mne

STAGE_MAP = {"W":0,"N1":1,"N2":2,"N3":3,"R":4}
IGNORE = {"?","MOVEMENT","MT","nan",""}

@dataclass
class SleepEDF:
    root: str = "./eegverse_data"
    split: str = "train"
    subjects: Optional[List[str]] = None
    epoch_sec: float = 30.0
    sfreq_target: Optional[float] = 100.0
    picks: Optional[List[str]] = None

    def _bids_root(self): return os.path.join(self.root, "sleep_edf")

    def subjects_list(self) -> List[str]:
        if self.subjects: return sorted(self.subjects)
        base = self._bids_root()
        subs = [d.replace("sub-","") for d in os.listdir(base) if d.startswith("sub-")] if os.path.isdir(base) else []
        subs.sort()
        if not subs: raise RuntimeError("No Sleep-EDF subjects found. Run conversion first.")
        return subs

    def _edf_files(self, sub: str) -> List[str]:
        eeg_dir = os.path.join(self._bids_root(), f"sub-{sub}", "eeg")
        if not os.path.isdir(eeg_dir): return []
        return sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith("_eeg.edf")])

    def _stage_from_desc(self, d: str):
        if not isinstance(d, str): return None
        u = d.strip().upper().replace("STAGE ","").replace("SLEEP ","")
        if u in IGNORE: return None
        if u in ("R","REM"): return 4
        if u in ("W","WAKE","N0"): return 0
        if u in ("N1","S1"): return 1
        if u in ("N2","S2"): return 2
        if u in ("N3","S3","S4","N4"): return 3
        return STAGE_MAP.get(u, None)

    def _epochize(self, raw: mne.io.BaseRaw):
        if self.sfreq_target and abs(raw.info["sfreq"]-self.sfreq_target)>1e-6:
            raw = raw.copy().resample(self.sfreq_target)
        if self.picks:
            picks = mne.pick_channels(raw.ch_names, include=self.picks); raw = raw.pick(picks)
        else:
            raw = raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False))
        sf = raw.info["sfreq"]; nper = int(round(self.epoch_sec*sf))
        X = raw.get_data(); T = X.shape[1]; n_epochs = T//nper
        if n_epochs==0: return np.empty((0, raw.info["nchan"], nper)), np.empty((0,), int)
        ann = raw.annotations or []
        stage_samples = np.full(T, -1, int)
        for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
            s = max(0, int(round(onset*sf))); e = min(T, s + int(round(duration*sf)))
            lab = self._stage_from_desc(desc)
            if lab is not None: stage_samples[s:e] = lab
        xs, ys = [], []
        for k in range(n_epochs):
            s = k*nper; e = s + nper
            vals, cnts = np.unique(stage_samples[s:e][stage_samples[s:e]>=0], return_counts=True)
            if len(vals)==0: continue
            y = int(vals[np.argmax(cnts)]); xs.append(X[:, s:e]); ys.append(y)
        if not xs: return np.empty((0, raw.info["nchan"], nper)), np.empty((0,), int)
        return np.stack(xs,0), np.array(ys,int)

    def get_data(self):
        subs = self.subjects_list()
        rng = np.random.default_rng(42); idx = rng.permutation(len(subs))
        n=len(subs); n_tr=int(0.7*n); n_val=int(0.15*n)
        if self.split=="train": subs=[subs[i] for i in idx[:n_tr]]
        elif self.split in ("val","valid","validation"): subs=[subs[i] for i in idx[n_tr:n_tr+n_val]]
        elif self.split=="test": subs=[subs[i] for i in idx[n_tr+n_val:]]
        allX, ally, who = [], [], []
        for sub in subs:
            for edf in self._edf_files(sub):
                raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
                X, y = self._epochize(raw)
                if len(y)==0: continue
                allX.append(X); ally.append(y); who += [sub]*len(y)
        X = np.concatenate(allX,0) if allX else np.empty((0,0,0))
        y = np.concatenate(ally,0) if ally else np.empty((0,))
        return X, y, {"subjects": who, "stage_map": STAGE_MAP}
