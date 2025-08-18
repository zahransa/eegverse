\
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import mne

@dataclass
class CHBMIT:
    root: str = "./eegverse_data"
    subjects: Optional[List[str]] = None
    win_sec: float = 4.0
    stride_sec: float = 2.0
    pos_overlap: float = 0.5
    preload: bool = True
    picks: Optional[List[str]] = None
    sfreq_target: Optional[float] = 256.0

    def _bids_root(self): return os.path.join(self.root, "chbmit")

    def subjects_list(self) -> List[str]:
        if self.subjects: return sorted(self.subjects)
        base = self._bids_root()
        subs = [d.replace("sub-","") for d in os.listdir(base) if d.startswith("sub-")] if os.path.isdir(base) else []
        subs.sort()
        if not subs: raise RuntimeError("No CHB-MIT subjects found. Run conversion first.")
        return subs

    def _recordings_for_subject(self, sub: str) -> List[str]:
        ed = os.path.join(self._bids_root(), f"sub-{sub}", "eeg")
        if not os.path.isdir(ed): return []
        return sorted([os.path.join(ed, f) for f in os.listdir(ed) if f.endswith("_eeg.edf")])

    def _load_events(self, edf_path: str) -> List[Tuple[float,float]]:
        events_path = edf_path.replace("_eeg.edf", "_events.tsv")
        if not os.path.isfile(events_path): return []
        df = pd.read_csv(events_path, sep="\t")
        spans = []
        for _, r in df.iterrows():
            if str(r.get("trial_type","")).lower()=="seizure":
                spans.append((float(r["onset"]), float(r["duration"])))
        return spans

    def _windowize(self, raw: mne.io.BaseRaw, seizures: List[Tuple[float,float]]):
        sf = raw.info["sfreq"]
        if self.sfreq_target and abs(self.sfreq_target - sf) > 1e-6:
            raw = raw.copy().resample(self.sfreq_target); sf = raw.info["sfreq"]
        if self.picks:
            picks = mne.pick_channels(raw.ch_names, include=self.picks); raw = raw.pick(picks)
        else:
            raw = raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False))
        X = raw.get_data(); T = X.shape[1]/sf
        win, stride = self.win_sec, self.stride_sec
        n_steps = max(0, int(np.floor((T - win)/stride)) + 1)
        seiz_intervals = [(s, s+d) for s,d in seizures]

        def overlap(a,b,c,d): return max(0.0, min(b,d)-max(a,c))

        xs, ys = [], []
        for k in range(n_steps):
            t0 = k*stride; t1 = t0 + win
            inter = 0.0
            for s,e in seiz_intervals: inter = max(inter, overlap(t0,t1,s,e))
            y = 1 if (inter/win) >= self.pos_overlap else 0
            s_idx = int(round(t0*sf)); e_idx = s_idx + int(round(win*sf))
            if e_idx <= X.shape[1]: xs.append(X[:, s_idx:e_idx]); ys.append(y)
        if not xs: return np.empty((0, raw.info["nchan"], int(win*sf))), np.empty((0,), int)
        return np.stack(xs, axis=0), np.array(ys, int)

    def get_data(self):
        allX, ally, who = [], [], []
        for sub in self.subjects_list():
            for edf in self._recordings_for_subject(sub):
                raw = mne.io.read_raw_edf(edf, preload=self.preload, verbose="ERROR")
                spans = self._load_events(edf)
                Xw, yw = self._windowize(raw, spans)
                if len(yw)==0: continue
                allX.append(Xw); ally.append(yw); who += [sub]*len(yw)
        X = np.concatenate(allX, axis=0) if allX else np.empty((0,0,0))
        y = np.concatenate(ally, axis=0) if ally else np.empty((0,))
        info = {"subjects": who}
        return X, y, info
