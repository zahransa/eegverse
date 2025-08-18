\
import os, re
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import mne

SUPPORTED_EXTS = (".edf", ".bdf", ".vhdr", ".set")

def _read_raw_auto(path: str, preload=True):
    p = path.lower()
    if p.endswith(".edf"): return mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    if p.endswith(".bdf"): return mne.io.read_raw_bdf(path, preload=preload, verbose="ERROR")
    if p.endswith(".vhdr"): return mne.io.read_raw_brainvision(path, preload=preload, verbose="ERROR")
    if p.endswith(".set"): return mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    raise ValueError(f"Unsupported EEG file: {path}")

@dataclass
class OpenNeuroERP:
    root: str = "./eegverse_data"
    dataset_id: str = "ds000117"
    split: str = "train"
    subjects: Optional[List[str]] = None
    include_types: Optional[List[str]] = None
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: Tuple[Optional[float], Optional[float]] = (None, 0.0)
    sfreq_target: Optional[float] = None
    picks: Optional[List[str]] = None

    def _bids_root(self): return os.path.join(self.root, "openneuro", self.dataset_id)

    def subjects_list(self) -> List[str]:
        if self.subjects: return sorted(self.subjects)
        base = self._bids_root()
        subs = [d.replace("sub-","") for d in os.listdir(base) if d.startswith("sub-")] if os.path.isdir(base) else []
        subs.sort()
        if not subs: raise RuntimeError(f"BIDS root not found or empty: {base}")
        return subs

    def _eeg_files(self, sub: str) -> List[str]:
        base = os.path.join(self._bids_root(), f"sub-{sub}")
        eeg_files = []
        for r,_,fs in os.walk(base):
            for f in fs:
                if f.lower().endswith(SUPPORTED_EXTS):
                    eeg_files.append(os.path.join(r,f))
        eeg_files.sort(); return eeg_files

    def _events_df(self, eeg_path: str):
        tsv = re.sub(r"(_eeg)?\.(edf|bdf|vhdr|set)$", "_events.tsv", eeg_path, flags=re.IGNORECASE)
        if os.path.isfile(tsv):
            try: return pd.read_csv(tsv, sep="\t")
            except Exception: return None
        return None

    def _label_column(self, df: pd.DataFrame):
        for col in ["trial_type","stim_type","value","condition","type"]:
            if col in df.columns: return col
        return None

    def _select_types(self, series: pd.Series):
        if self.include_types:
            labels = series.astype(str).str.lower()
            y = np.full(len(labels), -1, int); mapping = {}
            for i, pat in enumerate(self.include_types):
                mask = labels.str.contains(pat.lower(), regex=False)
                y[mask.values] = i; mapping[pat] = i
            keep = y>=0
            return y[keep], mapping, keep.values
        counts = series.value_counts()
        if len(counts)<2:
            return np.zeros(len(series), int), {str(counts.index[0]):0}, np.ones(len(series), bool)
        top2 = list(counts.index[:2])
        y = series.replace({top2[0]:0, top2[1]:1}).astype(int).values
        mapping = {str(top2[0]):0, str(top2[1]):1}
        keep = series.isin(top2).values
        return y, mapping, keep

    def _epochize(self, raw: mne.io.BaseRaw, events_df: Optional[pd.DataFrame]):
        raw = raw.copy()
        if self.sfreq_target and abs(raw.info["sfreq"]-self.sfreq_target)>1e-6:
            raw.resample(self.sfreq_target)
        if self.picks:
            picks = mne.pick_channels(raw.ch_names, include=self.picks); raw.pick(picks)
        else:
            raw.pick(mne.pick_types(raw.info, eeg=True, meg=False, eog=False, stim=False))
        if events_df is not None and "onset" in events_df.columns:
            label_col = self._label_column(events_df)
            if label_col is not None:
                y_all, mapping, keep = self._select_types(events_df[label_col])
                on = events_df["onset"].to_numpy(float)[keep]
                sf = raw.info["sfreq"]; samples = (on*sf).round().astype(int)
                uniq = np.unique(y_all); code_map = {cls:(i+1) for i,cls in enumerate(sorted(uniq))}
                codes = np.array([code_map[c] for c in y_all], int)
                events = np.c_[samples, np.zeros_like(samples), codes]
                event_id = {f"class_{cls}": code_map[cls] for cls in uniq}
                epochs = mne.Epochs(raw, events, event_id, tmin=self.tmin, tmax=self.tmax,
                                    baseline=self.baseline, preload=True, verbose="ERROR")
                X = epochs.get_data()
                order = np.argsort(samples); y_sorted = y_all[order]
                return X, y_sorted, {"mapping": mapping, "event_id": event_id}
        ev, idmap = mne.events_from_annotations(raw, verbose="ERROR")
        if ev.size==0:
            return np.empty((0, raw.info["nchan"], int((self.tmax-self.tmin)*raw.info["sfreq"]))), np.empty((0,), int), {}
        epochs = mne.Epochs(raw, ev, idmap, tmin=self.tmin, tmax=self.tmax, baseline=self.baseline, preload=True, verbose="ERROR")
        X = epochs.get_data(); y = epochs.events[:, -1]
        return X, y, {"event_id": idmap}

    def get_data(self):
        subs = self.subjects_list()
        rng = np.random.default_rng(42); perm = rng.permutation(len(subs))
        n=len(subs); n_tr=max(1,int(0.7*n)); n_val=max(1,int(0.15*n))
        if self.split=="train": subs=[subs[i] for i in perm[:n_tr]]
        elif self.split in ("val","valid","validation"): subs=[subs[i] for i in perm[n_tr:n_tr+n_val]]
        elif self.split=="test": subs=[subs[i] for i in perm[n_tr+n_val:]]
        allX, ally, who = [], [], []
        for s in subs:
            for eeg in self._eeg_files(s):
                raw = _read_raw_auto(eeg, preload=True)
                df = self._events_df(eeg)
                X, y, _ = self._epochize(raw, df)
                if len(y)==0: continue
                allX.append(X); ally.append(y); who += [s]*len(y)
        X = np.concatenate(allX,0) if allX else np.empty((0,0,0))
        y = np.concatenate(ally,0) if ally else np.empty((0,))
        return X, y, {"subjects": who}
