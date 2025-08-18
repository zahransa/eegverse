#!/usr/bin/env python3
import os, json, numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
import eegverse as ev

def loso(subjects): return {s: {"train":[x for x in subjects if x!=s], "test":[s]} for s in subjects}

def run(root="./eegverse_data", n_components=6, seed=42):
    base = ev.load("mi", root=root, split="train")
    subs = base.subjects_list(); parts = loso(subs)
    fold_acc, per = [], []
    for held in subs:
        tr_ds = ev.load("mi", root=root, subjects=parts[held]["train"], split="train")
        te_ds = ev.load("mi", root=root, subjects=parts[held]["test"], split="test")
        Xtr, ytr, _ = tr_ds.get_data(); Xte, yte, _ = te_ds.get_data()
        if len(np.unique(ytr))<2 or Xte.shape[0]==0: print(f"Skip {held}"); continue
        csp = CSP(n_components=n_components, random_state=seed)
        Xtr_f = csp.fit_transform(Xtr, ytr); Xte_f = csp.transform(Xte)
        clf = LinearDiscriminantAnalysis().fit(Xtr_f, ytr)
        acc = accuracy_score(yte, clf.predict(Xte_f))
        print(f"LOSO held-out sub-{held}: acc={acc:.3f}")
        fold_acc.append(acc); per.append({"held_out": held, "acc": float(acc)})
    os.makedirs("results", exist_ok=True)
    out = {"dataset":"physionet_mi","model":"CSP+LDA","metric":"Acc (LOSO)",
           "folds":per,"mean_acc":float(np.mean(fold_acc)) if fold_acc else None,
           "std_acc":float(np.std(fold_acc)) if fold_acc else None}
    with open("results/mi_baseline.json","w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    print("Saved: results/mi_baseline.json")

if __name__=="__main__":
    run()
