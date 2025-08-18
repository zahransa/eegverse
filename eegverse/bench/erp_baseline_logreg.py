#!/usr/bin/env python3
import os, json, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import eegverse as ev

def loso(subjects): return {s: {"train":[x for x in subjects if x!=s], "test":[s]} for s in subjects}

def main(root="./eegverse_data", dataset_id="ds000117", include_types=None):
    base = ev.load("openneuro_erp", root=root, dataset_id=dataset_id, split="train", include_types=include_types)
    subs = base.subjects_list(); parts = loso(subs)
    fold_acc, per = [], []
    for held in subs:
        tr = ev.load("openneuro_erp", root=root, dataset_id=dataset_id, subjects=parts[held]["train"], split="train", include_types=include_types)
        te = ev.load("openneuro_erp", root=root, dataset_id=dataset_id, subjects=parts[held]["test"], split="test", include_types=include_types)
        Xtr,ytr,_ = tr.get_data(); Xte,yte,_ = te.get_data()
        if Xtr.size==0 or Xte.size==0 or len(np.unique(ytr))<2: print(f"skip {held}"); continue
        pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                         ("clf", LogisticRegression(max_iter=200))])
        pipe.fit(Xtr.reshape(len(Xtr),-1), ytr)
        acc = pipe.score(Xte.reshape(len(Xte),-1), yte)
        fold_acc.append(acc); per.append({"held_out": held, "acc": float(acc)}); print(f"LOSO {held}: acc={acc:.3f}")
    os.makedirs("results", exist_ok=True)
    out = {"dataset":dataset_id, "task":"ERP binary subset" if include_types else "ERP (top-2 auto)",
           "model":"LogReg (flatten+std)", "metric":"Acc (LOSO)",
           "folds":per, "mean_acc":float(np.mean(fold_acc)) if fold_acc else None,
           "std_acc":float(np.std(fold_acc)) if fold_acc else None}
    with open("results/erp_logreg.json","w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    print("Saved: results/erp_logreg.json")

if __name__=="__main__":
    main()
