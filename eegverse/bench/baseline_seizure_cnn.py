#!/usr/bin/env python3
import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import eegverse as ev

class TinyCNN(nn.Module):
    def __init__(self, C, T):
        super().__init__(); self.conv1=nn.Conv1d(C,16,7,padding=3); self.bn1=nn.BatchNorm1d(16)
        self.conv2=nn.Conv1d(16,32,7,padding=3); self.bn2=nn.BatchNorm1d(32)
        self.pool=nn.AdaptiveAvgPool1d(1); self.fc=nn.Linear(32,1); self.act=nn.Sigmoid()
    def forward(self,x):
        x=torch.relu(self.bn1(self.conv1(x))); x=torch.relu(self.bn2(self.conv2(x))); x=self.pool(x).squeeze(-1); return self.act(self.fc(x))

def train_epoch(m, loader, opt, dev):
    m.train(); crit=nn.BCELoss(); losses=[]
    for xb,yb in loader:
        xb=xb.to(dev); yb=yb.float().to(dev); opt.zero_grad(); out=m(xb).squeeze(-1); loss=crit(out,yb)
        loss.backward(); opt.step(); losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def eval_auc(m, loader, dev):
    from sklearn.metrics import roc_auc_score
    m.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(dev); p=m(xb).squeeze(-1).cpu().numpy(); ps.append(p); ys.append(yb.numpy())
    if not ys: return 0.0
    y=np.concatenate(ys); p=np.concatenate(ps)
    if len(np.unique(y))<2: return float(np.mean(y))
    return float(roc_auc_score(y,p))

def main(root="./eegverse_data"):
    base_ds = ev.load("chbmit", root=root)
    X, y, info = base_ds.get_data()
    import numpy as np
    subs = np.array(info["subjects"]); subjects_unique = sorted(set(subs))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    fold_aucs=[]
    for held in subjects_unique:
        tr = subs!=held; te = subs==held
        Xtr, ytr = X[tr], y[tr]; Xte, yte = X[te], y[te]
        if Xtr.size==0 or Xte.size==0 or len(np.unique(ytr))<2: print(f"skip {held}"); continue
        C,T = Xtr.shape[1], Xtr.shape[2]
        m=TinyCNN(C,T).to(dev); opt=optim.Adam(m.parameters(), lr=1e-3)
        tr_loader=DataLoader(TensorDataset(torch.tensor(Xtr,float), torch.tensor(ytr,float)), batch_size=128, shuffle=True)
        te_loader=DataLoader(TensorDataset(torch.tensor(Xte,float), torch.tensor(yte,float)), batch_size=256, shuffle=False)
        for _ in range(5): train_epoch(m, tr_loader, opt, dev)
        auc = eval_auc(m, te_loader, dev); fold_aucs.append(auc); print(f"LOPO {held}: AUROC={auc:.3f}")
    os.makedirs("results", exist_ok=True)
    out = {"dataset":"chbmit","task":"Seizure vs Background (4s)","model":"TinyCNN","metric":"AUROC (LOPO)",
           "folds":[{"held_out":s,"auroc":float(a)} for s,a in zip(subjects_unique, fold_aucs)],
           "mean_auroc":float(np.mean(fold_aucs)) if fold_aucs else None,
           "std_auroc":float(np.std(fold_aucs)) if fold_aucs else None}
    with open("results/seizure_cnn.json","w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    print("Saved: results/seizure_cnn.json")

if __name__=="__main__":
    main()
