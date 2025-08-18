#!/usr/bin/env python3
import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import eegverse as ev

class TinySleepCNN(nn.Module):
    def __init__(self, C, T):
        super().__init__(); self.c1=nn.Conv1d(C,32,7,padding=3); self.c2=nn.Conv1d(32,64,7,padding=3)
        self.pool=nn.AdaptiveAvgPool1d(1); self.fc=nn.Linear(64,5)
    def forward(self,x):
        x=torch.relu(self.c1(x)); x=torch.relu(self.c2(x)); x=self.pool(x).squeeze(-1); return self.fc(x)

def train_epoch(m, loader, opt, dev):
    m.train(); crit=nn.CrossEntropyLoss(); losses=[]
    for xb,yb in loader:
        xb=xb.to(dev); yb=yb.long().to(dev); opt.zero_grad(); out=m(xb); loss=crit(out,yb)
        loss.backward(); opt.step(); losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

def eval_acc(m, loader, dev):
    m.eval(); corr=tot=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.long().to(dev); pred=m(xb).argmax(1); corr+=(pred==yb).sum().item(); tot+=len(yb)
    return corr/max(1,tot)

def main(root="./eegverse_data"):
    tr = ev.load("sleep_edf", root=root, split="train"); te = ev.load("sleep_edf", root=root, split="test")
    Xtr,ytr,_ = tr.get_data(); Xte,yte,_ = te.get_data()
    if Xtr.size==0 or Xte.size==0: print("No data"); return
    C,T = Xtr.shape[1], Xtr.shape[2]; dev="cuda" if torch.cuda.is_available() else "cpu"
    m=TinySleepCNN(C,T).to(dev); opt=optim.Adam(m.parameters(), lr=1e-3)
    tr_loader=DataLoader(TensorDataset(torch.tensor(Xtr,float), torch.tensor(ytr,int)), batch_size=128, shuffle=True)
    te_loader=DataLoader(TensorDataset(torch.tensor(Xte,float), torch.tensor(yte,int)), batch_size=256, shuffle=False)
    for _ in range(5): train_epoch(m, tr_loader, opt, dev)
    acc=eval_acc(m, te_loader, dev); print(f"Sleep-EDF test acc: {acc:.3f}")
    os.makedirs("results", exist_ok=True)
    with open("results/sleep_edf_cnn.json","w",encoding="utf-8") as f:
        json.dump({"dataset":"sleep_edf","task":"Sleep staging (30s)","model":"TinySleepCNN",
                   "metric":"Acc (train/test split)","score":float(acc)}, f, indent=2)
    print("Saved: results/sleep_edf_cnn.json")

if __name__=="__main__":
    main()
