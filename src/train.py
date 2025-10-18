import os, argparse, joblib, numpy as np, torch
from torch import nn
from .data import make_loaders
from .model import SimpleMLP

def train(epochs=25, lr=1e-3, batch_size=64, outdir="models"):
    outdir = os.path.join(outdir, "staging")
    os.makedirs(outdir, exist_ok=True)

    train_loader, test_loader, scaler = make_loaders(batch_size=batch_size, return_scaler=True)

    device = torch.device("cpu")
    model = SimpleMLP(in_features=4, hidden=16, out_features=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=lr)

    # pre-accumulo test set per valutazione veloce
    Xte, yte = [], []
    for xb, yb in test_loader:
        Xte.append(xb.numpy()); yte.append(yb.numpy())
    Xte = torch.from_numpy(np.concatenate(Xte, axis=0)).to(device)
    yte = torch.from_numpy(np.concatenate(yte, axis=0)).to(device)

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs+1):
        model.train(); run_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
        tr_loss = run_loss / len(train_loader.dataset)

        # eval
        model.eval()
        with torch.no_grad():
            pred = model(Xte).argmax(1)
            acc = (pred == yte).float().mean().item()

        if acc >= best_acc:
            best_acc = acc
            best_state = model.state_dict()

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:02d}] loss_tr={tr_loss:.4f} | acc_val={acc:.3f}")

    # salva best
    torch.save(best_state, os.path.join(outdir, "best.pt"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    print(f"[OK] Salvati: {outdir}\\best.pt, scaler.pkl | best_acc={best_acc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--out",    type=str,  default="models")
    args = ap.parse_args()
    train(args.epochs, args.lr, args.batch, args.out)
