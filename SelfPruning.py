import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import tarfile
import os

# STEP 1: Setup CIFAR-10

def setup_cifar10():
    if os.path.exists('./data/cifar-10-batches-py'):
        print("CIFAR-10 already extracted, skipping.")
        return

    try:
        import urllib.request
        print("Trying to download CIFAR-10 automatically...")
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        os.makedirs('./data', exist_ok=True)
        urllib.request.urlretrieve(url, './data/cifar-10-python.tar.gz')
        print("Download successful!")
        with tarfile.open('./data/cifar-10-python.tar.gz', 'r:gz') as tar:
            tar.extractall('./data')
        print("Extracted successfully.")

    except Exception as e:
        print(f"Auto download failed: {e}")
        print("Run upload_and_extract() manually in the next cell.")


def upload_and_extract():
    from google.colab import files
    print("Select cifar-10-python.tar.gz to upload...")
    uploaded = files.upload()
    fname = list(uploaded.keys())[0]
    os.makedirs('./data', exist_ok=True)
    with tarfile.open(fname, 'r:gz') as tar:
        tar.extractall('./data')
    print("Extracted successfully!")


# Part 1 - Custom Prunable Linear Layer

class PrunableLinear(nn.Module):
    # basically nn.Linear but with an extra gate for each weight
    # gate = sigmoid(gate_score), multiplied elementwise with weights
    # if gate -> 0, that weight is effectively dead / pruned

    def __init__(self, in_f, out_f):
        super(PrunableLinear, self).__init__()

        self.in_f = in_f
        self.out_f = out_f

        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))

        # gate scores same shape as weight
        # init to 0 so sigmoid(0) = 0.5, neutral start
        self.gate_scores = nn.Parameter(torch.zeros(out_f, in_f))

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        w = self.weight * gates
        return F.linear(x, w, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()

    def l1_gates(self):
        return torch.sigmoid(self.gate_scores).sum()


# -----------------------------------------------------------
# Network
# -----------------------------------------------------------

class PruningNet(nn.Module):

    def __init__(self):
        super(PruningNet, self).__init__()

        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def sparsity_loss(self):
        total = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total += m.l1_gates()
        return total

    def sparsity_percent(self, thresh=1e-2):
        all_g = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                all_g.append(m.get_gates().flatten())
        all_g = torch.cat(all_g)
        pct = (all_g < thresh).float().mean().item()
        return pct * 100

    def all_gates(self):
        all_g = []
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                all_g.append(m.get_gates().flatten())
        return torch.cat(all_g).cpu().numpy()


# -----------------------------------------------------------
# Data
# -----------------------------------------------------------

def load_data(bs=256):
    mu  = (0.4914, 0.4822, 0.4465)
    std = (0.247,  0.243,  0.261)

    tr = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])
    te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])

    try:
        train_ds = datasets.CIFAR10('./data', train=True,  download=True,  transform=tr)
        test_ds  = datasets.CIFAR10('./data', train=False, download=True,  transform=te)
        print("CIFAR-10 loaded.")
    except Exception:
        print("Network unavailable, loading from local files...")
        train_ds = datasets.CIFAR10('./data', train=True,  download=False, transform=tr)
        test_ds  = datasets.CIFAR10('./data', train=False, download=False, transform=te)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# -----------------------------------------------------------
# Train / Eval
# -----------------------------------------------------------

def train_epoch(model, loader, optimizer, device, lam):
    model.train()
    total_loss = 0
    correct = 0
    n = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)

        cls_loss  = F.cross_entropy(out, labels)
        spar_loss = model.sparsity_loss()

        loss = cls_loss + lam * spar_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, 100. * correct / n


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)
    return 100. * correct / n


def run_experiment(lam, epochs, device, train_loader, test_loader):
    print(f"\n--- lambda = {lam} ---")

    model = PruningNet().to(device)

    # BUG FIX: We must separate the parameters.
    # 1. Weight decay on gate_scores prevents them from becoming negative, blocking pruning.
    # 2. Gate scores need a slightly higher LR (1e-2) to travel far enough to reach 0 within 30 epochs.
    gate_params = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    base_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    
    opt = optim.Adam([
        {'params': base_params},
        {'params': gate_params, 'weight_decay': 0.0, 'lr': 1e-2}
    ], lr=1e-3, weight_decay=1e-4)
    
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device, lam)
        sched.step()

        if ep % 5 == 0 or ep == epochs:
            te_acc = evaluate(model, test_loader, device)
            sp     = model.sparsity_percent()
            print(f"  ep {ep:3d} | loss {tr_loss:.4f} | train {tr_acc:.1f}% | test {te_acc:.1f}% | sparsity {sp:.1f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sp  = model.sparsity_percent()
    gates     = model.all_gates()

    print(f"  => Test Acc: {final_acc:.2f}%  |  Sparsity: {final_sp:.2f}%")
    return {"lam": lam, "acc": final_acc, "sparsity": final_sp, "gates": gates}


# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------

def plot_gates(results):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    colors = ["#e76f51", "#2a9d8f", "#264653"]

    for ax, r, c in zip(axes, results, colors):
        ax.hist(r["gates"], bins=80, color=c, edgecolor="white", linewidth=0.4)
        ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2, label="thresh=0.01")
        ax.set_title(f"λ={r['lam']}  acc={r['acc']:.1f}%  sparsity={r['sparsity']:.1f}%")
        ax.set_xlabel("Gate value")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend()

    plt.suptitle("Gate value distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved gate_distributions.png")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

if __name__ == "__main__":

    setup_cifar10()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS  = 30
    LAMBDAS = [1e-5, 1e-4, 1e-3]

    train_loader, test_loader = load_data(bs=256)

    results = []
    for lam in LAMBDAS:
        r = run_experiment(lam, EPOCHS, device, train_loader, test_loader)
        results.append(r)

    print("\n\nResults Summary")
    print(f"{'Lambda':<12} {'Test Accuracy':>14} {'Sparsity':>12}")
    print("-" * 42)
    for r in results:
        print(f"{r['lam']:<12} {r['acc']:>13.2f}% {r['sparsity']:>11.2f}%")

    plot_gates(results)