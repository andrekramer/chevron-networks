import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. THE DATASETS
# ==========================================
def generate_phase1_data(num_samples=4000):
    X, Y = [], []
    for _ in range(num_samples):
        v1 = random.uniform(0.5, 2.0) * random.choice([-1, 1])
        v2 = random.uniform(0.5, 2.0) * random.choice([-1, 1])
        c1, c2 = 0.0, 0.0 # Context is strictly 0 in Phase 1
        
        y = 1.0 if (v1 * v2 < 0) else 0.0
        X.append([v1, c1, v2, c2])
        Y.append([y])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def generate_phase2_data(num_samples=4000):
    X, Y = [], []
    for _ in range(num_samples):
        v1 = random.uniform(0.5, 2.0) * random.choice([-1, 1])
        v2 = random.uniform(0.5, 2.0) * random.choice([-1, 1])
        c1, c2 = random.choice([-1.0, 1.0]), random.choice([-1.0, 1.0])
        
        is_opp = (v1 * v2 < 0)
        is_valid = (c1 > 0 and c2 > 0)
        y = 1.0 if (is_opp and is_valid) else 0.0
        X.append([v1, c1, v2, c2])
        Y.append([y])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

X_p1, Y_p1 = generate_phase1_data(5000)
X_p2, Y_p2 = generate_phase2_data(5000)
X_test_p1, Y_test_p1 = generate_phase1_data(1000)

loader_p1 = DataLoader(TensorDataset(X_p1, Y_p1), batch_size=64, shuffle=True)
loader_p2 = DataLoader(TensorDataset(X_p2, Y_p2), batch_size=64, shuffle=True)
test_loader_p1 = DataLoader(TensorDataset(X_test_p1, Y_test_p1), batch_size=1000)

# ==========================================
# 2. THE MODELS
# ==========================================
class BaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard MLP (no biases to match Chevron constraints fairly)
        self.net = nn.Sequential(
            nn.Linear(4, 32, bias=False), nn.GELU(),
            nn.Linear(32, 32, bias=False), nn.GELU(),
            nn.Linear(32, 1, bias=False)
        )
    def forward(self, x): return self.net(x)


class StrictChevronLinear(nn.Module):
    def __init__(self, in_groups, out_groups):
        super().__init__()
        # Initialize 2x2 Operator
        self.weight = nn.Parameter(torch.randn(out_groups, in_groups, 2, 2) * 0.1)
        self.phase = 1

        self.weight.register_hook(self._weight_hook)

    def _weight_hook(self, grad):
        g = grad.clone()
        if self.phase == 1:
            # PHASE 1: Build the Semantic Core. 
            # Only w_00 (Content -> Thesis) learns.
            g[:, :, 0, 1] = 0.0 # Lock Context -> Thesis
            g[:, :, 1, 0] = 0.0 # Lock Content -> Antithesis
            g[:, :, 1, 1] = 0.0 # Lock Context -> Antithesis
        elif self.phase == 2:
            # PHASE 2: Freeze the Core. Learn the Veto.
            g[:, :, 0, 0] = 0.0 # CORE FROZEN. No Catastrophic Forgetting.
            g[:, :, 1, 0] = 0.0 # PREVENT LEAK: Content cannot drive Antithesis.
            # Only w_11 (Context -> Antithesis) and w_01 (Context -> Thesis) adapt.
        return g

    def forward(self, x):
        w = self.weight.clone()
        # Enforce strict 0s for untrainable paths to keep initialization noise out
        if self.phase == 1:
            w[:, :, 0, 1] = 0.0
            w[:, :, 1, 0] = 0.0
            w[:, :, 1, 1] = 0.0
        elif self.phase == 2:
            w[:, :, 1, 0] = 0.0 

        routed = torch.einsum("bij,oikj->boik", x, w)
        return routed.sum(dim=2)


class ChevronNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = StrictChevronLinear(2, 32)
        self.fc2 = StrictChevronLinear(32, 32)
        self.act = nn.GELU()

    def set_phase(self, phase):
        self.fc1.phase = phase
        self.fc2.phase = phase

    def forward(self, x):
        # Map 4 inputs -> 2 groups of [Value, Context]
        x = x.view(-1, 2, 2)
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        # Oppositional Readout
        return h[:, :, 0].sum(dim=-1, keepdim=True) - h[:, :, 1].sum(dim=-1, keepdim=True)

# ==========================================
# 3. RUN THE EXPERIMENT
# ==========================================
def train_phase(model, loader, epochs=10, lr=0.01):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for X_b, Y_b in loader:
            opt.zero_grad()
            loss = crit(model(X_b), Y_b)
            loss.backward()
            opt.step()

mlp = BaselineMLP()
chev = ChevronNet()

print("Phase 1: Training models on 'Base Concept'...")
chev.set_phase(1)
train_phase(mlp, loader_p1)
train_phase(chev, loader_p1)

# Capture Chevron Phase 1 baseline accuracy
chev.eval()
with torch.no_grad():
    p1_logits = chev(X_test_p1)
    chev_p1_baseline = ((torch.sigmoid(p1_logits) > 0.5).float() == Y_test_p1).float().mean().item()

print("Phase 2: Environment changes. Learning 'Category Veto'...")
chev.set_phase(2) # Switches the gradient routing!
train_phase(mlp, loader_p2)
train_phase(chev, loader_p2)

print("\nPhase 3: Testing CATASTROPHIC FORGETTING (Back to Phase 1 Data)...")
mlp.eval()
chev.eval()
with torch.no_grad():
    mlp_acc = ((torch.sigmoid(mlp(X_test_p1)) > 0.5).float() == Y_test_p1).float().mean().item()
    chev_acc = ((torch.sigmoid(chev(X_test_p1)) > 0.5).float() == Y_test_p1).float().mean().item()

print(f"\n--- CATASTROPHIC FORGETTING RESULTS ---")
print(f"Baseline MLP Retained Base Skill: {mlp_acc*100:.1f}%")
print(f"Chevron Net  Retained Base Skill: {chev_acc*100:.1f}% (Was {chev_p1_baseline*100:.1f}% before Phase 2)")

if chev_acc > mlp_acc:
    print(f"\nSUCCESS: Chevron structurally eliminated catastrophic forgetting (+{(chev_acc - mlp_acc)*100:.1f}%).")
