# ==========================================
# INSTRUMENTED CHEVRON DUAL-ATTENTION DEMO
# WITH PER-HIDDEN-UNIT CONTEXT SHIFT ANALYSIS
# ==========================================

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# 0. Setup
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# 1. Simple dual-cue dataset
# -----------------------------
# Input:
#   [pos_cue, neg_cue, context, distractor]
#
# Rule:
#   if context == 0: y = pos_cue
#   if context == 1: y = 1 - neg_cue

def make_dataset(n=4000, seed=0):
    random.seed(seed)
    X, Y = [], []
    for _ in range(n):
        pos_cue = random.randint(0, 1)
        neg_cue = random.randint(0, 1)
        context = random.randint(0, 1)
        distractor = random.randint(0, 1)

        if context == 0:
            y = pos_cue
        else:
            y = 1 - neg_cue

        X.append([pos_cue, neg_cue, context, distractor])
        Y.append(y)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.long)
    )

X_train, Y_train = make_dataset(n=4000, seed=1)
X_test, Y_test = make_dataset(n=1000, seed=2)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256, shuffle=False)

# -----------------------------
# 2. Instrumented Chevron model
# -----------------------------
class InstrumentedChevronDualAttention(nn.Module):
    """
    Two channels:
      channel 0 = positive evidence stream
      channel 1 = negative evidence stream

    Context creates dual attention weights alpha[:,:,0] and alpha[:,:,1].
    A per-hidden-unit 2x2 chevron mixes the channels.
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden

        self.pos_proj = nn.Linear(4, hidden)
        self.neg_proj = nn.Linear(4, hidden)

        # context -> dual attention logits
        self.ctx_gate = nn.Linear(1, hidden * 2)

        # per-hidden-unit 2x2 chevron transform
        self.chevron = nn.Parameter(torch.randn(hidden, 2, 2) * 0.1)

        self.out = nn.Linear(hidden * 2, 2)

    def forward(self, x, return_internal=False):
        context = x[:, 2:3]  # [B,1]

        pos = self.pos_proj(x)  # [B,H]
        neg = self.neg_proj(x)  # [B,H]

        state = torch.stack([pos, neg], dim=-1)  # [B,H,2]

        gate_logits = self.ctx_gate(context).view(x.size(0), self.hidden, 2)  # [B,H,2]
        alpha = torch.softmax(gate_logits, dim=-1)  # [B,H,2]

        attended = state * alpha
        mixed = torch.einsum("bhj,hji->bhi", attended, self.chevron)  # [B,H,2]

        h = mixed.reshape(x.size(0), -1)
        logits = self.out(h)

        if return_internal:
            return logits, {
                "alpha": alpha,         # [B,H,2]
                "gate_logits": gate_logits,  # [B,H,2]
                "state": state,         # [B,H,2]
                "attended": attended,   # [B,H,2]
                "mixed": mixed,         # [B,H,2]
            }
        return logits

# -----------------------------
# 3. Train / eval helpers
# -----------------------------
def train_model(model, loader, epochs=20, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, total = 0.0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            total += yb.size(0)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | loss={total_loss/total:.4f}")

    return model

def eval_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# -----------------------------
# 4. Train model
# -----------------------------
print("\nTraining Instrumented Chevron Dual-Attention...")
model = train_model(InstrumentedChevronDualAttention(hidden=32), train_loader, epochs=20, lr=1e-3)

acc = eval_acc(model, test_loader)
print(f"\nTest accuracy: {acc:.4f}")

# -----------------------------
# 5. Inspect internals
# -----------------------------
model.eval()
with torch.no_grad():
    xb = X_test.to(device)
    logits, internal = model(xb, return_internal=True)

    alpha = internal["alpha"].cpu()             # [N,H,2]
    gate_logits = internal["gate_logits"].cpu() # [N,H,2]
    contexts = X_test[:, 2]                     # [N]

    c0 = (contexts == 0)
    c1 = (contexts == 1)

    # Global averages across all examples + hidden units
    alpha_c0_global = alpha[c0].mean(dim=(0, 1))  # [2]
    alpha_c1_global = alpha[c1].mean(dim=(0, 1))  # [2]

    # Per-hidden-unit averages (average over examples only)
    # shape: [H,2]
    alpha_c0_per_unit = alpha[c0].mean(dim=0)
    alpha_c1_per_unit = alpha[c1].mean(dim=0)

    # Positive-channel attention shift per hidden unit
    # >0 means unit attends more to positive channel in context=0 than context=1
    pos_shift = alpha_c0_per_unit[:, 0] - alpha_c1_per_unit[:, 0]  # [H]

    # Negative-channel attention shift per hidden unit
    neg_shift = alpha_c0_per_unit[:, 1] - alpha_c1_per_unit[:, 1]  # [H]

    # Per-hidden-unit gate logit shift (before softmax), sometimes more revealing
    gate_c0_per_unit = gate_logits[c0].mean(dim=0)  # [H,2]
    gate_c1_per_unit = gate_logits[c1].mean(dim=0)  # [H,2]
    gate_pos_shift = gate_c0_per_unit[:, 0] - gate_c1_per_unit[:, 0]
    gate_neg_shift = gate_c0_per_unit[:, 1] - gate_c1_per_unit[:, 1]

    # Average chevron matrix across hidden units
    avg_chevron = model.chevron.detach().cpu().mean(dim=0)  # [2,2]

print("\n--- GLOBAL AVERAGE DUAL ATTENTION ---")
print(f"Context=0 -> alpha_pos={alpha_c0_global[0].item():.4f}, alpha_neg={alpha_c0_global[1].item():.4f}")
print(f"Context=1 -> alpha_pos={alpha_c1_global[0].item():.4f}, alpha_neg={alpha_c1_global[1].item():.4f}")

print("\n--- PER-UNIT POSITIVE ATTENTION SHIFT SUMMARY ---")
print(f"mean shift: {pos_shift.mean().item():.4f}")
print(f"max shift : {pos_shift.max().item():.4f}")
print(f"min shift : {pos_shift.min().item():.4f}")

print("\n--- PER-UNIT GATE-LOGIT SHIFT SUMMARY (positive channel) ---")
print(f"mean shift: {gate_pos_shift.mean().item():.4f}")
print(f"max shift : {gate_pos_shift.max().item():.4f}")
print(f"min shift : {gate_pos_shift.min().item():.4f}")

print("\n--- AVERAGE 2x2 CHEVRON MATRIX ---")
print(avg_chevron)

# -----------------------------
# 6. Plot global average attention
# -----------------------------
labels = ["Positive channel", "Negative channel"]
c0_vals = [alpha_c0_global[0].item(), alpha_c0_global[1].item()]
c1_vals = [alpha_c1_global[0].item(), alpha_c1_global[1].item()]

x = torch.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, c0_vals, width, label="Context=0", color="steelblue")
plt.bar(x + width/2, c1_vals, width, label="Context=1", color="darkorange")
plt.xticks(x, labels)
plt.ylabel("Average attention weight")
plt.title("Global Average Dual Attention by Context")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Plot per-unit positive attention shift
# -----------------------------
# Sort units by shift for easier viewing
sorted_shift, sort_idx = torch.sort(pos_shift)

plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_shift)), sorted_shift.numpy(), color="mediumpurple")
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Hidden units (sorted)")
plt.ylabel("alpha_pos(context=0) - alpha_pos(context=1)")
plt.title("Per-Hidden-Unit Positive-Attention Shift")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Histogram of per-unit positive attention shift
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(pos_shift.numpy(), bins=12, color="teal", edgecolor="black", alpha=0.8)
plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Positive-attention shift")
plt.ylabel("Number of hidden units")
plt.title("Distribution of Per-Unit Positive-Attention Shifts")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Plot per-unit gate-logit shift (positive channel)
# -----------------------------
sorted_gate_shift, _ = torch.sort(gate_pos_shift)

plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_gate_shift)), sorted_gate_shift.numpy(), color="salmon")
plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Hidden units (sorted)")
plt.ylabel("gate_logit_pos(context=0) - gate_logit_pos(context=1)")
plt.title("Per-Hidden-Unit Positive Gate-Logit Shift")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Plot average chevron matrix
# -----------------------------
plt.figure(figsize=(5, 4))
plt.imshow(avg_chevron.numpy(), cmap="coolwarm", aspect="equal")
plt.colorbar(label="Weight")
plt.xticks([0, 1], ["to +", "to -"])
plt.yticks([0, 1], ["from +", "from -"])
plt.title("Average Learned 2x2 Chevron Matrix")
for i in range(2):
    for j in range(2):
        plt.text(
            j, i, f"{avg_chevron[i, j].item():.2f}",
            ha="center", va="center", color="black", fontweight="bold"
        )
plt.tight_layout()
plt.show()
