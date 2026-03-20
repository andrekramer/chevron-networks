import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. THE SEQUENTIAL DATASET (The Veto Task)
# ==========================================
SEQ_LEN = 40
VOCAB_SIZE = 4
# 0: Neutral, 1: Up (+1), 2: Down (-1), 3: VETO (Reset memory to 0)

def generate_sequence_data(num_samples=5000):
    X = torch.zeros((num_samples, SEQ_LEN), dtype=torch.long)
    Y = torch.zeros((num_samples,), dtype=torch.long)
    
    for i in range(num_samples):
        seq = [0] * SEQ_LEN
        # Insert 10-20 pieces of random evidence
        for _ in range(random.randint(10, 20)):
            seq[random.randint(0, SEQ_LEN-1)] = random.choice([1, 2])
            
        # Insert 1-3 Veto tokens
        for _ in range(random.randint(1, 3)):
            seq[random.randint(5, SEQ_LEN-5)] = 3
            
        X[i] = torch.tensor(seq)
        
        # Calculate True Label: The sum of evidence AFTER the LAST Veto
        current_sum = 0
        for token in seq:
            if token == 1: current_sum += 1
            elif token == 2: current_sum -= 1
            elif token == 3: current_sum = 0 # THE RESET
            
        if current_sum == 0: Y[i] = 0
        elif current_sum > 0: Y[i] = 1
        elif current_sum < 0: Y[i] = 2
            
    return X, Y

X_train, Y_train = generate_sequence_data(8000)
X_test, Y_test = generate_sequence_data(2000)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=2000, shuffle=False)

# ==========================================
# 2. THE THREE CHEVRON REGIMES
# ==========================================

class FreeChevronRNN(nn.Module):
    def __init__(self, d_emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.w_in = nn.Linear(d_emb, hidden * 2)
        # Unconstrained 2x2 mixing matrix per hidden dimension
        self.w_rec = nn.Parameter(torch.randn(hidden, 2, 2) * 0.1)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        b, seq_len = x.size()
        h = torch.zeros(b, self.w_rec.size(0), 2, device=x.device)
        
        for t in range(seq_len):
            i_t = self.w_in(self.emb(x[:, t])).view(b, -1, 2)
            # Unconstrained Routing
            h_routed = torch.einsum("bhi,hij->bhj", h, self.w_rec)
            # Tanh needed to prevent exploding gradients in Free RNNs
            h = torch.tanh(h_routed + i_t)
            
        # Collapse state for readout: Thesis - Antithesis
        belief = h[:, :, 0] - h[:, :, 1]
        return self.head(belief)

class ComplexChevronRNN(nn.Module):
    def __init__(self, d_emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.w_in = nn.Linear(d_emb, hidden * 2)
        
        # Generates the rotation angle theta based on input
        self.theta_proj = nn.Linear(d_emb, hidden)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        b, seq_len = x.size()
        h = torch.zeros(b, self.head.in_features, 2, device=x.device)
        
        for t in range(seq_len):
            emb_t = self.emb(x[:, t])
            i_t = self.w_in(emb_t).view(b, -1, 2)
            
            # COMPLEX REGIME: State updates purely via Rotation
            theta = self.theta_proj(emb_t)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            
            h_A_new = h[:, :, 0] * cos_t - h[:, :, 1] * sin_t + i_t[:, :, 0]
            h_N_new = h[:, :, 0] * sin_t + h[:, :, 1] * cos_t + i_t[:, :, 1]
            
            h = torch.stack([h_A_new, h_N_new], dim=-1)
            
        belief = h[:, :, 0] - h[:, :, 1]
        return self.head(belief)

class StructuredChevronRNN(nn.Module):
    def __init__(self, d_emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.w_in = nn.Linear(d_emb, hidden * 2)
        self.w_rec = nn.Parameter(torch.randn(hidden, 2, 2) * 0.1)
        
        # STRUCTURED REGIME: "World" channel acts as Epistemic Gate on the "Self" Memory
        self.gate_w = nn.Parameter(torch.randn(hidden) * 0.1)
        self.gate_b = nn.Parameter(torch.ones(hidden) * 2.0) # Start open
        self.head = nn.Linear(hidden, 3)

    def forward(self, x):
        b, seq_len = x.size()
        h = torch.zeros(b, self.w_rec.size(0), 2, device=x.device)
        
        for t in range(seq_len):
            i_t = self.w_in(self.emb(x[:, t])).view(b, -1, 2)
            h_routed = torch.einsum("bhi,hij->bhj", h, self.w_rec)
            
            # The "World/Control" channel (index 1) controls the Gate
            gate = torch.sigmoid(self.gate_b - i_t[:, :, 1] * self.gate_w)
            
            # The Gate physically squashes the "Self" memory if a Reset is detected
            h_A_new = gate * h_routed[:, :, 0] + i_t[:, :, 0]
            h_N_new = torch.tanh(h_routed[:, :, 1] + i_t[:, :, 1])
            
            h = torch.stack([h_A_new, h_N_new], dim=-1)
            
        belief = h[:, :, 0] - h[:, :, 1]
        return self.head(belief)

# ==========================================
# 3. TRAINING AND EVALUATION
# ==========================================
def train_and_eval(model_class, name):
    print(f"Training {name}...")
    model = model_class()
    opt = torch.optim.AdamW(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(15):
        for X_b, Y_b in train_loader:
            opt.zero_grad()
            loss = crit(model(X_b), Y_b)
            loss.backward()
            opt.step()
            
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_b, Y_b in test_loader:
            preds = model(X_b).argmax(dim=-1)
            correct += (preds == Y_b).sum().item()
            
    acc = (correct / len(Y_test)) * 100
    print(f"{name} Accuracy: {acc:.1f}%\n")
    return acc

free_acc = train_and_eval(FreeChevronRNN, "Free Chevron")
comp_acc = train_and_eval(ComplexChevronRNN, "Complex Chevron (Mamba-like)")
stru_acc = train_and_eval(StructuredChevronRNN, "Structured Chevron (Self/World Gate)")

# ==========================================
# 4. PLOT RESULTS
# ==========================================
labels =['Free Chevron\n(Unconstrained)', 'Complex Chevron\n(Pure Rotation)', 'Structured Chevron\n(Self/World Gate)']
rates = [free_acc, comp_acc, stru_acc]
colors =['lightcoral', 'mediumpurple', 'steelblue']

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, rates, color=colors, width=0.5)
ax.set_ylabel('Accuracy on Long-Context Veto Task (%)')
ax.set_title('The Three Regimes: Memory vs. Control')
plt.ylim(0, 105)

for i, v in enumerate(rates):
    ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

plt.tight_layout()
plt.show()