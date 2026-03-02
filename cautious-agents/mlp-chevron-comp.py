import random
import statistics as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [1, 2, 3, 4, 5]
THRESHOLDS = [i / 20 for i in range(1, 20)]  # 0.05 ... 0.95

NUM_ENTITIES = 20
NUM_RELS = 10
NUM_ANSWERS = 20
IDK_TOKEN = 20

ENTITY_OFFSET = 0
REL_OFFSET = NUM_ENTITIES
OVERRIDE_OFFSET = NUM_ENTITIES + NUM_RELS
VOCAB_SIZE = NUM_ENTITIES + NUM_RELS + 2

SAFE_RELS = list(range(0, 6))
TRAP_RELS = list(range(6, 10))

print("Using device:", DEVICE)

# ==========================================
# DATASET
# ==========================================
def build_dataset(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    fact_table = {}
    for e in range(NUM_ENTITIES):
        for r in range(NUM_RELS):
            fact_table[(e, r)] = (3 * e + 5 * r) % NUM_ANSWERS

    # Rule:
    # - safe rels: always answer
    # - trap rels: answer iff (override XOR entity_parity) == 1, else IDK
    data = []
    for e in range(NUM_ENTITIES):
        parity = e % 2
        for r in range(NUM_RELS):
            for override in [0, 1]:
                if r in SAFE_RELS:
                    should_idk = 0
                else:
                    should_answer = (override ^ parity) == 1
                    should_idk = 0 if should_answer else 1
                ans = fact_table[(e, r)]
                data.append((e, r, override, ans, should_idk))

    random.shuffle(data)
    n = len(data)
    train_data = data[: int(0.7 * n)]
    test_data = data[int(0.7 * n):]
    return train_data, test_data

def make_loader(data, batch_size=64, shuffle=True):
    X = torch.tensor(
        [[e, r + REL_OFFSET, o + OVERRIDE_OFFSET] for e, r, o, ans, should_idk in data],
        dtype=torch.long
    )
    ans_y = torch.tensor([ans for e, r, o, ans, should_idk in data], dtype=torch.long)
    idk_y = torch.tensor([should_idk for e, r, o, ans, should_idk in data], dtype=torch.float32)
    return DataLoader(TensorDataset(X, ans_y, idk_y), batch_size=batch_size, shuffle=shuffle)

# ==========================================
# MODELS
# ==========================================
class BaselineMLP(nn.Module):
    def __init__(self, d_emb=32, hidden=96):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.net = nn.Sequential(
            nn.Linear(3 * d_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.ans_head = nn.Linear(hidden, NUM_ANSWERS)
        self.idk_head = nn.Linear(hidden, 1)

    def forward(self, e_tok, r_tok, o_tok):
        x = torch.cat([self.emb(e_tok), self.emb(r_tok), self.emb(o_tok)], dim=-1)
        h = self.net(x)
        return self.ans_head(h), self.idk_head(h).squeeze(-1)

class ChevronLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 2, 2) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features, 2))

    def forward(self, x):
        routed = torch.einsum("bij,oikj->boik", x, self.weight)  # [B,out,in,2]
        return routed.sum(dim=2) + self.bias.unsqueeze(0)

class ChevronNet(nn.Module):
    def __init__(self, d_emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.input_proj = nn.Linear(3 * d_emb, hidden)
        self.override_proj = nn.Linear(d_emb, hidden)
        self.fc1 = ChevronLinear(hidden, hidden)
        self.fc2 = ChevronLinear(hidden, hidden)
        self.act = nn.Tanh()
        self.ans_head = nn.Linear(hidden, NUM_ANSWERS)
        self.idk_head = nn.Linear(hidden, 1)
        self.idk_override_scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, e_tok, r_tok, o_tok):
        e_emb = self.emb(e_tok)
        r_emb = self.emb(r_tok)
        o_emb = self.emb(o_tok)

        base = self.input_proj(torch.cat([e_emb, r_emb, o_emb], dim=-1))
        override_signal = self.override_proj(o_emb)

        x = torch.zeros(base.size(0), base.size(1), 2, device=base.device)
        x[:, :, 0] = base
        x[:, :, 1] = override_signal

        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))

        belief = h[:, :, 0] - 0.2 * h[:, :, 1]
        control = h[:, :, 1]

        ans_logits = self.ans_head(belief)
        idk_logit = self.idk_head(control).squeeze(-1) - self.idk_override_scale * control.mean(dim=-1)
        return ans_logits, idk_logit

class ChevronControlHeadNet(nn.Module):
    def __init__(self, d_emb=32, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_emb)
        self.content_proj = nn.Linear(2 * d_emb, hidden)
        self.control_proj = nn.Linear(d_emb, hidden)
        self.fc1 = ChevronLinear(hidden, hidden)
        self.fc2 = ChevronLinear(hidden, hidden)
        self.act = nn.Tanh()
        self.ans_head = nn.Linear(hidden, NUM_ANSWERS)
        self.idk_head = nn.Linear(hidden * 2, 1)

    def forward(self, e_tok, r_tok, o_tok):
        e_emb = self.emb(e_tok)
        r_emb = self.emb(r_tok)
        o_emb = self.emb(o_tok)

        content = self.content_proj(torch.cat([e_emb, r_emb], dim=-1))
        control = self.control_proj(o_emb)

        x = torch.zeros(content.size(0), content.size(1), 2, device=content.device)
        x[:, :, 0] = content
        x[:, :, 1] = control

        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))

        h_content = h[:, :, 0]
        h_control = h[:, :, 1]

        ans_logits = self.ans_head(h_content)
        idk_features = torch.cat([h_control, torch.abs(h_content)], dim=-1)
        idk_logit = self.idk_head(idk_features).squeeze(-1)
        return ans_logits, idk_logit

# ==========================================
# TRAINING
# ==========================================
def train_model(model, train_loader, epochs=60, lr=0.003,
                idk_loss_weight=1.0, special_answer_boost=2.0):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        model.train()
        for X_b, ans_y, idk_y in train_loader:
            X_b = X_b.to(DEVICE)
            ans_y = ans_y.to(DEVICE)
            idk_y = idk_y.to(DEVICE)

            ans_logits, idk_logit = model(X_b[:, 0], X_b[:, 1], X_b[:, 2])

            answer_mask = (idk_y < 0.5)
            if answer_mask.any():
                ans_loss_per = F.cross_entropy(
                    ans_logits[answer_mask],
                    ans_y[answer_mask],
                    reduction="none"
                )

                r_raw = X_b[:, 1] - REL_OFFSET
                boost_mask = torch.isin(r_raw, torch.tensor(TRAP_RELS, device=DEVICE)) & answer_mask

                weights = torch.ones_like(ans_loss_per)
                weights[boost_mask[answer_mask]] = special_answer_boost
                ans_loss = (ans_loss_per * weights).mean()
            else:
                ans_loss = torch.tensor(0.0, device=DEVICE)

            idk_loss = F.binary_cross_entropy_with_logits(idk_logit, idk_y)
            loss = ans_loss + idk_loss_weight * idk_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

    return model

# ==========================================
# THRESHOLD SWEEP EVAL
# ==========================================
def collect_logits(model, loader):
    model.eval()
    rows = []
    with torch.no_grad():
        for X_b, ans_y, idk_y in loader:
            X_b = X_b.to(DEVICE)
            ans_y = ans_y.to(DEVICE)
            idk_y = idk_y.to(DEVICE)

            ans_logits, idk_logit = model(X_b[:, 0], X_b[:, 1], X_b[:, 2])

            rows.append({
                "ans_pred": ans_logits.argmax(dim=-1).cpu(),
                "idk_prob": torch.sigmoid(idk_logit).cpu(),
                "ans_y": ans_y.cpu(),
                "idk_y": idk_y.cpu(),
                "r_raw": (X_b[:, 1] - REL_OFFSET).cpu(),
            })
    return rows

def evaluate_at_threshold(collected, threshold):
    total = correct = 0
    safe_total = safe_correct = 0
    abstain_total = abstain_correct = abstain_hallucinations = 0
    trap_answer_total = trap_answer_correct = 0

    for batch in collected:
        ans_pred = batch["ans_pred"]
        idk_prob = batch["idk_prob"]
        ans_y = batch["ans_y"]
        idk_y = batch["idk_y"]
        r_raw = batch["r_raw"]

        final_pred = ans_pred.clone()
        final_pred[idk_prob > threshold] = IDK_TOKEN

        y_final = ans_y.clone()
        y_final[idk_y > 0.5] = IDK_TOKEN

        correct += (final_pred == y_final).sum().item()
        total += y_final.size(0)

        safe_mask = torch.isin(r_raw, torch.tensor(SAFE_RELS))
        if safe_mask.any():
            safe_total += safe_mask.sum().item()
            safe_correct += (final_pred[safe_mask] == y_final[safe_mask]).sum().item()

        abstain_mask = torch.isin(r_raw, torch.tensor(TRAP_RELS)) & (idk_y > 0.5)
        if abstain_mask.any():
            abstain_total += abstain_mask.sum().item()
            abstain_correct += (final_pred[abstain_mask] == IDK_TOKEN).sum().item()
            abstain_hallucinations += (final_pred[abstain_mask] != IDK_TOKEN).sum().item()

        trap_answer_mask = torch.isin(r_raw, torch.tensor(TRAP_RELS)) & (idk_y < 0.5)
        if trap_answer_mask.any():
            trap_answer_total += trap_answer_mask.sum().item()
            trap_answer_correct += (final_pred[trap_answer_mask] == y_final[trap_answer_mask]).sum().item()

    return {
        "overall_acc": correct / max(total, 1),
        "safe_acc": safe_correct / max(safe_total, 1),
        "trap_abstain_acc": abstain_correct / max(abstain_total, 1),
        "trap_hallucination_rate": abstain_hallucinations / max(abstain_total, 1),
        "trap_reopen_acc": trap_answer_correct / max(trap_answer_total, 1),
    }

def sweep_thresholds(model, loader, thresholds):
    collected = collect_logits(model, loader)
    return [
        {**evaluate_at_threshold(collected, t), "threshold": t}
        for t in thresholds
    ]

def best_by_metric(results, key, maximize=True):
    return max(results, key=lambda x: x[key]) if maximize else min(results, key=lambda x: x[key])

# ==========================================
# RUN ONE SEED
# ==========================================
def run_one_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    train_data, test_data = build_dataset(seed)
    train_loader = make_loader(train_data, batch_size=64, shuffle=True)
    test_loader = make_loader(test_data, batch_size=64, shuffle=False)

    mlp = train_model(BaselineMLP(), train_loader, special_answer_boost=2.0)
    chev = train_model(ChevronNet(), train_loader, special_answer_boost=2.5)
    ctrl = train_model(ChevronControlHeadNet(), train_loader, special_answer_boost=2.5)

    mlp_sweep = sweep_thresholds(mlp, test_loader, THRESHOLDS)
    chev_sweep = sweep_thresholds(chev, test_loader, THRESHOLDS)
    ctrl_sweep = sweep_thresholds(ctrl, test_loader, THRESHOLDS)

    result = {
        "MLP": {
            "sweep": mlp_sweep,
            "best_overall": best_by_metric(mlp_sweep, "overall_acc", True),
            "best_reopen": best_by_metric(mlp_sweep, "trap_reopen_acc", True),
            "best_abstain": best_by_metric(mlp_sweep, "trap_abstain_acc", True),
            "best_hall": best_by_metric(mlp_sweep, "trap_hallucination_rate", False),
        },
        "Chevron_NoGate": {
            "sweep": chev_sweep,
            "best_overall": best_by_metric(chev_sweep, "overall_acc", True),
            "best_reopen": best_by_metric(chev_sweep, "trap_reopen_acc", True),
            "best_abstain": best_by_metric(chev_sweep, "trap_abstain_acc", True),
            "best_hall": best_by_metric(chev_sweep, "trap_hallucination_rate", False),
        },
        "Chevron_ControlHead": {
            "sweep": ctrl_sweep,
            "best_overall": best_by_metric(ctrl_sweep, "overall_acc", True),
            "best_reopen": best_by_metric(ctrl_sweep, "trap_reopen_acc", True),
            "best_abstain": best_by_metric(ctrl_sweep, "trap_abstain_acc", True),
            "best_hall": best_by_metric(ctrl_sweep, "trap_hallucination_rate", False),
        },
    }
    return result

# ==========================================
# MULTI-SEED LOOP
# ==========================================
all_runs = {}
for seed in SEEDS:
    print(f"\n=== Running seed {seed} ===")
    all_runs[seed] = run_one_seed(seed)
    for model_name, res in all_runs[seed].items():
        print(f"{model_name}:")
        print(f"  best overall_acc      {res['best_overall']['overall_acc']:.4f} @ thr={res['best_overall']['threshold']:.2f}")
        print(f"  best trap_reopen_acc  {res['best_reopen']['trap_reopen_acc']:.4f} @ thr={res['best_reopen']['threshold']:.2f}")
        print(f"  best trap_abstain_acc {res['best_abstain']['trap_abstain_acc']:.4f} @ thr={res['best_abstain']['threshold']:.2f}")
        print(f"  lowest hallucination  {res['best_hall']['trap_hallucination_rate']:.4f} @ thr={res['best_hall']['threshold']:.2f}")

# ==========================================
# AGGREGATE SUMMARY
# ==========================================
def summarize_model(model_name):
    out = {
        "best_overall_acc": [],
        "best_overall_thr": [],
        "best_reopen_acc": [],
        "best_reopen_thr": [],
        "best_abstain_acc": [],
        "best_abstain_thr": [],
        "lowest_hall_rate": [],
        "lowest_hall_thr": [],
    }
    for seed in SEEDS:
        res = all_runs[seed][model_name]
        out["best_overall_acc"].append(res["best_overall"]["overall_acc"])
        out["best_overall_thr"].append(res["best_overall"]["threshold"])
        out["best_reopen_acc"].append(res["best_reopen"]["trap_reopen_acc"])
        out["best_reopen_thr"].append(res["best_reopen"]["threshold"])
        out["best_abstain_acc"].append(res["best_abstain"]["trap_abstain_acc"])
        out["best_abstain_thr"].append(res["best_abstain"]["threshold"])
        out["lowest_hall_rate"].append(res["best_hall"]["trap_hallucination_rate"])
        out["lowest_hall_thr"].append(res["best_hall"]["threshold"])
    return out

def mean_std(xs):
    return stats.mean(xs), (stats.pstdev(xs) if len(xs) > 1 else 0.0)

summaries = {}
for model_name in ["MLP", "Chevron_NoGate", "Chevron_ControlHead"]:
    s = summarize_model(model_name)
    summaries[model_name] = s

    bo_m, bo_s = mean_std(s["best_overall_acc"])
    bt_m, bt_s = mean_std(s["best_overall_thr"])
    br_m, br_s = mean_std(s["best_reopen_acc"])
    brt_m, brt_s = mean_std(s["best_reopen_thr"])
    ba_m, ba_s = mean_std(s["best_abstain_acc"])
    bat_m, bat_s = mean_std(s["best_abstain_thr"])
    bh_m, bh_s = mean_std(s["lowest_hall_rate"])
    bht_m, bht_s = mean_std(s["lowest_hall_thr"])

    print(f"\n=== {model_name} : mean ± std over {len(SEEDS)} seeds ===")
    print(f"Best overall_acc      : {bo_m:.4f} ± {bo_s:.4f}   @ thr {bt_m:.2f} ± {bt_s:.2f}")
    print(f"Best trap_reopen_acc  : {br_m:.4f} ± {br_s:.4f}   @ thr {brt_m:.2f} ± {brt_s:.2f}")
    print(f"Best trap_abstain_acc : {ba_m:.4f} ± {ba_s:.4f}   @ thr {bat_m:.2f} ± {bat_s:.2f}")
    print(f"Lowest hallucination  : {bh_m:.4f} ± {bh_s:.4f}   @ thr {bht_m:.2f} ± {bht_s:.2f}")

# ==========================================
# GRAPH 1: mean ± std bars
# ==========================================
model_labels = ["MLP", "Chevron_NoGate", "Chevron_ControlHead"]
pretty_labels = ["MLP", "Chevron\nNo Gate", "Chevron\nCtrl Head"]
colors = ["lightcoral", "goldenrod", "steelblue"]

metric_specs = [
    ("best_overall_acc", "Best Overall Accuracy", False),
    ("best_reopen_acc", "Best Trap Reopen Accuracy", False),
    ("best_abstain_acc", "Best Trap Abstain Accuracy", False),
    ("lowest_hall_rate", "Lowest Hallucination Rate", True),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (metric_key, title, lower_better) in zip(axes, metric_specs):
    means, errs = [], []
    for model_name in model_labels:
        m, s = mean_std(summaries[model_name][metric_key])
        means.append(100 * m)
        errs.append(100 * s)

    bars = ax.bar(pretty_labels, means, yerr=errs, capsize=5, color=colors)
    ax.set_title(title)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, v + 2, f"{v:.1f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("benchmark_summary_bars.png", dpi=160)
plt.show()

# ==========================================
# GRAPH 2: threshold curves for first seed
# ==========================================
ref_seed = SEEDS[0]
ref = all_runs[ref_seed]

def plot_curve(metric_key, title):
    plt.figure(figsize=(6, 4))
    for model_name, label in [
        ("MLP", "MLP"),
        ("Chevron_NoGate", "Chevron No Gate"),
        ("Chevron_ControlHead", "Chevron Ctrl Head"),
    ]:
        sweep = ref[model_name]["sweep"]
        xs = [row["threshold"] for row in sweep]
        ys = [100 * row[metric_key] for row in sweep]
        plt.plot(xs, ys, marker="o", label=label)

    plt.title(f"{title} (seed {ref_seed})")
    plt.xlabel("IDK threshold")
    plt.ylabel("%")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

plot_curve("overall_acc", "Overall Accuracy")
plt.savefig("benchmark_curve_overall.png", dpi=160)
plt.show()

plot_curve("trap_reopen_acc", "Trap Reopen Accuracy")
plt.savefig("benchmark_curve_reopen.png", dpi=160)
plt.show()

plot_curve("trap_abstain_acc", "Trap Abstain Accuracy")
plt.savefig("benchmark_curve_abstain.png", dpi=160)
plt.show()

plot_curve("trap_hallucination_rate", "Hallucination Rate")
plt.savefig("benchmark_curve_hallucination.png", dpi=160)
plt.show()
