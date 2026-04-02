from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_frame(frame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(frame)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
