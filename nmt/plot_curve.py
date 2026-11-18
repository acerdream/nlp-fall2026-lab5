"""根据 ``nmt/results/train.log`` 画训练损失与验证困惑度曲线（题 (b)）。

用法::

    python nmt/plot_curve.py
"""

from __future__ import annotations

import re

from common import FIGURES_DIR, RESULTS_DIR, setup_cjk_font

TRAIN_RE = re.compile(r"step (\d+)/\d+ .*?train loss ([0-9.]+)")
VAL_RE = re.compile(r"step (\d+) \| val loss ([0-9.]+) \| val ppl ([0-9.]+)")


def parse_log(path):
    train_steps, train_losses = [], []
    val_steps, val_losses, val_ppls = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                train_steps.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
                continue
            m = VAL_RE.search(line)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_ppls.append(float(m.group(3)))
    return train_steps, train_losses, val_steps, val_losses, val_ppls


def main() -> None:
    log_path = RESULTS_DIR / "train.log"
    if not log_path.exists():
        raise SystemExit(f"找不到 {log_path}，请先运行 python nmt/train.py")
    train_steps, train_losses, val_steps, val_losses, val_ppls = parse_log(log_path)
    if not train_steps:
        raise SystemExit("日志中没有解析到训练损失")

    setup_cjk_font()
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    axes[0].plot(train_steps, train_losses, color="#2a6fb0", linewidth=1.0, label="训练损失")
    if val_steps:
        axes[0].plot(val_steps, val_losses, color="#d05a2a", marker="o", markersize=3,
                     linewidth=1.0, label="验证损失")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("训练与验证损失")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    if val_steps:
        axes[1].plot(val_steps, val_ppls, color="#2a8a5a", marker="o", markersize=3,
                     linewidth=1.2, label="验证困惑度")
        best_idx = min(range(len(val_ppls)), key=lambda i: val_ppls[i])
        axes[1].scatter([val_steps[best_idx]], [val_ppls[best_idx]], color="#c00000", zorder=3)
        axes[1].annotate(
            f"最佳 {val_ppls[best_idx]:.2f}\n(step {val_steps[best_idx]})",
            (val_steps[best_idx], val_ppls[best_idx]),
            textcoords="offset points", xytext=(-65, 12), fontsize=10,
        )
        axes[1].set_xlabel("训练步数")
        axes[1].set_ylabel("困惑度")
        axes[1].set_title("验证集困惑度")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

    fig.tight_layout()
    out = FIGURES_DIR / "loss_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()
