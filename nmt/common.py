"""公共工具：路径、随机种子、设备选择、中文字体与日志格式。

本文件属于骨架中已写好的部分，各脚本都从这里导入路径与工具函数。
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- 路径
NMT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NMT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = NMT_DIR / "results"
FIGURES_DIR = NMT_DIR / "figures"

for _p in (RESULTS_DIR, FIGURES_DIR, OUTPUT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 特殊符号
# 与原作业一致：<unk>=0、<s>=1、</s>=2、<pad>=3
UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = "<unk>", "<s>", "</s>", "<pad>"
UNK_ID, BOS_ID, EOS_ID, PAD_ID = 0, 1, 2, 3


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """按命令行参数选择设备；auto 时优先 CUDA。"""
    if spec == "auto":
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用的 GPU")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device


def describe_device(device: torch.device) -> str:
    """返回设备描述，用于训练日志第一行。"""
    if device.type != "cuda":
        return f"device=cpu ({os.cpu_count()} threads)"
    index = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total = torch.cuda.get_device_properties(index).total_memory / 1024 ** 3
    return f"device=cuda:{index} / {name} ({total:.1f} GiB, CUDA {torch.version.cuda})"


def setup_cjk_font() -> None:
    """让 matplotlib 能显示中文（服务器上需安装 fonts-noto-cjk）。"""
    import matplotlib
    from matplotlib import font_manager, rcParams

    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in available]
    rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.unicode_minus"] = False
    matplotlib.use("Agg")


def read_jsonl(path: str | Path) -> list[dict]:
    """读取每行一个 JSON 对象的平行语料文件。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class AverageMeter:
    """滑动平均，用于打印训练损失。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class Tee:
    """把标准输出同时写入日志文件。"""

    def __init__(self, path: str | Path) -> None:
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m{sec:.0f}s"


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
