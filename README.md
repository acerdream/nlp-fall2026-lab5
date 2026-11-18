# 实验五：神经机器翻译（中译英）

北京语言大学 信息科学学院 · 自然语言处理（2026 秋）

- 姓名：TODO（由本人填写）
- 学号：202411581243

本仓库从零实现 Encoder-Decoder Transformer，在 9000 对中英平行句对上训练，
使用 beam search 解码并用 BLEU 评测，同时可视化最后一个解码器层的
cross-attention。

## 环境

本地与服务器使用同一套依赖：

```bash
conda create -n nlp2026 python=3.11
conda activate nlp2026
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers<5" "datasets<5" tokenizers nltk matplotlib
sudo apt install fonts-noto-cjk
```

实际运行环境：Python 3.10.21、PyTorch 2.5.1+cu121、transformers 4.57.6、
tokenizers 0.22.2、nltk 3.10.4、matplotlib 3.10.9。

## 数据

把课程下发的 `data/` 整体放在仓库根目录（`data/` 已被 `.gitignore` 忽略）：

```text
data/
├── zh_en/{train,validation,test}.jsonl
├── tokenizer_zh/
└── tokenizer_en/
```

## 复现

在仓库根目录依次运行：

```bash
python nmt/sanity_check.py
python nmt/train.py
python nmt/plot_curve.py
python nmt/batch_stats.py
python nmt/evaluate.py
python nmt/evaluate.py --greedy
python nmt/visualize_attention.py
```

GPU 选择（CUDA 默认编号与 `nvidia-smi` 不一致时）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python nmt/train.py
```

## 主要结果

| 项目 | 结果 |
| --- | --- |
| 模型参数量 | 398,336 |
| 训练 | 15 个 epoch，4230 步，3 分 01 秒（GTX 1080 Ti） |
| 最佳验证困惑度 | 86.87（第 4200 步） |
| Beam search（宽度 5）BLEU | 0.0127 |
| 贪心解码 BLEU | 0.0156 |
| 源/目标词表大小 | 3200 / 3200 |
| 训练 batch 平均真实 token 数 | 372.03 / 376.63 |
| 训练 batch 的 `<pad>` 比例 | 37.67% / 29.79% |

结果文件在 `nmt/results/`，图片在 `nmt/figures/`，报告在 `report/report.pdf`。

## 目录

```text
nlp-fall2026-lab5/
├── README.md
├── .gitignore
├── data/                    # 不提交
├── output/model.pt          # 不提交
├── nmt/
│   ├── common.py
│   ├── data.py
│   ├── model.py
│   ├── sanity_check.py
│   ├── train.py
│   ├── plot_curve.py
│   ├── batch_stats.py
│   ├── evaluate.py
│   ├── visualize_attention.py
│   ├── results/
│   └── figures/
└── report/
    ├── report.tex
    ├── refs.bib
    └── report.pdf
```
