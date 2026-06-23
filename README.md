# 非平稳环境下推荐系统的多臂老虎机算法研究

**Multi-Armed Bandit Algorithms for Recommender Systems in Non-Stationary Environments**

作者：张雪纯 · 南京信息工程大学

---

## 项目简介

本项目针对推荐系统在**非平稳环境**（动态臂集合、流行度变化、冷启动）下的探索-利用问题，提出并实现了三种改进 UCB 算法：

| 算法 | 解决的问题 |
|------|-----------|
| **Dynamic-Arm-Management UCB** | 新物品加入 / 旧物品退出时的动态臂管理 |
| **Popularity-Aware UCB** | 利用时间窗口内的流行度信息优化决策 |
| **Cold-Start UCB** | 基于物品相似度矩阵的知识迁移，缓解冷启动 |

在 [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) 数据集上完成实验验证。

---

## 目录结构

```
.
├── README.md                      # 本文件
├── requirements.txt               # Python 依赖
│
├── docs/                          # 论文与实验文档
│   ├── paper_en.md                # 英文论文正文（最新）
│   ├── paper_zh.md                # 中文论文初稿
│   ├── paper.tex                  # LaTeX 版本
│   ├── paper_outline.md           # 论文大纲
│   └── experiment_design.md       # 实验设计说明
│
├── figures/                       # 论文配图（最终版）
│   ├── dynamic_arm_flowchart.png
│   ├── framework_architecture.png
│   ├── dynamic_vs_static_comparison.png
│   ├── popularity_aware_analysis.png
│   ├── cold_start_analysis.png
│   └── performance_comparison.png
│
├── data/
│   └── README.md                  # 数据集下载与预处理说明
│
├── ml-25m/                        # 数据目录（大文件需自行下载）
│   ├── README.txt                 # MovieLens 官方说明
│   ├── time_windows.csv           # 预处理产物（小文件，已纳入版本库）
│   └── *.md / *.png               # 实验报告与结果图
│
├── mab_algorithms.py              # 核心：MAB 算法实现
├── data_exploration.py            # 数据探索
├── data_preprocessing.py          # 数据预处理
├── experiment_runner.py           # 完整实验运行
├── simple_test.py                 # 快速验证测试
└── generate_high_quality_figures.py  # 论文配图生成（需本地数据）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

MovieLens 25M 原始数据约 **1 GB**，未纳入 Git 仓库。请按 [`data/README.md`](data/README.md) 下载并放入 `ml-25m/` 目录。

### 3. 运行实验

```bash
# 数据探索（可采样，加快速度）
python data_exploration.py

# 数据预处理（生成 processed_*.csv 等）
python data_preprocessing.py

# 完整三组实验
python experiment_runner.py

# 快速冒烟测试
python simple_test.py
```

### 4. 重新生成论文配图

```bash
python generate_high_quality_figures.py
```

> 需要本地完整的 `ml-25m/` 原始数据。输出默认写入 `figures/` 目录。

---

## 论文文件说明

| 文件 | 说明 |
|------|------|
| `docs/paper_en.md` | 英文正文，对应 `paper.md`，内容最新（2025-09） |
| `docs/paper_zh.md` | 中文初稿 |
| `docs/paper.tex` | LaTeX 格式，可用 `pdflatex` 编译 |
| `docs/paper_outline.md` | 写作大纲与章节结构 |
| `docs/experiment_design.md` | 三组实验的设计文档 |

---

## 核心算法模块

`mab_algorithms.py` 包含：

- `StandardUCB` — 标准 UCB 基线
- `DynamicArmUCB` — 动态臂管理 UCB
- `PopularityAwareUCB` — 流行度感知 UCB
- `ColdStartUCB` — 冷启动 UCB（相似度知识迁移）

---

## 注意事项

1. **大文件不上传 GitHub**：`ratings.csv`（~650 MB）等原始/预处理大 CSV 已在 `.gitignore` 中排除。
2. **旧版脚本**：`generate_paper_figures*.py` 为早期迭代版本，已被 `generate_high_quality_figures.py` 取代，不纳入版本库。
3. **实验采样**：`data_preprocessing.py` 默认对评分数据采样以加速实验，完整数据运行需修改 `sample_size` 参数。

---

## 引用

如使用本项目，请引用 MovieLens 数据集：

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI: [10.1145/2827872](https://doi.org/10.1145/2827872)

---

## License

代码仅供学术研究使用。MovieLens 数据遵循 [GroupLens 使用条款](https://grouplens.org/datasets/movielens/)。
