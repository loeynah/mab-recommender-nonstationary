# 数据集说明

本项目使用 **MovieLens 25M** 数据集。

## 下载

1. 访问 [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)
2. 下载 `ml-25m.zip` 并解压
3. 将解压后的 CSV 文件放入项目根目录的 `ml-25m/` 文件夹

## 所需文件

| 文件 | 大小（约） | 用途 |
|------|-----------|------|
| `ratings.csv` | 650 MB | 用户评分（核心） |
| `movies.csv` | 3 MB | 电影元数据 |
| `tags.csv` | 37 MB | 用户标签 |
| `genome-scores.csv` | 415 MB | 物品相似度（冷启动实验） |
| `genome-tags.csv` | 18 KB | 基因组标签 |
| `links.csv` | 1.3 MB | 外部链接（可选） |

## 预处理

下载完成后，在项目根目录运行：

```bash
python data_preprocessing.py
```

将生成以下文件（同样较大，默认不纳入 Git）：

- `processed_ratings.csv` — 清洗后的评分
- `processed_movies.csv` — 清洗后的电影信息
- `movie_features.csv` — 电影特征
- `time_windows.csv` — 按年份划分的时间窗口（小文件，已在仓库中）

## 快速测试（无需完整数据）

`data_exploration.py` 和 `simple_test.py` 支持采样模式，可在部分数据上快速验证代码逻辑。

在 `data_exploration.py` 中设置 `sample_size` 参数即可限制读取行数。
