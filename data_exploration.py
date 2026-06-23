#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens 25M 数据集探索脚本
用于分析数据的非平稳性特征，为MAB实验做准备
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MovieLensDataExplorer:
    def __init__(self, data_path, sample_size=None):
        """
        初始化数据探索器
        
        Args:
            data_path: 数据文件夹路径
            sample_size: 采样大小，None表示使用全部数据
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.ratings = None
        self.movies = None
        self.tags = None
        
    def load_data(self):
        """加载数据文件"""
        print("正在加载数据...")
        
        # 加载评分数据（使用更高效的方式）
        print("加载 ratings.csv...")
        if self.sample_size:
            # 如果指定了采样大小，先读取文件头获取列名
            self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv", nrows=self.sample_size)
            print(f"已加载 {self.sample_size:,} 条评分记录（采样数据）")
        else:
            # 尝试使用更高效的数据类型
            dtypes = {
                'userId': 'int32',
                'movieId': 'int32', 
                'rating': 'float32',
                'timestamp': 'int64'
            }
            self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv", dtype=dtypes)
            print(f"已加载 {len(self.ratings):,} 条评分记录")
        
        # 加载电影数据
        print("加载 movies.csv...")
        self.movies = pd.read_csv(f"{self.data_path}/movies.csv")
        
        # 加载标签数据（如果文件太大，也进行采样）
        print("加载 tags.csv...")
        if self.sample_size:
            self.tags = pd.read_csv(f"{self.data_path}/tags.csv", nrows=self.sample_size)
        else:
            self.tags = pd.read_csv(f"{self.data_path}/tags.csv")
        
        print("数据加载完成！")
        
    def convert_timestamps(self):
        """转换时间戳为可读格式"""
        print("转换时间戳...")
        
        # 转换评分时间戳
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['datetime'].dt.year
        self.ratings['month'] = self.ratings['datetime'].dt.month
        self.ratings['quarter'] = self.ratings['datetime'].dt.quarter
        
        # 转换标签时间戳
        self.tags['datetime'] = pd.to_datetime(self.tags['timestamp'], unit='s')
        self.tags['year'] = self.tags['datetime'].dt.year
        
        print("时间戳转换完成！")
        
    def basic_statistics(self):
        """基本统计信息"""
        print("\n=== 基本统计信息 ===")
        
        print(f"评分数据规模: {len(self.ratings):,} 条记录")
        print(f"用户数量: {self.ratings['userId'].nunique():,}")
        print(f"电影数量: {self.ratings['movieId'].nunique():,}")
        print(f"时间跨度: {self.ratings['year'].min()} - {self.ratings['year'].max()}")
        
        print(f"\n评分统计:")
        print(self.ratings['rating'].describe())
        
        print(f"\n标签数据规模: {len(self.tags):,} 条记录")
        print(f"电影数量: {self.tags['movieId'].nunique():,}")
        
    def temporal_analysis(self):
        """时间分布分析"""
        print("\n=== 时间分布分析 ===")
        
        # 按年份统计评分数量
        yearly_ratings = self.ratings.groupby('year').size()
        print("年度评分数量:")
        print(yearly_ratings)
        
        # 按季度统计
        quarterly_ratings = self.ratings.groupby(['year', 'quarter']).size().reset_index(name='count')
        print("\n季度评分数量 (前10个季度):")
        print(quarterly_ratings.head(10))
        
        return yearly_ratings, quarterly_ratings
        
    def user_activity_analysis(self):
        """用户活跃度分析"""
        print("\n=== 用户活跃度分析 ===")
        
        # 用户评分数量分布
        user_activity = self.ratings.groupby('userId').size()
        
        print("用户评分数量统计:")
        print(user_activity.describe())
        
        # 活跃用户（评分数量前10%）
        active_threshold = user_activity.quantile(0.9)
        active_users = user_activity[user_activity >= active_threshold]
        print(f"\n活跃用户数量 (评分数量 >= {active_threshold:.0f}): {len(active_users)}")
        
        return user_activity
        
    def movie_popularity_analysis(self):
        """电影流行度分析"""
        print("\n=== 电影流行度分析 ===")
        
        # 电影评分数量分布
        movie_popularity = self.ratings.groupby('movieId').size()
        
        print("电影评分数量统计:")
        print(movie_popularity.describe())
        
        # 热门电影（评分数量前10%）
        popular_threshold = movie_popularity.quantile(0.9)
        popular_movies = movie_popularity[movie_popularity >= popular_threshold]
        print(f"\n热门电影数量 (评分数量 >= {popular_threshold:.0f}): {len(popular_movies)}")
        
        return movie_popularity
        
    def non_stationarity_analysis(self):
        """非平稳性分析"""
        print("\n=== 非平稳性分析 ===")
        
        # 1. 平均评分随时间变化
        yearly_avg_rating = self.ratings.groupby('year')['rating'].mean()
        print("年度平均评分:")
        print(yearly_avg_rating)
        
        # 2. 评分分布随时间变化
        yearly_rating_dist = self.ratings.groupby(['year', 'rating']).size().unstack(fill_value=0)
        print("\n年度评分分布 (前5年):")
        print(yearly_rating_dist.head())
        
        # 3. 用户活跃度随时间变化
        yearly_active_users = self.ratings.groupby('year')['userId'].nunique()
        print("\n年度活跃用户数:")
        print(yearly_active_users)
        
        return yearly_avg_rating, yearly_rating_dist, yearly_active_users
        
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== 创建可视化图表 ===")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 年度评分数量趋势
        yearly_ratings = self.ratings.groupby('year').size()
        axes[0, 0].plot(yearly_ratings.index, yearly_ratings.values, marker='o')
        axes[0, 0].set_title('年度评分数量趋势')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('评分数量')
        axes[0, 0].grid(True)
        
        # 2. 年度平均评分变化
        yearly_avg_rating = self.ratings.groupby('year')['rating'].mean()
        axes[0, 1].plot(yearly_avg_rating.index, yearly_avg_rating.values, marker='o', color='orange')
        axes[0, 1].set_title('年度平均评分变化')
        axes[0, 1].set_xlabel('年份')
        axes[0, 1].set_ylabel('平均评分')
        axes[0, 1].grid(True)
        
        # 3. 评分分布
        axes[1, 0].hist(self.ratings['rating'], bins=10, alpha=0.7, color='green')
        axes[1, 0].set_title('评分分布')
        axes[1, 0].set_xlabel('评分')
        axes[1, 0].set_ylabel('频次')
        
        # 4. 用户活跃度分布
        user_activity = self.ratings.groupby('userId').size()
        axes[1, 1].hist(user_activity, bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('用户评分数量分布')
        axes[1, 1].set_xlabel('评分数量')
        axes[1, 1].set_ylabel('用户数量')
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/data_exploration_plots.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {self.data_path}/data_exploration_plots.png")
        
    def generate_report(self):
        """生成数据探索报告"""
        print("\n=== 生成数据探索报告 ===")
        
        report = []
        report.append("# MovieLens 25M 数据集探索报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.sample_size:
            report.append(f"\n**注意：此报告基于 {self.sample_size:,} 条采样数据生成**")
        
        # 基本统计
        report.append(f"\n## 基本统计信息")
        report.append(f"- 评分数据规模: {len(self.ratings):,} 条记录")
        report.append(f"- 用户数量: {self.ratings['userId'].nunique():,}")
        report.append(f"- 电影数量: {self.ratings['movieId'].nunique():,}")
        report.append(f"- 时间跨度: {self.ratings['year'].min()} - {self.ratings['year'].max()}")
        
        # 评分统计
        rating_stats = self.ratings['rating'].describe()
        report.append(f"\n## 评分统计")
        report.append(f"- 平均评分: {rating_stats['mean']:.2f}")
        report.append(f"- 标准差: {rating_stats['std']:.2f}")
        report.append(f"- 最小评分: {rating_stats['min']:.1f}")
        report.append(f"- 最大评分: {rating_stats['max']:.1f}")
        
        # 时间分析
        yearly_ratings = self.ratings.groupby('year').size()
        report.append(f"\n## 时间分布")
        report.append(f"- 数据最多年份: {yearly_ratings.idxmax()} ({yearly_ratings.max():,} 条评分)")
        report.append(f"- 数据最少年份: {yearly_ratings.idxmin()} ({yearly_ratings.min():,} 条评分)")
        
        # 非平稳性分析
        yearly_avg_rating = self.ratings.groupby('year')['rating'].mean()
        report.append(f"\n## 非平稳性分析")
        report.append(f"- 评分变化范围: {yearly_avg_rating.max():.2f} - {yearly_avg_rating.min():.2f}")
        report.append(f"- 评分变化标准差: {yearly_avg_rating.std():.3f}")
        
        # 保存报告
        with open(f'{self.data_path}/data_exploration_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"报告已保存到: {self.data_path}/data_exploration_report.md")
        
    def run_exploration(self):
        """运行完整的数据探索流程"""
        print("开始MovieLens 25M数据集探索...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 转换时间戳
        self.convert_timestamps()
        
        # 3. 基本统计
        self.basic_statistics()
        
        # 4. 时间分析
        yearly_ratings, quarterly_ratings = self.temporal_analysis()
        
        # 5. 用户活跃度分析
        user_activity = self.user_activity_analysis()
        
        # 6. 电影流行度分析
        movie_popularity = self.movie_popularity_analysis()
        
        # 7. 非平稳性分析
        yearly_avg_rating, yearly_rating_dist, yearly_active_users = self.non_stationarity_analysis()
        
        # 8. 创建可视化
        self.create_visualizations()
        
        # 9. 生成报告
        self.generate_report()
        
        print("\n数据探索完成！")
        print(f"请查看以下文件:")
        print(f"- 数据探索报告: {self.data_path}/data_exploration_report.md")
        print(f"- 可视化图表: {self.data_path}/data_exploration_plots.png")

if __name__ == "__main__":
    # 设置数据路径
    data_path = "ml-25m"
    
    # 创建数据探索器（使用100万条数据进行采样）
    explorer = MovieLensDataExplorer(data_path, sample_size=1000000)
    
    # 运行探索
    explorer.run_exploration() 