#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAB实验运行器
将MAB算法应用到MovieLens数据上进行实验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from mab_algorithms import (
    UCBAlgorithm, DynamicArmUCBAlgorithm, PopularityAwareUCBAlgorithm,
    ContextualUCBAlgorithm, ColdStartUCBAlgorithm, create_similarity_matrix
)

class MABExperimentRunner:
    """MAB实验运行器"""
    
    def __init__(self, data_path):
        """
        初始化实验运行器
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.movie_features = None
        self.time_windows = None
        
    def load_processed_data(self):
        """加载预处理后的数据"""
        print("加载预处理后的数据...")
        
        self.ratings = pd.read_csv(f"{self.data_path}/processed_ratings.csv")
        self.movies = pd.read_csv(f"{self.data_path}/processed_movies.csv")
        self.movie_features = pd.read_csv(f"{self.data_path}/movie_features.csv")
        self.time_windows = pd.read_csv(f"{self.data_path}/time_windows.csv")
        
        print(f"数据加载完成:")
        print(f"- 评分记录: {len(self.ratings):,}")
        print(f"- 电影数量: {len(self.movies):,}")
        print(f"- 时间窗口: {len(self.time_windows)}")
        
    def experiment_1_dynamic_arms(self, max_rounds=10000):
        """实验1：动态臂MAB"""
        print("\n=== 实验1：动态臂MAB ===")
        
        # 按年份组织数据
        yearly_data = self.ratings.groupby('year')
        years = sorted(self.ratings['year'].unique())
        
        # 初始化算法
        initial_n_arms = 100  # 初始臂数量
        ucb = UCBAlgorithm(initial_n_arms)
        dynamic_ucb = DynamicArmUCBAlgorithm(initial_n_arms)
        
        # 记录结果
        results = {
            'year': [],
            'ucb_reward': [],
            'dynamic_ucb_reward': [],
            'ucb_regret': [],
            'dynamic_ucb_regret': [],
            'n_arms': []
        }
        
        current_arms = set()
        
        for year in years:
            print(f"处理年份: {year}")
            
            # 获取该年份的数据
            year_data = yearly_data.get_group(year)
            
            # 识别该年份的新电影
            year_movies = set(year_data['movieId'].unique())
            new_movies = year_movies - current_arms
            current_arms.update(new_movies)
            
            # 为动态UCB添加新臂
            if len(new_movies) > 0:
                dynamic_ucb.add_new_arms(list(new_movies))
            
            # 运行该年份的实验
            year_results = self._run_year_experiment(
                year_data, ucb, dynamic_ucb, max_rounds//len(years)
            )
            
            # 记录结果
            results['year'].append(year)
            results['ucb_reward'].append(year_results['ucb_reward'])
            results['dynamic_ucb_reward'].append(year_results['dynamic_ucb_reward'])
            results['ucb_regret'].append(year_results['ucb_regret'])
            results['dynamic_ucb_regret'].append(year_results['dynamic_ucb_regret'])
            results['n_arms'].append(len(current_arms))
        
        return pd.DataFrame(results)
    
    def experiment_2_popularity_aware(self, max_rounds=10000):
        """实验2：流行度感知MAB"""
        print("\n=== 实验2：流行度感知MAB ===")
        
        # 使用采样数据减少计算量
        sample_movies = self.movies.sample(n=min(1000, len(self.movies)), random_state=42)
        
        # 获取流行度分数
        popularity_scores = sample_movies['popularity_score'].values
        
        # 初始化算法
        n_arms = len(sample_movies)
        ucb = UCBAlgorithm(n_arms)
        popularity_ucb = PopularityAwareUCBAlgorithm(n_arms, popularity_scores)
        
        # 运行实验
        results = self._run_popularity_experiment(
            self.ratings[self.ratings['movieId'].isin(sample_movies['movieId'])], 
            ucb, popularity_ucb, max_rounds
        )
        
        return results
    
    def experiment_3_cold_start(self, max_rounds=10000):
        """实验3：冷启动MAB"""
        print("\n=== 实验3：冷启动MAB ===")
        
        # 简化相似度矩阵计算
        print("创建简化相似度矩阵...")
        
        # 只使用部分电影进行实验，减少计算量
        sample_movies = self.movies.sample(n=min(1000, len(self.movies)), random_state=42)
        sample_features = self.movie_features[self.movie_features['movieId'].isin(sample_movies['movieId'])]
        
        # 重置索引，确保索引从0开始
        sample_movies = sample_movies.reset_index(drop=True)
        sample_features = sample_features.reset_index(drop=True)
        
        # 计算简化相似度矩阵
        numeric_features = sample_features.select_dtypes(include=[np.number]).drop('movieId', axis=1, errors='ignore')
        similarity_matrix = np.eye(len(sample_features))  # 使用单位矩阵作为简化版本
        
        # 识别新电影
        new_movies = sample_movies[sample_movies['is_new_movie'] == True]['movieId'].values
        known_movies = sample_movies[sample_movies['is_new_movie'] == False]['movieId'].values
        
        # 初始化算法
        n_arms = len(sample_movies)
        ucb = UCBAlgorithm(n_arms)
        cold_start_ucb = ColdStartUCBAlgorithm(n_arms, similarity_matrix)
        
        # 标记已知电影 - 使用重置后的索引
        for movie_id in known_movies:
            movie_idx = sample_movies[sample_movies['movieId'] == movie_id].index[0]
            cold_start_ucb.mark_arm_as_known(movie_idx)
        
        # 运行实验
        results = self._run_cold_start_experiment(
            self.ratings[self.ratings['movieId'].isin(sample_movies['movieId'])], 
            ucb, cold_start_ucb, max_rounds, sample_movies
        )
        
        return results
    
    def _run_year_experiment(self, year_data, ucb, dynamic_ucb, max_rounds):
        """运行年度实验"""
        # 随机采样评分进行实验
        if len(year_data) > max_rounds:
            year_data = year_data.sample(n=max_rounds, random_state=42)
        
        ucb_rewards = []
        dynamic_ucb_rewards = []
        
        for _, row in year_data.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # 标准化奖励（0-1范围）
            normalized_rating = (rating - 0.5) / 4.5
            
            # UCB选择
            ucb_arm = ucb.select_arm()
            ucb.update(ucb_arm, normalized_rating)
            ucb_rewards.append(normalized_rating)
            
            # 动态UCB选择
            dynamic_arm = dynamic_ucb.select_arm()
            if dynamic_arm is not None:
                dynamic_ucb.update(dynamic_arm, normalized_rating)
                dynamic_ucb_rewards.append(normalized_rating)
        
        return {
            'ucb_reward': np.sum(ucb_rewards),
            'dynamic_ucb_reward': np.sum(dynamic_ucb_rewards),
            'ucb_regret': 0,  # 简化处理
            'dynamic_ucb_regret': 0
        }
    
    def _run_popularity_experiment(self, ratings_data, ucb, popularity_ucb, max_rounds):
        """运行流行度感知实验"""
        # 随机采样评分进行实验
        if len(ratings_data) > max_rounds:
            ratings_data = ratings_data.sample(n=max_rounds, random_state=42)
        
        ucb_rewards = []
        popularity_ucb_rewards = []
        
        for _, row in ratings_data.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # 标准化奖励
            normalized_rating = (rating - 0.5) / 4.5
            
            # UCB选择
            ucb_arm = ucb.select_arm()
            ucb.update(ucb_arm, normalized_rating)
            ucb_rewards.append(normalized_rating)
            
            # 流行度UCB选择
            popularity_arm = popularity_ucb.select_arm()
            popularity_ucb.update(popularity_arm, normalized_rating)
            popularity_ucb_rewards.append(normalized_rating)
        
        return {
            'ucb_reward': np.sum(ucb_rewards),
            'popularity_ucb_reward': np.sum(popularity_ucb_rewards),
            'ucb_regret': 0,
            'popularity_ucb_regret': 0
        }
    
    def _run_cold_start_experiment(self, ratings_data, ucb, cold_start_ucb, max_rounds, sample_movies):
        """运行冷启动实验"""
        # 随机采样评分进行实验
        if len(ratings_data) > max_rounds:
            ratings_data = ratings_data.sample(n=max_rounds, random_state=42)
        
        ucb_rewards = []
        cold_start_rewards = []
        
        for _, row in ratings_data.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # 找到电影索引
            movie_idx = sample_movies[sample_movies['movieId'] == movie_id].index[0]
            
            # 标准化奖励
            normalized_rating = (rating - 0.5) / 4.5
            
            # UCB选择
            ucb_arm = ucb.select_arm()
            ucb.update(ucb_arm, normalized_rating)
            ucb_rewards.append(normalized_rating)
            
            # 冷启动UCB选择
            cold_start_arm = cold_start_ucb.select_arm()
            cold_start_ucb.update(cold_start_arm, normalized_rating)
            cold_start_rewards.append(normalized_rating)
        
        return {
            'ucb_reward': np.sum(ucb_rewards),
            'cold_start_reward': np.sum(cold_start_rewards),
            'ucb_regret': 0,
            'cold_start_regret': 0
        }
    
    def create_visualizations(self, results_dict):
        """创建可视化图表"""
        print("创建可视化图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 实验1结果：动态臂
        if 'experiment_1' in results_dict:
            exp1_results = results_dict['experiment_1']
            
            # 累积奖励对比
            axes[0, 0].plot(exp1_results['year'], exp1_results['ucb_reward'], 
                           marker='o', label='UCB')
            axes[0, 0].plot(exp1_results['year'], exp1_results['dynamic_ucb_reward'], 
                           marker='s', label='Dynamic UCB')
            axes[0, 0].set_title('实验1：动态臂MAB - 累积奖励')
            axes[0, 0].set_xlabel('年份')
            axes[0, 0].set_ylabel('累积奖励')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 臂数量变化
            axes[0, 1].plot(exp1_results['year'], exp1_results['n_arms'], 
                           marker='o', color='green')
            axes[0, 1].set_title('臂数量变化')
            axes[0, 1].set_xlabel('年份')
            axes[0, 1].set_ylabel('臂数量')
            axes[0, 1].grid(True)
        
        # 实验2结果：流行度感知
        if 'experiment_2' in results_dict:
            exp2_results = results_dict['experiment_2']
            
            # 奖励对比
            algorithms = ['UCB', 'Popularity UCB']
            rewards = [exp2_results['ucb_reward'], exp2_results['popularity_ucb_reward']]
            
            axes[1, 0].bar(algorithms, rewards, color=['blue', 'orange'])
            axes[1, 0].set_title('实验2：流行度感知MAB - 累积奖励')
            axes[1, 0].set_ylabel('累积奖励')
            axes[1, 0].grid(True)
        
        # 实验3结果：冷启动
        if 'experiment_3' in results_dict:
            exp3_results = results_dict['experiment_3']
            
            # 奖励对比
            algorithms = ['UCB', 'Cold Start UCB']
            rewards = [exp3_results['ucb_reward'], exp3_results['cold_start_reward']]
            
            axes[1, 1].bar(algorithms, rewards, color=['blue', 'red'])
            axes[1, 1].set_title('实验3：冷启动MAB - 累积奖励')
            axes[1, 1].set_ylabel('累积奖励')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/experiment_results.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {self.data_path}/experiment_results.png")
    
    def generate_experiment_report(self, results_dict):
        """生成实验报告"""
        print("生成实验报告...")
        
        report = []
        report.append("# MAB实验报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 实验1结果
        if 'experiment_1' in results_dict:
            exp1_results = results_dict['experiment_1']
            report.append(f"\n## 实验1：动态臂MAB")
            report.append(f"- 时间跨度: {exp1_results['year'].min()} - {exp1_results['year'].max()}")
            report.append(f"- UCB累积奖励: {exp1_results['ucb_reward'].sum():.2f}")
            report.append(f"- 动态UCB累积奖励: {exp1_results['dynamic_ucb_reward'].sum():.2f}")
            report.append(f"- 性能提升: {((exp1_results['dynamic_ucb_reward'].sum() / exp1_results['ucb_reward'].sum() - 1) * 100):.1f}%")
        
        # 实验2结果
        if 'experiment_2' in results_dict:
            exp2_results = results_dict['experiment_2']
            report.append(f"\n## 实验2：流行度感知MAB")
            report.append(f"- UCB累积奖励: {exp2_results['ucb_reward']:.2f}")
            report.append(f"- 流行度UCB累积奖励: {exp2_results['popularity_ucb_reward']:.2f}")
            report.append(f"- 性能提升: {((exp2_results['popularity_ucb_reward'] / exp2_results['ucb_reward'] - 1) * 100):.1f}%")
        
        # 实验3结果
        if 'experiment_3' in results_dict:
            exp3_results = results_dict['experiment_3']
            report.append(f"\n## 实验3：冷启动MAB")
            report.append(f"- UCB累积奖励: {exp3_results['ucb_reward']:.2f}")
            report.append(f"- 冷启动UCB累积奖励: {exp3_results['cold_start_reward']:.2f}")
            report.append(f"- 性能提升: {((exp3_results['cold_start_reward'] / exp3_results['ucb_reward'] - 1) * 100):.1f}%")
        
        # 保存报告
        with open(f'{self.data_path}/experiment_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"报告已保存到: {self.data_path}/experiment_report.md")
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("开始运行MAB实验...")
        
        # 加载数据
        self.load_processed_data()
        
        # 运行实验
        results = {}
        
        # 实验1：动态臂MAB
        results['experiment_1'] = self.experiment_1_dynamic_arms()
        
        # 实验2：流行度感知MAB
        results['experiment_2'] = self.experiment_2_popularity_aware()
        
        # 实验3：冷启动MAB
        results['experiment_3'] = self.experiment_3_cold_start()
        
        # 创建可视化
        self.create_visualizations(results)
        
        # 生成报告
        self.generate_experiment_report(results)
        
        print("所有实验完成！")
        return results

if __name__ == "__main__":
    # 设置数据路径
    data_path = "ml-25m"
    
    # 创建实验运行器
    runner = MABExperimentRunner(data_path)
    
    # 运行所有实验
    results = runner.run_all_experiments() 