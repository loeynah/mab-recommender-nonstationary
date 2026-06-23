#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens 25M 数据预处理模块
用于MAB实验的数据准备和特征工程
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MovieLensPreprocessor:
    def __init__(self, data_path):
        """
        初始化数据预处理器
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.ratings = None
        self.movies = None
        self.processed_data = None
        
    def load_data(self, sample_size=None):
        """加载原始数据"""
        print("加载原始数据...")
        
        # 加载评分数据
        if sample_size:
            self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv", nrows=sample_size)
        else:
            dtypes = {
                'userId': 'int32',
                'movieId': 'int32', 
                'rating': 'float32',
                'timestamp': 'int64'
            }
            self.ratings = pd.read_csv(f"{self.data_path}/ratings.csv", dtype=dtypes)
        
        # 加载电影数据
        self.movies = pd.read_csv(f"{self.data_path}/movies.csv")
        
        print(f"已加载 {len(self.ratings):,} 条评分记录")
        print(f"电影数量: {len(self.movies):,}")
        
    def convert_timestamps(self):
        """转换时间戳"""
        print("转换时间戳...")
        
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        self.ratings['year'] = self.ratings['datetime'].dt.year
        self.ratings['month'] = self.ratings['datetime'].dt.month
        self.ratings['quarter'] = self.ratings['datetime'].dt.quarter
        
        print("时间戳转换完成")
        
    def extract_movie_features(self):
        """提取电影特征"""
        print("提取电影特征...")
        
        # 提取发行年份
        self.movies['release_year'] = self.movies['title'].str.extract(r'\((\d{4})\)').astype(float)
        
        # 处理缺失的发行年份
        self.movies['release_year'].fillna(self.movies['release_year'].median(), inplace=True)
        
        # 提取电影类型
        genres = self.movies['genres'].str.get_dummies(sep='|')
        self.movies = pd.concat([self.movies, genres], axis=1)
        
        # 计算电影年龄（相对于数据集的起始年份）
        min_year = self.ratings['year'].min()
        self.movies['movie_age'] = self.movies['release_year'] - min_year
        
        print(f"电影特征提取完成，包含 {len(genres.columns)} 种类型")
        
    def calculate_popularity_features(self):
        """计算流行度特征"""
        print("计算流行度特征...")
        
        # 按电影统计评分数量和平均评分
        movie_stats = self.ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique'
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'rating_mean', 'rating_std', 'unique_users']
        
        # 计算流行度分数（综合评分数量和平均评分）
        movie_stats['popularity_score'] = (
            movie_stats['rating_count'] * movie_stats['rating_mean'] / 
            movie_stats['rating_count'].max()
        )
        
        # 合并到电影数据
        self.movies = self.movies.merge(movie_stats, on='movieId', how='left')
        
        # 填充缺失值
        self.movies['rating_count'].fillna(0, inplace=True)
        self.movies['rating_mean'].fillna(3.0, inplace=True)
        self.movies['popularity_score'].fillna(0, inplace=True)
        
        print("流行度特征计算完成")
        
    def create_time_windows(self, window_type='year'):
        """创建时间窗口"""
        print(f"创建{window_type}时间窗口...")
        
        if window_type == 'year':
            self.ratings['time_window'] = self.ratings['year']
        elif window_type == 'quarter':
            self.ratings['time_window'] = self.ratings['year'].astype(str) + 'Q' + self.ratings['quarter'].astype(str)
        
        # 统计每个时间窗口的数据
        window_stats = self.ratings.groupby('time_window').agg({
            'userId': 'nunique',
            'movieId': 'nunique',
            'rating': 'count'
        }).reset_index()
        
        window_stats.columns = ['time_window', 'unique_users', 'unique_movies', 'total_ratings']
        
        print(f"时间窗口创建完成，共 {len(window_stats)} 个窗口")
        print("时间窗口统计:")
        print(window_stats.head(10))
        
        return window_stats
        
    def identify_new_movies(self):
        """识别新电影（冷启动场景）"""
        print("识别新电影...")
        
        # 按年份统计电影首次出现的时间
        movie_first_appearance = self.ratings.groupby('movieId')['year'].min().reset_index()
        movie_first_appearance.columns = ['movieId', 'first_year']
        
        # 合并到电影数据
        self.movies = self.movies.merge(movie_first_appearance, on='movieId', how='left')
        
        # 计算电影在数据集中的"年龄"
        min_year = self.ratings['year'].min()
        self.movies['dataset_age'] = self.movies['first_year'] - min_year
        
        # 识别新电影（在数据集中的年龄小于等于2年的电影）
        self.movies['is_new_movie'] = self.movies['dataset_age'] <= 2
        
        new_movies_count = self.movies['is_new_movie'].sum()
        print(f"识别出 {new_movies_count} 部新电影（冷启动场景）")
        
    def create_mab_environment(self, min_ratings=10, min_users=5):
        """创建MAB实验环境"""
        print("创建MAB实验环境...")
        
        # 过滤数据：只保留有足够评分的电影
        movie_rating_counts = self.ratings.groupby('movieId').size()
        valid_movies = movie_rating_counts[movie_rating_counts >= min_ratings].index
        
        # 过滤数据：只保留有足够用户的电影
        movie_user_counts = self.ratings.groupby('movieId')['userId'].nunique()
        valid_movies = valid_movies.intersection(
            movie_user_counts[movie_user_counts >= min_users].index
        )
        
        # 过滤评分数据
        filtered_ratings = self.ratings[self.ratings['movieId'].isin(valid_movies)]
        
        # 过滤电影数据
        filtered_movies = self.movies[self.movies['movieId'].isin(valid_movies)]
        
        print(f"过滤后数据规模:")
        print(f"- 评分记录: {len(filtered_ratings):,}")
        print(f"- 电影数量: {len(filtered_movies):,}")
        print(f"- 用户数量: {filtered_ratings['userId'].nunique():,}")
        
        # 创建实验数据结构
        self.processed_data = {
            'ratings': filtered_ratings,
            'movies': filtered_movies,
            'time_windows': self.create_time_windows(),
            'movie_features': self._extract_movie_features_for_mab(filtered_movies)
        }
        
        print("MAB实验环境创建完成")
        
    def _extract_movie_features_for_mab(self, movies_df):
        """为MAB算法提取电影特征"""
        # 选择用于MAB的特征
        feature_columns = [
            'movieId', 'release_year', 'movie_age', 'dataset_age',
            'rating_count', 'rating_mean', 'popularity_score', 'is_new_movie'
        ]
        
        # 添加电影类型特征
        genre_columns = [col for col in movies_df.columns if col not in [
            'movieId', 'title', 'genres', 'release_year', 'movie_age', 
            'dataset_age', 'rating_count', 'rating_mean', 'rating_std', 
            'unique_users', 'popularity_score', 'is_new_movie', 'first_year'
        ]]
        
        feature_columns.extend(genre_columns)
        
        movie_features = movies_df[feature_columns].copy()
        
        # 标准化数值特征
        numeric_features = ['release_year', 'movie_age', 'dataset_age', 
                          'rating_count', 'rating_mean', 'popularity_score']
        
        for feature in numeric_features:
            if feature in movie_features.columns:
                mean_val = movie_features[feature].mean()
                std_val = movie_features[feature].std()
                if std_val > 0:
                    movie_features[feature] = (movie_features[feature] - mean_val) / std_val
        
        return movie_features
        
    def save_processed_data(self):
        """保存处理后的数据"""
        print("保存处理后的数据...")
        
        # 保存评分数据
        self.processed_data['ratings'].to_csv(
            f"{self.data_path}/processed_ratings.csv", index=False
        )
        
        # 保存电影特征数据
        self.processed_data['movies'].to_csv(
            f"{self.data_path}/processed_movies.csv", index=False
        )
        
        # 保存电影特征（用于MAB）
        self.processed_data['movie_features'].to_csv(
            f"{self.data_path}/movie_features.csv", index=False
        )
        
        # 保存时间窗口统计
        self.processed_data['time_windows'].to_csv(
            f"{self.data_path}/time_windows.csv", index=False
        )
        
        print("数据保存完成")
        
    def run_preprocessing(self, sample_size=None, min_ratings=10, min_users=5):
        """运行完整的数据预处理流程"""
        print("开始数据预处理...")
        
        # 1. 加载数据
        self.load_data(sample_size)
        
        # 2. 转换时间戳
        self.convert_timestamps()
        
        # 3. 提取电影特征
        self.extract_movie_features()
        
        # 4. 计算流行度特征
        self.calculate_popularity_features()
        
        # 5. 识别新电影
        self.identify_new_movies()
        
        # 6. 创建MAB环境
        self.create_mab_environment(min_ratings, min_users)
        
        # 7. 保存处理后的数据
        self.save_processed_data()
        
        print("数据预处理完成！")
        print(f"处理后的数据文件保存在: {self.data_path}/")

if __name__ == "__main__":
    # 设置数据路径
    data_path = "ml-25m"
    
    # 创建预处理器
    preprocessor = MovieLensPreprocessor(data_path)
    
    # 运行预处理（使用100万条数据进行采样）
    preprocessor.run_preprocessing(sample_size=1000000, min_ratings=10, min_users=5) 