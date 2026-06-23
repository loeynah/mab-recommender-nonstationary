#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的MAB算法测试脚本
验证基本功能是否正常
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from mab_algorithms import (
    UCBAlgorithm, DynamicArmUCBAlgorithm, PopularityAwareUCBAlgorithm,
    ColdStartUCBAlgorithm
)

def test_basic_ucb():
    """测试基础UCB算法"""
    print("=== 测试基础UCB算法 ===")
    
    # 创建简单的测试数据
    n_arms = 5
    n_rounds = 1000
    
    # 真实奖励分布
    true_rewards = np.array([0.3, 0.5, 0.7, 0.4, 0.6])
    
    # 初始化算法
    ucb = UCBAlgorithm(n_arms, alpha=1.0)
    
    # 运行实验
    rewards = []
    for t in range(n_rounds):
        arm = ucb.select_arm()
        # 添加噪声
        reward = np.random.normal(true_rewards[arm], 0.1)
        reward = max(0, min(1, reward))  # 限制在[0,1]范围
        ucb.update(arm, reward)
        rewards.append(reward)
    
    print(f"UCB累积奖励: {np.sum(rewards):.2f}")
    print(f"平均奖励: {np.mean(rewards):.3f}")
    print(f"最优臂选择次数: {np.argmax(ucb.arm_counts)}")
    print("基础UCB测试完成！")
    
    return ucb, rewards

def test_popularity_ucb():
    """测试流行度感知UCB算法"""
    print("\n=== 测试流行度感知UCB算法 ===")
    
    n_arms = 5
    n_rounds = 1000
    
    # 真实奖励分布
    true_rewards = np.array([0.3, 0.5, 0.7, 0.4, 0.6])
    
    # 流行度分数
    popularity_scores = np.array([0.2, 0.8, 0.9, 0.3, 0.7])
    
    # 初始化算法
    ucb = UCBAlgorithm(n_arms)
    popularity_ucb = PopularityAwareUCBAlgorithm(n_arms, popularity_scores)
    
    # 运行实验
    ucb_rewards = []
    popularity_rewards = []
    
    for t in range(n_rounds):
        # UCB
        ucb_arm = ucb.select_arm()
        reward = np.random.normal(true_rewards[ucb_arm], 0.1)
        reward = max(0, min(1, reward))
        ucb.update(ucb_arm, reward)
        ucb_rewards.append(reward)
        
        # 流行度UCB
        pop_arm = popularity_ucb.select_arm()
        reward = np.random.normal(true_rewards[pop_arm], 0.1)
        reward = max(0, min(1, reward))
        popularity_ucb.update(pop_arm, reward)
        popularity_rewards.append(reward)
    
    print(f"UCB累积奖励: {np.sum(ucb_rewards):.2f}")
    print(f"流行度UCB累积奖励: {np.sum(popularity_rewards):.2f}")
    print("流行度UCB测试完成！")
    
    return ucb, popularity_ucb, ucb_rewards, popularity_rewards

def test_dynamic_arms():
    """测试动态臂UCB算法"""
    print("\n=== 测试动态臂UCB算法 ===")
    
    initial_n_arms = 3
    n_rounds = 500
    
    # 初始化算法
    ucb = UCBAlgorithm(initial_n_arms)
    dynamic_ucb = DynamicArmUCBAlgorithm(initial_n_arms)
    
    # 第一阶段：只有3个臂
    print("第一阶段：3个臂")
    for t in range(n_rounds // 2):
        ucb_arm = ucb.select_arm()
        dynamic_arm = dynamic_ucb.select_arm()
        
        reward = np.random.normal(0.5, 0.1)
        reward = max(0, min(1, reward))
        
        ucb.update(ucb_arm, reward)
        dynamic_ucb.update(dynamic_arm, reward)
    
    # 第二阶段：添加新臂
    print("第二阶段：添加新臂")
    new_arms = [3, 4, 5]
    dynamic_ucb.add_new_arms(new_arms)
    
    for t in range(n_rounds // 2):
        ucb_arm = ucb.select_arm()
        dynamic_arm = dynamic_ucb.select_arm()
        
        reward = np.random.normal(0.5, 0.1)
        reward = max(0, min(1, reward))
        
        ucb.update(ucb_arm, reward)
        if dynamic_arm is not None:
            dynamic_ucb.update(dynamic_arm, reward)
    
    print(f"UCB累积奖励: {np.sum(ucb.rewards):.2f}")
    print(f"动态UCB累积奖励: {np.sum(dynamic_ucb.rewards):.2f}")
    print(f"动态UCB可用臂数量: {len(dynamic_ucb.available_arms)}")
    print("动态臂UCB测试完成！")
    
    return ucb, dynamic_ucb

def test_cold_start():
    """测试冷启动UCB算法"""
    print("\n=== 测试冷启动UCB算法 ===")
    
    n_arms = 5
    n_rounds = 1000
    
    # 创建简单相似度矩阵
    similarity_matrix = np.eye(n_arms)
    similarity_matrix[0, 1] = 0.8  # 臂0和臂1相似
    similarity_matrix[1, 0] = 0.8
    similarity_matrix[2, 3] = 0.7  # 臂2和臂3相似
    similarity_matrix[3, 2] = 0.7
    
    # 初始化算法
    ucb = UCBAlgorithm(n_arms)
    cold_start_ucb = ColdStartUCBAlgorithm(n_arms, similarity_matrix)
    
    # 标记一些臂为已知
    cold_start_ucb.mark_arm_as_known(0)
    cold_start_ucb.mark_arm_as_known(2)
    
    # 运行实验
    ucb_rewards = []
    cold_start_rewards = []
    
    for t in range(n_rounds):
        # UCB
        ucb_arm = ucb.select_arm()
        reward = np.random.normal(0.5, 0.1)
        reward = max(0, min(1, reward))
        ucb.update(ucb_arm, reward)
        ucb_rewards.append(reward)
        
        # 冷启动UCB
        cold_arm = cold_start_ucb.select_arm()
        reward = np.random.normal(0.5, 0.1)
        reward = max(0, min(1, reward))
        cold_start_ucb.update(cold_arm, reward)
        cold_start_rewards.append(reward)
    
    print(f"UCB累积奖励: {np.sum(ucb_rewards):.2f}")
    print(f"冷启动UCB累积奖励: {np.sum(cold_start_rewards):.2f}")
    print("冷启动UCB测试完成！")
    
    return ucb, cold_start_ucb, ucb_rewards, cold_start_rewards

def create_simple_visualization(results):
    """创建简单的可视化"""
    print("\n=== 创建可视化 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 基础UCB奖励
    if 'basic_ucb' in results:
        rewards = results['basic_ucb']
        axes[0, 0].plot(rewards[:100], alpha=0.7)
        axes[0, 0].set_title('基础UCB奖励')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True)
    
    # 2. 流行度UCB对比
    if 'popularity_comparison' in results:
        ucb_rewards, pop_rewards = results['popularity_comparison']
        axes[0, 1].plot(np.cumsum(ucb_rewards), label='UCB', alpha=0.7)
        axes[0, 1].plot(np.cumsum(pop_rewards), label='Popularity UCB', alpha=0.7)
        axes[0, 1].set_title('累积奖励对比')
        axes[0, 1].set_ylabel('累积奖励')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 3. 动态臂对比
    if 'dynamic_comparison' in results:
        ucb_rewards, dynamic_rewards = results['dynamic_comparison']
        axes[1, 0].plot(np.cumsum(ucb_rewards), label='UCB', alpha=0.7)
        axes[1, 0].plot(np.cumsum(dynamic_rewards), label='Dynamic UCB', alpha=0.7)
        axes[1, 0].set_title('动态臂对比')
        axes[1, 0].set_ylabel('累积奖励')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. 冷启动对比
    if 'cold_start_comparison' in results:
        ucb_rewards, cold_rewards = results['cold_start_comparison']
        axes[1, 1].plot(np.cumsum(ucb_rewards), label='UCB', alpha=0.7)
        axes[1, 1].plot(np.cumsum(cold_rewards), label='Cold Start UCB', alpha=0.7)
        axes[1, 1].set_title('冷启动对比')
        axes[1, 1].set_ylabel('累积奖励')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ml-25m/simple_test_results.png', dpi=300, bbox_inches='tight')
    print("可视化已保存到: ml-25m/simple_test_results.png")

def run_simple_tests():
    """运行所有简单测试"""
    print("开始简单测试...")
    
    results = {}
    
    # 测试基础UCB
    ucb, rewards = test_basic_ucb()
    results['basic_ucb'] = rewards
    
    # 测试流行度UCB
    ucb, pop_ucb, ucb_rewards, pop_rewards = test_popularity_ucb()
    results['popularity_comparison'] = (ucb_rewards, pop_rewards)
    
    # 测试动态臂UCB
    ucb, dynamic_ucb = test_dynamic_arms()
    results['dynamic_comparison'] = (ucb.rewards, dynamic_ucb.rewards)
    
    # 测试冷启动UCB
    ucb, cold_ucb, ucb_rewards, cold_rewards = test_cold_start()
    results['cold_start_comparison'] = (ucb_rewards, cold_rewards)
    
    # 创建可视化
    create_simple_visualization(results)
    
    # 生成简单报告
    report = []
    report.append("# 简单测试报告")
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n## 测试结果")
    report.append(f"- 基础UCB平均奖励: {np.mean(results['basic_ucb']):.3f}")
    report.append(f"- 流行度UCB性能提升: {((np.sum(results['popularity_comparison'][1]) / np.sum(results['popularity_comparison'][0]) - 1) * 100):.1f}%")
    report.append(f"- 动态UCB性能提升: {((np.sum(results['dynamic_comparison'][1]) / np.sum(results['dynamic_comparison'][0]) - 1) * 100):.1f}%")
    report.append(f"- 冷启动UCB性能提升: {((np.sum(results['cold_start_comparison'][1]) / np.sum(results['cold_start_comparison'][0]) - 1) * 100):.1f}%")
    
    with open('ml-25m/simple_test_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("简单测试完成！")
    print("报告已保存到: ml-25m/simple_test_report.md")

if __name__ == "__main__":
    run_simple_tests() 