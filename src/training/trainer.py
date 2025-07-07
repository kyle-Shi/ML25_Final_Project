"""
通用训练器实现
适用于所有时间序列预测模型的训练器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm

from training.base_trainer import BaseTrainer
from models.base.base_model import BaseTimeSeriesModel
from core.utils import calculate_metrics, set_seed


class UniversalTrainer(BaseTrainer):
    """
    通用训练器
    适用于LSTM、Transformer等所有时间序列预测模型
    """
    
    def __init__(self, 
                 model: BaseTimeSeriesModel,
                 device: torch.device = None,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 gradient_clip_val: float = 1.0):
        """
        初始化通用训练器
        
        Args:
            model: 时间序列模型
            device: 计算设备
            criterion: 损失函数
            optimizer: 优化器
            gradient_clip_val: 梯度裁剪值
        """
        super().__init__(model, device, criterion, optimizer)
        self.gradient_clip_val = gradient_clip_val
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # 将数据移动到设备
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_val
                )
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # 将数据移动到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # 累计损失
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def get_sample_predictions(self, 
                              test_loader: DataLoader,
                              stats: Dict[str, Any] = None,
                              num_samples: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        获取多个样本的预测结果用于可视化
        
        Args:
            test_loader: 测试数据加载器
            stats: 数据统计信息
            num_samples: 样本数量
            
        Returns:
            (样本预测列表, 样本真实值列表)
        """
        self.model.eval()
        sample_predictions = []
        sample_targets = []
        
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(test_loader):
                if len(sample_predictions) >= num_samples:
                    break
                    
                # 将数据移动到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 预测
                outputs = self.model(X_batch)
                
                # 只取每个batch的第一个样本
                pred = outputs[0].cpu().numpy()
                target = y_batch[0].cpu().numpy()
                
                # 反标准化
                if stats:
                    from core.constants import TARGET_COLUMN
                    if TARGET_COLUMN in stats:
                        pred = pred * stats[TARGET_COLUMN]['std'] + stats[TARGET_COLUMN]['mean']
                        target = target * stats[TARGET_COLUMN]['std'] + stats[TARGET_COLUMN]['mean']
                
                sample_predictions.append(pred)
                sample_targets.append(target)
        
        return sample_predictions, sample_targets


def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_test: np.ndarray, 
                       y_test: np.ndarray,
                       batch_size: int = 32,
                       val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        batch_size: 批大小
        val_split: 验证集比例
        
    Returns:
        (训练数据加载器, 验证数据加载器, 测试数据加载器)
    """
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 划分训练集和验证集
    num_train = len(X_train_tensor)
    num_val = int(num_train * val_split)
    num_train_actual = num_train - num_val
    
    # 随机划分
    indices = torch.randperm(num_train)
    train_indices = indices[:num_train_actual]
    val_indices = indices[num_train_actual:]
    
    # 创建数据集
    train_dataset = TensorDataset(
        X_train_tensor[train_indices], 
        y_train_tensor[train_indices]
    )
    val_dataset = TensorDataset(
        X_train_tensor[val_indices], 
        y_train_tensor[val_indices]
    )
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def run_experiments(model_creator,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   test_loader: DataLoader,
                   stats: Dict[str, Any],
                   num_experiments: int = 5,
                   device: torch.device = None,
                   verbose: bool = False,
                   **train_kwargs) -> Dict[str, List[float]]:
    """
    运行多次实验
    
    Args:
        model_creator: 模型创建函数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        stats: 数据统计信息
        num_experiments: 实验次数
        device: 计算设备
        **train_kwargs: 训练参数
        
    Returns:
        实验结果字典
    """
    from core.constants import RANDOM_SEED
    
    all_results = {
        'MSE_mean': [],
        'MAE_mean': [],
        'Overall_MSE': [],
        'Overall_MAE': []
    }
    
    for i in range(num_experiments):
        print(f"\n=== 第 {i + 1} 次实验 ===")
        
        # 设置随机种子
        set_seed(RANDOM_SEED + i)
        
        # 创建模型和训练器
        model = model_creator()
        trainer = UniversalTrainer(model, device)
        
        # 训练模型
        trainer.train(train_loader, val_loader, verbose=verbose, **train_kwargs)
        
        # 评估模型
        results = trainer.evaluate(test_loader, stats)
        
        # 收集结果
        for key in all_results:
            if key in results:
                all_results[key].append(results[key])
        
        # 输出本次实验结果
        print("本次实验结果:")
        for key, value in results.items():
            print(f"  {key}: {value:.6f}")
    
    return all_results


def summarize_experiments(all_results: Dict[str, List[float]]) -> Dict[str, float]:
    """
    汇总实验结果
    
    Args:
        all_results: 所有实验结果
        
    Returns:
        汇总统计结果
    """
    summary = {}
    
    for key, values in all_results.items():
        if values:  # 确保列表不为空
            summary[f"{key}_final_mean"] = np.mean(values)
            summary[f"{key}_final_std"] = np.std(values)
    
    return summary 