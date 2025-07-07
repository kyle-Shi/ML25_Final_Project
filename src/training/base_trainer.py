"""
基础训练器抽象类
定义所有训练器的通用接口
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import time

from models.base.base_model import BaseTimeSeriesModel


class BaseTrainer(ABC):
    """
    基础训练器抽象类
    所有训练器都应该继承这个类
    """
    
    def __init__(self, 
                 model: BaseTimeSeriesModel,
                 device: torch.device = None,
                 criterion: nn.Module = None,
                 optimizer: torch.optim.Optimizer = None):
        """
        初始化基础训练器
        
        Args:
            model: 时间序列模型
            device: 计算设备
            criterion: 损失函数
            optimizer: 优化器
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 默认损失函数
        self.criterion = criterion or nn.MSELoss()
        
        # 默认优化器
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 学习率调度器
        self.scheduler = None
        
        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            平均训练损失
        """
        pass
    
    @abstractmethod
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        验证一个epoch
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            平均验证损失
        """
        pass
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              epochs: int = 100,
              patience: int = 10,
              verbose: bool = True,
              save_best: bool = True,
              **kwargs) -> Dict[str, List[float]]:
        """
        完整的训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            patience: 早停耐心值
            verbose: 是否输出训练信息
            save_best: 是否保存最佳模型
            **kwargs: 其他参数
            
        Returns:
            训练历史字典
        """
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        if verbose:
            print("开始训练...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if save_best:
                        self._save_best_model()
                else:
                    self.patience_counter += 1
                    
                # 早停
                if self.patience_counter >= patience:
                    if verbose:
                        print(f"早停于第 {epoch + 1} 轮")
                    break
            
            # 学习率调度
            if self.scheduler is not None:
                if val_loss is not None:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 输出训练信息
            if verbose:
                epoch_time = time.time() - start_time
                val_info = f" - Val Loss: {val_loss:.6f}" if val_loss is not None else ""
                if (epoch+1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                            f"Train Loss: {train_loss:.6f}{val_info} - "
                            f"Time: {epoch_time:.2f}s")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(self, 
                test_loader: DataLoader,
                stats: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            test_loader: 测试数据加载器
            stats: 数据统计信息（用于反标准化）
            
        Returns:
            (预测值, 真实值)
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                # 将数据移动到设备
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 预测
                outputs = self.model(X_batch)
                
                # 收集结果
                predictions.append(outputs.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        
        # 合并所有结果
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # 反标准化
        if stats is not None:
            from core.constants import TARGET_COLUMN
            if TARGET_COLUMN in stats:
                predictions = predictions * stats[TARGET_COLUMN]['std'] + stats[TARGET_COLUMN]['mean']
                targets = targets * stats[TARGET_COLUMN]['std'] + stats[TARGET_COLUMN]['mean']
        
        return predictions, targets
    
    def evaluate(self, 
                test_loader: DataLoader,
                stats: Dict[str, Any] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_loader: 测试数据加载器
            stats: 数据统计信息
            
        Returns:
            评估指标字典
        """
        predictions, targets = self.predict(test_loader, stats)
        
        from core.utils import calculate_metrics
        
        # 计算整体指标
        overall_mse, overall_mae = calculate_metrics(predictions.flatten(), targets.flatten())
        
        # 计算序列平均指标
        if len(predictions.shape) > 1:
            sequence_mses = []
            sequence_maes = []
            
            for i in range(predictions.shape[0]):
                mse, mae = calculate_metrics(predictions[i], targets[i])
                sequence_mses.append(mse)
                sequence_maes.append(mae)
            
            mse_mean = np.mean(sequence_mses)
            mse_std = np.std(sequence_mses)
            mae_mean = np.mean(sequence_maes)
            mae_std = np.std(sequence_maes)
        else:
            mse_mean = overall_mse
            mse_std = 0.0
            mae_mean = overall_mae
            mae_std = 0.0
        
        return {
            'Overall_MSE': overall_mse,
            'Overall_MAE': overall_mae,
            'MSE_mean': mse_mean,
            'MSE_std': mse_std,
            'MAE_mean': mae_mean,
            'MAE_std': mae_std
        }
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """设置优化器"""
        self.optimizer = optimizer
    
    def set_criterion(self, criterion: nn.Module):
        """设置损失函数"""
        self.criterion = criterion
    
    def set_scheduler(self, scheduler):
        """设置学习率调度器"""
        self.scheduler = scheduler
    
    def _save_best_model(self):
        """保存最佳模型（私有方法）"""
        # 这里可以实现模型保存逻辑
        pass
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def reset_training_history(self):
        """重置训练历史"""
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0 