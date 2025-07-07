"""
基础时间序列模型抽象类
定义了所有时间序列预测模型的通用接口
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseTimeSeriesModel(nn.Module, ABC):
    """
    基础时间序列模型抽象类
    所有时间序列预测模型都应该继承这个类
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0.2,
                 **kwargs):
        """
        初始化基础模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            output_dim: 输出维度
            dropout: Dropout率
            **kwargs: 其他模型特定参数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 模型特定参数
        self.model_params = kwargs
        
        # 初始化模型架构
        self._build_model()
    
    @abstractmethod
    def _build_model(self):
        """
        构建模型架构
        子类必须实现这个方法
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            预测输出 (batch_size, output_dim)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            **self.model_params
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"=== 模型信息 ===")
        print(f"模型类型: {info['model_type']}")
        print(f"输入维度: {info['input_dim']}")
        print(f"隐藏层维度: {info['hidden_dim']}")
        print(f"层数: {info['num_layers']}")
        print(f"输出维度: {info['output_dim']}")
        print(f"Dropout率: {info['dropout']}")
        print(f"参数总数: {info['total_params']:,}")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"模型大小: {info['model_size_mb']:.2f} MB")
    
    def initialize_weights(self):
        """
        初始化模型权重
        子类可以重写这个方法来实现自定义的权重初始化
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        计算模型参数数量
        
        Returns:
            (总参数数, 可训练参数数)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def freeze_parameters(self, freeze: bool = True):
        """
        冻结或解冻模型参数
        
        Args:
            freeze: True表示冻结参数，False表示解冻参数
        """
        for param in self.parameters():
            param.requires_grad = not freeze
    
    def get_device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device
    
    def save_checkpoint(self, filepath: str, optimizer=None, epoch: int = 0, loss: float = 0.0):
        """
        保存模型检查点
        
        Args:
            filepath: 保存路径
            optimizer: 优化器状态
            epoch: 当前epoch
            loss: 当前损失
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'epoch': epoch,
            'loss': loss
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, optimizer=None):
        """
        加载模型检查点
        
        Args:
            filepath: 模型文件路径
            optimizer: 优化器（如果需要加载优化器状态）
            
        Returns:
            包含epoch和loss信息的字典
        """
        checkpoint = torch.load(filepath, map_location=self.get_device())
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        info = {'epoch': checkpoint.get('epoch', 0), 'loss': checkpoint.get('loss', 0.0)}
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return info 