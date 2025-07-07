"""
LSTM模型实现
继承BaseTimeSeriesModel，用于电力消费预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.base.base_model import BaseTimeSeriesModel


class PowerLSTM(BaseTimeSeriesModel):
    """
    多变量时间序列LSTM模型
    用于电力消费预测
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2, **kwargs):
        """
        初始化LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出序列长度
            dropout: Dropout率
            **kwargs: 其他参数
        """
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout, **kwargs)
    
    def _build_model(self):
        """构建LSTM模型架构"""
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, self.output_dim)
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim // 2)
        
        # 初始化权重
        self.initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            预测结果，形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 应用dropout
        last_output = self.dropout_layer(last_output)
        
        # 全连接层
        output = F.relu(self.fc1(last_output))
        output = self.batch_norm(output)
        output = self.dropout_layer(output)
        output = self.fc2(output)
        
        return output


class PowerLSTMWithAttention(BaseTimeSeriesModel):
    """
    带注意力机制的LSTM模型
    用于更好地捕捉长期依赖关系
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2, num_heads: int = 8, **kwargs):
        """
        初始化带注意力机制的LSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出序列长度
            dropout: Dropout率
            num_heads: 注意力头数
            **kwargs: 其他参数
        """
        self.num_heads = num_heads
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout, 
                        num_heads=num_heads, **kwargs)
    
    def _build_model(self):
        """构建带注意力机制的LSTM模型架构"""
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 全连接层
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, self.output_dim)
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim // 2)
        
        # 初始化权重
        self.initialize_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, sequence_length, input_dim)
            
        Returns:
            预测结果，形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 应用注意力机制
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        last_output = attn_output[:, -1, :]
        
        # 应用dropout
        last_output = self.dropout_layer(last_output)
        
        # 全连接层
        output = F.relu(self.fc1(last_output))
        output = self.batch_norm(output)
        output = self.dropout_layer(output)
        output = self.fc2(output)
        
        return output
    
    def initialize_weights(self):
        """重写权重初始化，包含注意力机制的初始化"""
        super().initialize_weights()
        
        # 注意力机制权重初始化
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0) 