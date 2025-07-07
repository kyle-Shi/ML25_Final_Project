"""
Transformer模型实现
用于时间序列预测的专门Transformer架构
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.base.base_model import BaseTimeSeriesModel


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为序列中的每个位置添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为buffer，不会被训练但会保存在state_dict中
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 (seq_len, batch_size, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer基础块
    包含多头自注意力和前馈网络
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, activation: str = "relu"):
        """
        初始化Transformer块
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            activation: 激活函数类型
        """
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = getattr(F, activation)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 输入张量 (batch_size, seq_len, d_model)
            src_mask: 注意力掩码
            
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        # 多头自注意力
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class PowerTransformer(BaseTimeSeriesModel):
    """
    用于电力消费预测的Transformer模型
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 nhead: int = 8,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 **kwargs):
        """
        初始化PowerTransformer模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 模型隐藏维度（d_model）
            num_layers: Transformer层数
            output_dim: 输出维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            max_seq_length: 最大序列长度
            **kwargs: 其他参数
        """
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward or hidden_dim * 4
        self.max_seq_length = max_seq_length
        
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout,
                        nhead=nhead, dim_feedforward=self.dim_feedforward, 
                        max_seq_length=max_seq_length, **kwargs)
    
    def _build_model(self):
        """构建Transformer模型架构"""
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=self.hidden_dim,
            max_len=self.max_seq_length,
            dropout=self.dropout
        )
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # 简化稳定设计：模仿LSTM架构
        # 1. 全局平均池化（最稳定的聚合方式）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. 简单FC层（类似LSTM的输出层）
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, self.output_dim)
        
        # 3. 批标准化和Dropout
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim // 2)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 初始化权重
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            预测结果 (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # 转换为 (seq_len, batch_size, hidden_dim) 以适应位置编码
        x = x.transpose(0, 1)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 转换回 (batch_size, seq_len, hidden_dim)
        x = x.transpose(0, 1)
        
        # 通过Transformer块进行编码
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # 简化稳定聚合（完全模仿LSTM）
        # x: (batch_size, seq_len, hidden_dim)
        
        # 全局平均池化（最稳定的聚合方式）
        pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 应用dropout
        pooled = self.dropout_layer(pooled)
        
        # 第一个FC层
        output = torch.relu(self.fc1(pooled))
        output = self.batch_norm(output)
        output = self.dropout_layer(output)
        
        # 第二个FC层（输出层）
        output = self.fc2(output)
        
        return output
    
    def initialize_weights(self):
        """初始化模型权重"""
        super().initialize_weights()
        
        # 特殊初始化各种层
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                # 注意力权重初始化
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.BatchNorm1d):
                # 批标准化层初始化
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # 线性层使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 特殊初始化FC层
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
    
    def create_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        创建填充掩码
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            lengths: 每个序列的实际长度 (batch_size,)
            
        Returns:
            掩码张量 (batch_size, seq_len)
        """
        batch_size, max_len = x.size(0), x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(
            batch_size, max_len
        ) >= lengths.unsqueeze(1)
        return mask
    
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建因果掩码（下三角掩码）
        
        Args:
            seq_len: 序列长度
            
        Returns:
            因果掩码 (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def get_attention_weights(self, x: torch.Tensor) -> dict:
        """
        获取注意力权重（用于可视化）
        
        Args:
            x: 输入张量 (batch_size, seq_len, input_dim)
            
        Returns:
            注意力权重字典，包含各层权重和全局注意力权重
        """
        attention_weights = {}
        
        # 输入投影
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # 获取每个transformer块的注意力权重（简化版）
        for i, transformer_block in enumerate(self.transformer_blocks):
            x = transformer_block(x)
            # 注意：实际获取权重需要修改transformer_block，这里简化处理
            
        # 新的稳定化设计不需要全局注意力权重
        # 返回基本信息
        attention_weights['info'] = "稳定化Transformer - 使用平均池化和最后时间步融合"
        
        return attention_weights 