"""
Multi-Scale CNN-Transformer Hybrid (MSCTH) 模型
创新混合架构用于时间序列预测

设计特点：
1. 多尺度CNN特征提取 - 捕获不同时间尺度的局部模式
2. 自适应特征融合 - 动态学习特征重要性
3. 残差Transformer编码器 - 改进长期依赖建模
4. 渐进式预测机制 - 从短期到长期的层次化预测
5. 注意力权重可视化 - 模型可解释性

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import math

from ..base.base_model import BaseTimeSeriesModel


class MultiScaleCNNExtractor(nn.Module):
    """
    多尺度CNN特征提取器
    使用不同卷积核大小提取不同时间尺度的局部特征
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, scales: List[int] = [3, 5, 7, 9]):
        super().__init__()
        self.scales = scales
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 为每个尺度创建CNN分支
        self.scale_branches = nn.ModuleList()
        for scale in scales:
            branch = nn.Sequential(
                # 第一层卷积
                nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # 第二层卷积
                nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                
                # 第三层卷积
                nn.Conv1d(hidden_dim // 4, hidden_dim // 4, kernel_size=scale, padding=scale//2),
                nn.BatchNorm1d(hidden_dim // 4),
                nn.ReLU(),
            )
            self.scale_branches.append(branch)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            List of tensors from different scales: [(batch_size, seq_len, hidden_dim//4), ...]
        """
        # 转换为CNN格式: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        scale_features = []
        for branch in self.scale_branches:
            feature = branch(x)  # (batch_size, hidden_dim//4, seq_len)
            feature = feature.transpose(1, 2)  # (batch_size, seq_len, hidden_dim//4)
            scale_features.append(feature)
        
        return scale_features


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块
    动态学习不同尺度特征的重要性权重
    """
    
    def __init__(self, num_scales: int, feature_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        
        # 注意力权重学习网络
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换网络
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            scale_features: List of (batch_size, seq_len, feature_dim)
            
        Returns:
            fused_features: (batch_size, seq_len, feature_dim)
        """
        # 堆叠所有尺度的特征
        stacked_features = torch.stack(scale_features, dim=2)  # (batch_size, seq_len, num_scales, feature_dim)
        
        # 计算全局特征用于注意力权重计算
        global_feature = torch.mean(stacked_features, dim=(1, 2))  # (batch_size, feature_dim)
        
        # 计算注意力权重
        attention_weights = self.attention_net(global_feature)  # (batch_size, num_scales)
        attention_weights = attention_weights.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, num_scales, 1)
        
        # 加权融合
        fused_features = torch.sum(stacked_features * attention_weights, dim=2)  # (batch_size, seq_len, feature_dim)
        
        # 特征变换
        fused_features = self.feature_transform(fused_features)
        
        return fused_features


class ResidualTransformerBlock(nn.Module):
    """
    残差Transformer块
    改进的Transformer编码器，加入残差连接和层归一化
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
            
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # 第一个子层：多头自注意力 + 残差连接
        attention_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.residual_weight * attention_output)
        
        # 第二个子层：前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.residual_weight * ff_output)
        
        return x


class ProgressivePredictionHead(nn.Module):
    """
    渐进式预测头
    从短期到长期的层次化预测机制
    """
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 短期预测头 (预测前30天)
        self.short_term_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 30)
        )
        
        # 中期预测头 (预测中间30天)
        self.mid_term_head = nn.Sequential(
            nn.Linear(hidden_dim + 30, hidden_dim // 2),  # 加入短期预测结果
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 30)
        )
        
        # 长期预测头 (预测最后30天)
        self.long_term_head = nn.Sequential(
            nn.Linear(hidden_dim + 60, hidden_dim // 2),  # 加入短期+中期预测结果
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim - 60)
        )
        
        # 预测融合权重
        self.prediction_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, hidden_dim) - 全局特征表示
            
        Returns:
            predictions: (batch_size, output_dim)
        """
        # 短期预测
        short_pred = self.short_term_head(x)  # (batch_size, 30)
        
        # 中期预测 (利用短期预测结果)
        mid_input = torch.cat([x, short_pred], dim=1)
        mid_pred = self.mid_term_head(mid_input)  # (batch_size, 30)
        
        # 长期预测 (利用短期+中期预测结果)
        long_input = torch.cat([x, short_pred, mid_pred], dim=1)
        long_pred = self.long_term_head(long_input)  # (batch_size, output_dim-60)
        
        # 组合所有预测
        full_prediction = torch.cat([short_pred, mid_pred, long_pred], dim=1)
        
        return full_prediction


class MSCTHModel(BaseTimeSeriesModel):
    """
    Multi-Scale CNN-Transformer Hybrid (MSCTH) 模型
    
    创新架构结合了：
    1. 多尺度CNN特征提取
    2. 自适应特征融合
    3. 残差Transformer编码器
    4. 渐进式预测机制
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 num_heads: int = 8,
                 cnn_scales: List[int] = [3, 5, 7, 9],
                 dropout: float = 0.1,
                 **kwargs):
        """
        初始化MSCTH模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            output_dim: 输出维度
            num_heads: 多头注意力头数
            cnn_scales: CNN卷积核尺寸列表
            dropout: Dropout率
        """
        # 先设置参数
        self.num_heads = num_heads
        self.cnn_scales = cnn_scales
        
        # 然后调用父类初始化（会自动调用_build_model）
        super().__init__(input_dim, hidden_dim, num_layers, output_dim, dropout)
    
    def _build_model(self):
        """构建模型架构"""
        # 1. 多尺度CNN特征提取器
        self.cnn_extractor = MultiScaleCNNExtractor(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            scales=self.cnn_scales
        )
        
        # 2. 自适应特征融合
        cnn_feature_dim = self.hidden_dim // 4
        self.feature_fusion = AdaptiveFeatureFusion(
            num_scales=len(self.cnn_scales),
            feature_dim=cnn_feature_dim
        )
        
        # 3. 输入投影层 (将融合后的特征投影到hidden_dim)
        self.input_projection = nn.Linear(cnn_feature_dim, self.hidden_dim)
        
        # 4. 位置编码
        self.register_buffer('pos_encoding', self._create_positional_encoding(max_len=1000))
        
        # 5. 残差Transformer编码器
        self.transformer_layers = nn.ModuleList([
            ResidualTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # 6. 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 7. 渐进式预测头
        self.prediction_head = ProgressivePredictionHead(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        )
        
        # 8. 初始化权重
        self.initialize_weights()
    
    def _create_positional_encoding(self, max_len: int) -> torch.Tensor:
        """创建位置编码"""
        pe = torch.zeros(max_len, self.hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * 
                           (-math.log(10000.0) / self.hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            predictions: (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 多尺度CNN特征提取
        scale_features = self.cnn_extractor(x)
        
        # 2. 自适应特征融合
        fused_features = self.feature_fusion(scale_features)
        
        # 3. 输入投影
        x = self.input_projection(fused_features)
        
        # 4. 添加位置编码
        if hasattr(self, 'pos_encoding'):
            pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_enc
        
        # 5. 通过残差Transformer层
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
        
        # 6. 全局池化
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        # 7. 渐进式预测
        predictions = self.prediction_head(x)
        
        return predictions
    
    def initialize_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取注意力权重用于可视化
        
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            attention_weights: 包含各层注意力权重的字典
        """
        attention_weights = {}
        
        # 获取多尺度特征
        scale_features = self.cnn_extractor(x)
        fused_features = self.feature_fusion(scale_features)
        
        # 获取特征融合权重
        global_feature = torch.mean(torch.stack(scale_features, dim=2), dim=(1, 2))
        fusion_weights = self.feature_fusion.attention_net(global_feature)
        attention_weights['feature_fusion'] = fusion_weights
        
        # 获取Transformer注意力权重 (简化版本)
        x = self.input_projection(fused_features)
        if hasattr(self, 'pos_encoding'):
            pos_enc = self.pos_encoding[:, :x.size(1), :].to(x.device)
            x = x + pos_enc
        
        layer_weights = []
        for i, transformer_layer in enumerate(self.transformer_layers):
            # 简化的注意力权重获取
            attention_weights[f'transformer_layer_{i}'] = f"Layer {i} attention"
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Multi-Scale CNN-Transformer Hybrid (MSCTH)',
            'model_name': 'Multi-Scale CNN-Transformer Hybrid (MSCTH)',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设每个参数4字节
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_transformer_layers': self.num_layers,
            'num_attention_heads': self.num_heads,
            'cnn_scales': self.cnn_scales,
            'dropout': self.dropout,
            'dropout_rate': self.dropout,
            'architecture_components': [
                'Multi-Scale CNN Feature Extractor',
                'Adaptive Feature Fusion',
                'Residual Transformer Encoder',
                'Progressive Prediction Head'
            ]
        } 