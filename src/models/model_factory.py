"""
模型工厂
统一创建和管理不同类型的时间序列预测模型
"""

from typing import Dict, Any, Type
import torch.nn as nn

from models.base.base_model import BaseTimeSeriesModel
from models.lstm.lstm_model import PowerLSTM, PowerLSTMWithAttention
from models.transformer.transformer_model import PowerTransformer
from models.improved.improved_model import MSCTHModel


class ModelFactory:
    """
    模型工厂类
    负责创建和管理不同类型的模型
    """
    
    # 注册所有可用的模型
    _models: Dict[str, Type[BaseTimeSeriesModel]] = {
        'lstm': PowerLSTM,
        'lstm_attention': PowerLSTMWithAttention,
        'transformer': PowerTransformer,
        'mscth': MSCTHModel
    }
    
    @classmethod
    def create_model(cls, 
                    model_type: str,
                    input_dim: int,
                    hidden_dim: int,
                    num_layers: int,
                    output_dim: int,
                    **kwargs) -> BaseTimeSeriesModel:
        """
        创建指定类型的模型
        
        Args:
            model_type: 模型类型
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            output_dim: 输出维度
            **kwargs: 其他模型特定参数
            
        Returns:
            初始化的模型实例
            
        Raises:
            ValueError: 如果模型类型不支持
        """
        if model_type not in cls._models:
            available_models = ', '.join(cls._models.keys())
            raise ValueError(f"不支持的模型类型: {model_type}. "
                           f"可用模型: {available_models}")
        
        model_class = cls._models[model_type]
        
        # 创建模型实例
        model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            **kwargs
        )
        
        return model
    
    @classmethod
    def get_available_models(cls) -> list:
        """
        获取所有可用的模型类型
        
        Returns:
            可用模型类型列表
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseTimeSeriesModel]):
        """
        注册新的模型类型
        
        Args:
            name: 模型名称
            model_class: 模型类
        """
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError("模型类必须继承BaseTimeSeriesModel")
        
        cls._models[name] = model_class
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """
        获取模型类型信息
        
        Args:
            model_type: 模型类型
            
        Returns:
            模型类型信息字典
        """
        if model_type not in cls._models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model_class = cls._models[model_type]
        
        return {
            'name': model_type,
            'class_name': model_class.__name__,
            'description': model_class.__doc__,
        }
    
    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """
        获取模型的默认配置
        
        Args:
            model_type: 模型类型
            
        Returns:
            默认配置字典
        """
        configs = {
            'lstm': {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2
            },
            'lstm_attention': {
                'hidden_dim': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'num_heads': 8
            },
            'transformer': {
                'hidden_dim': 256,
                'num_layers': 6,
                'dropout': 0.1,
                'nhead': 8,
                'dim_feedforward': 1024,
                'max_seq_length': 1000
            },
            'mscth': {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.1,
                'num_heads': 8,
                'cnn_scales': [3, 5, 7, 9]
            }
        }
        
        if model_type not in configs:
            return {}
        
        return configs[model_type]
    
    @classmethod
    def create_model_with_config(cls, 
                                model_type: str,
                                input_dim: int,
                                output_dim: int,
                                config: Dict[str, Any] = None) -> BaseTimeSeriesModel:
        """
        使用配置创建模型
        
        Args:
            model_type: 模型类型
            input_dim: 输入维度
            output_dim: 输出维度
            config: 自定义配置（可选）
            
        Returns:
            初始化的模型实例
        """
        # 获取默认配置
        default_config = cls.get_default_config(model_type)
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        # 创建模型
        return cls.create_model(
            model_type=model_type,
            input_dim=input_dim,
            output_dim=output_dim,
            **default_config
        )
    
    @classmethod
    def print_all_models(cls):
        """打印所有可用模型的信息"""
        print("=== 可用模型列表 ===")
        for model_type in cls._models:
            info = cls.get_model_info(model_type)
            config = cls.get_default_config(model_type)
            print(f"\n模型类型: {model_type}")
            print(f"类名: {info['class_name']}")
            print(f"描述: {info.get('description', '无描述')}")
            print(f"默认配置: {config}")


# 便捷函数
def create_lstm(input_dim: int, output_dim: int, **kwargs) -> PowerLSTM:
    """创建LSTM模型的便捷函数"""
    return ModelFactory.create_model('lstm', input_dim=input_dim, output_dim=output_dim, **kwargs)


def create_lstm_attention(input_dim: int, output_dim: int, **kwargs) -> PowerLSTMWithAttention:
    """创建带注意力的LSTM模型的便捷函数"""
    return ModelFactory.create_model('lstm_attention', input_dim=input_dim, output_dim=output_dim, **kwargs)


def create_transformer(input_dim: int, output_dim: int, **kwargs) -> PowerTransformer:
    """创建Transformer模型的便捷函数"""
    return ModelFactory.create_model('transformer', input_dim=input_dim, output_dim=output_dim, **kwargs)


def create_mscth(input_dim: int, output_dim: int, **kwargs) -> MSCTHModel:
    """创建MSCTH改进模型的便捷函数"""
    return ModelFactory.create_model('mscth', input_dim=input_dim, output_dim=output_dim, **kwargs) 