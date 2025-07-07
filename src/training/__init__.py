"""
训练模块包
包含通用训练器和特定训练器实现
"""

from training.base_trainer import BaseTrainer
from training.trainer import UniversalTrainer

__all__ = ['BaseTrainer', 'UniversalTrainer'] 