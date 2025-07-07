import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def set_seed(seed: int) -> None:
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_model(model: torch.nn.Module, name: str, base_path: str = "models") -> None:
    """保存模型到指定路径"""
    models_dir = Path(base_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), models_dir / filename)
    print(f"模型已保存至: {models_dir / filename}")

def load_model(model: torch.nn.Module, filename: str, base_path: str = "models") -> torch.nn.Module:
    """从指定路径加载模型"""
    models_dir = Path(base_path)
    model.load_state_dict(torch.load(models_dir / filename))
    return model

def save_results(results: dict, name: str, base_path: str = "output") -> None:
    """保存实验结果"""
    output_dir = Path(base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_dir / filename, 'w', encoding='utf-8') as f:
        f.write("=== 实验结果 ===\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"结果已保存至: {output_dir / filename}")

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """计算MSE和MAE指标"""
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    return mse, mae

def plot_predictions(predictions: np.ndarray, targets: np.ndarray, 
                    title: str = "Prediction Results", save_path: str = None) -> None:
    """绘制预测结果对比图"""
    
    # 确保数据是一维的
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()
    if len(targets.shape) > 1:
        targets = targets.flatten()
    
    # 限制数据长度，避免图表过于密集
    max_points = 1000
    if len(predictions) > max_points:
        step = len(predictions) // max_points
        predictions = predictions[::step]
        targets = targets[::step]
    
    print(f"绘制数据点数: {len(predictions)}")
    
    plt.figure(figsize=(15, 8))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 时间序列对比图
    ax1.plot(targets, label='Ground Truth', alpha=0.8, linewidth=2, color='blue')
    ax1.plot(predictions, label='Prediction', alpha=0.8, linewidth=2, color='red')
    ax1.set_title('Time Series Comparison')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Global Active Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图（真实值 vs 预测值）
    ax2.scatter(targets, predictions, alpha=0.6, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Ground Truth (kW)')
    ax2.set_ylabel('Prediction (kW)')
    ax2.set_title('Scatter Plot: Truth vs Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 误差分析图
    errors = predictions - targets
    ax3.plot(errors, color='purple', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.fill_between(range(len(errors)), errors, alpha=0.3, color='purple')
    ax3.set_title('Prediction Error Analysis')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error (Prediction - Truth)')
    ax3.grid(True, alpha=0.3)
    
    # 4. 误差分布直方图
    ax4.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=errors.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
    ax4.set_title('Error Distribution')
    ax4.set_xlabel('Error (kW)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def plot_multiple_predictions(all_predictions: List[np.ndarray], all_targets: List[np.ndarray], 
                            titles: List[str] = None, save_path: str = None) -> None:
    """绘制多个预测结果对比图"""
    n_samples = len(all_predictions)
    if n_samples == 0:
        return
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(15, 4*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, (predictions, targets) in enumerate(zip(all_predictions, all_targets)):
        ax = axes[i]
        ax.plot(targets, label='Ground Truth', alpha=0.8, linewidth=2, color='blue')
        ax.plot(predictions, label='Prediction', alpha=0.8, linewidth=2, color='red')
        
        title = titles[i] if titles and i < len(titles) else f'Sample {i+1}'
        ax.set_title(title)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Global Active Power (kW)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 计算并显示指标
        mse, mae = calculate_metrics(predictions, targets)
        ax.text(0.02, 0.98, f'MSE: {mse:.2f}\nMAE: {mae:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Multiple Predictions Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"多样本对比图已保存至: {save_path}")
    
    plt.show()

def plot_training_history(train_losses: List[float], val_losses: List[float] = None, 
                         save_path: str = None) -> None:
    """绘制训练历史图"""
    plt.figure(figsize=(12, 5))
    
    if val_losses:
        plt.subplot(1, 2, 1)
    
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if val_losses:
        # 添加学习曲线分析
        plt.subplot(1, 2, 2)
        plt.plot(np.log(train_losses), label='Log Training Loss', color='blue', linewidth=2)
        plt.plot(np.log(val_losses), label='Log Validation Loss', color='red', linewidth=2)
        plt.title('Training History (Log Scale)')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()

def plot_metrics_comparison(experiment_results: dict, save_path: str = None) -> None:
    """绘制多次实验指标对比图"""
    metrics = ['MSE_mean', 'MAE_mean', 'Overall_MSE', 'Overall_MAE']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if metric in experiment_results:
            values = experiment_results[metric]
            ax = axes[i]
            
            # 绘制折线图
            ax.plot(range(1, len(values)+1), values, 'o-', linewidth=2, markersize=8)
            ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(values):.2f}')
            ax.fill_between(range(1, len(values)+1), 
                           np.mean(values) - np.std(values),
                           np.mean(values) + np.std(values),
                           alpha=0.2, color='red',
                           label=f'±1 Std: {np.std(values):.2f}')
            
            ax.set_title(f'{metric} Across Experiments')
            ax.set_xlabel('Experiment Number')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for j, v in enumerate(values):
                ax.annotate(f'{v:.1f}', (j+1, v), textcoords="offset points", 
                           xytext=(0,10), ha='center')
    
    plt.suptitle('Experimental Results Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"指标对比图已保存至: {save_path}")
    
    plt.show()

def get_device() -> torch.device:
    """获取可用设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device 