"""
统一主程序
支持LSTM、Transformer等多种时间序列预测模型
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import numpy as np

from core.utils import set_seed, save_results, get_device
from core.data_processor import PowerDataProcessor
from core.constants import (
    INPUT_WINDOW, OUTPUT_WINDOW_SHORT, OUTPUT_WINDOW_LONG,
    NUM_EXPERIMENTS, RANDOM_SEED, TARGET_COLUMN
)
from models.model_factory import ModelFactory
from training.trainer import UniversalTrainer, create_data_loaders, run_experiments, summarize_experiments


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='时间序列预测模型训练')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'lstm_attention', 'transformer', 'mscth'],
                       help='模型类型')
    parser.add_argument('--prediction_type', type=str, default='short',
                       choices=['short', 'long'],
                       help='预测类型：short(90天) 或 long(365天)')
    
    # 模型超参数
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout率')
    
    # Transformer特有参数
    parser.add_argument('--nhead', type=int, default=8,
                       help='注意力头数（仅用于Transformer）')
    parser.add_argument('--dim_feedforward', type=int, default=None,
                       help='前馈网络维度（仅用于Transformer）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--patience', type=int, default=10,
                       help='早停耐心值')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='梯度裁剪值')
    
    # 数据参数
    parser.add_argument('--train_file', type=str, default='train.csv',
                       help='训练数据文件')
    parser.add_argument('--test_file', type=str, default='test.csv',
                       help='测试数据文件')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    
    # 实验参数
    parser.add_argument('--num_experiments', type=int, default=NUM_EXPERIMENTS,
                       help='实验次数')
    parser.add_argument('--single_run', action='store_true',
                       help='仅运行单次实验（不重复）')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='随机种子')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录')
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    parser.add_argument('--verbose', action='store_true',
                       help='是否显示详细的训练过程')
    
    return parser.parse_args()


def create_model(args, input_dim, output_dim):
    """创建模型"""
    model_config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
    }
    
    # 添加模型特有参数
    if args.model_type == 'transformer':
        model_config['nhead'] = args.nhead
        if args.dim_feedforward:
            model_config['dim_feedforward'] = args.dim_feedforward
    elif args.model_type == 'lstm_attention':
        model_config['num_heads'] = args.nhead
    
    model = ModelFactory.create_model(
        model_type=args.model_type,
        input_dim=input_dim,
        output_dim=output_dim,
        **model_config
    )
    
    return model


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=== 时间序列预测模型训练 ===")
    print(f"模型类型: {args.model_type}")
    print(f"预测类型: {args.prediction_type}")
    print(f"设备: {device}")
    
    # 数据处理
    print("\n=== 数据处理 ===")
    processor = PowerDataProcessor()
    
    # 确定输出窗口大小
    output_window = args.prediction_type
    
    # 处理数据
    X_train, y_train, X_test, y_test, stats = processor.process_data(
        train_file=args.train_file,
        test_file=args.test_file,
        output_window=output_window
    )
    
    print(f"训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"测试数据形状: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    # 获取数据维度
    input_dim = X_train.shape[2]
    output_dim = y_train.shape[1]
    
    print(f"输入维度: {input_dim}")
    print(f"输出维度: {output_dim}")
    
    # 创建模型并打印信息
    model = create_model(args, input_dim, output_dim)
    print(f"\n=== 模型信息 ===")
    model.print_model_info()
    
    if args.single_run:
        print(f"\n=== 单次训练 ===")
        
        # 创建训练器 - 对大模型使用更小的梯度裁剪
        total_params, _ = model.count_parameters()
        gradient_clip_val = 0.1 if total_params > 1000000 else args.gradient_clip
        trainer = UniversalTrainer(
            model=model,
            device=device,
            gradient_clip_val=gradient_clip_val
        )
        
        # 设置优化器 - 对大模型使用更小的学习率
        learning_rate = args.learning_rate * 0.1 if total_params > 1000000 else args.learning_rate
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        trainer.set_optimizer(optimizer)
        
        # 设置学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        trainer.set_scheduler(scheduler)
        
        # 训练模型
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            patience=args.patience,
            verbose=True
        )
        
        # 评估模型
        results = trainer.evaluate(test_loader, stats)
        
        print(f"\n=== 评估结果 ===")
        for key, value in results.items():
            print(f"{key}: {value:.6f}")
        
        # 可视化
        if args.visualize:
            from core.utils import plot_training_history, plot_predictions
            
            # 绘制训练历史
            plot_training_history(
                training_history['train_losses'],
                training_history['val_losses'],
                save_path=output_dir / f"{args.model_type}_training_history.png"
            )
            
            # 获取预测结果
            predictions, targets = trainer.predict(test_loader, stats)
            
            print(f"预测结果形状: predictions={predictions.shape}, targets={targets.shape}")
            
            # 只绘制第一个样本的预测结果，避免数据过多造成阴影区域
            if len(predictions.shape) > 1:
                # 多样本的情况：取第一个样本
                sample_pred = predictions[0]
                sample_target = targets[0]
            else:
                # 单维数组的情况：取前面部分数据避免过长
                sample_length = min(90, len(predictions))
                sample_pred = predictions[:sample_length]
                sample_target = targets[:sample_length]
            
            print(f"绘图数据形状: sample_pred={sample_pred.shape}, sample_target={sample_target.shape}")
            
            # 绘制预测结果
            plot_predictions(
                sample_pred,
                sample_target,
                title=f"{args.model_type.upper()} Prediction Results (Single Sample)",
                save_path=output_dir / f"{args.model_type}_predictions.png"
            )
        
        # 保存结果
        save_results(
            results,
            f"{args.model_type}_{args.prediction_type}_single",
            base_path=args.output_dir
        )
        
        # 保存模型
        if args.save_model:
            model_path = output_dir / f"{args.model_type}_{args.prediction_type}_model.pt"
            model.save_checkpoint(str(model_path))
            print(f"模型已保存至: {model_path}")
    
    else:
        print(f"\n=== 多次实验 ({args.num_experiments}次) ===")
        
        # 创建模型创建函数
        def model_creator():
            model = create_model(args, input_dim, output_dim)
            return model
        
        # 运行多次实验
        all_results = run_experiments(
            model_creator=model_creator,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            stats=stats,
            num_experiments=args.num_experiments,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            verbose=args.verbose
        )
        
        # 汇总结果
        summary = summarize_experiments(all_results)
        
        print(f"\n=== 最终结果汇总 ===")
        for key, value in summary.items():
            print(f"{key}: {value:.6f}")
        
        # 保存结果
        save_results(
            summary,
            f"{args.model_type}_{args.prediction_type}_experiments",
            base_path=args.output_dir
        )
        
        # 可视化实验结果
        if args.visualize:
            from core.utils import plot_metrics_comparison
            
            plot_metrics_comparison(
                all_results,
                save_path=output_dir / f"{args.model_type}_experiments_comparison.png"
            )
    
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main() 