#!/bin/bash
# LSTM模型运行脚本

export CUDA_VISIBLE_DEVICES=1

echo "======================================"
echo "LSTM电力消费预测"
echo "======================================"

cd "$(dirname "$0")/.."

# 短期预测
echo "🚀 运行LSTM预测..."
python main.py \
    --model_type lstm \
    --prediction_type long \
    --epochs 100 \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers 2 \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --single_run \
    --verbose \
    --visualize

echo "✅ LSTM短期预测完成!" 