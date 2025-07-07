#!/bin/bash
# Transformer模型运行脚本

export CUDA_VISIBLE_DEVICES=1

echo "======================================"
echo "Transformer电力消费预测"
echo "======================================"

cd "$(dirname "$0")/.."

# 短期预测
echo "🚀 运行Transformer预测..."
python main.py \
    --model_type transformer \
    --prediction_type long \
    --epoch 30 \
    --batch_size 32 \
    --hidden_dim 512 \
    --num_layers 4 \
    --learning_rate 0.001 \
    --dropout 0.1 \
    --single_run \
    --verbose \
    --visualize

echo "✅ Transformer预测完成!" 