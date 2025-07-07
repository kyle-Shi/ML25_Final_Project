export CUDA_VISIBLE_DEVICES=1

echo "======================================"
echo "Improved电力消费预测"
echo "======================================"

cd "$(dirname "$0")/.."

echo "🚀 运行Improved预测..."
python main.py \
    --model_type mscth \
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

echo "✅ Improved预测完成!" 