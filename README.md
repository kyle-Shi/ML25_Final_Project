# 电力消费预测项目

基于深度学习的电力消费预测系统，使用LSTM和Transformer模型预测家庭电力消费。

## 🎯 项目目标

根据历史电力消费数据和天气信息，预测未来的电力消费量：
- **短期预测**: 基于过去90天数据预测未来90天
- **长期预测**: 基于过去90天数据预测未来365天

## 📊 数据说明

### 数据特征
- `Global_active_power`: 全局有功功率(kW)
- `Global_reactive_power`: 全局无功功率(kW)
- `Voltage`: 电压(V)
- `Global_intensity`: 电流强度(A)
- `Sub_metering_1/2/3`: 分表能耗(Wh)
- `Sub_metering_remainder`: 剩余能耗(计算得出)
- `RR, NBJRR1/5/10, NBJBROU`: 天气数据

### 数据处理
- 功率数据按天求和
- 电压和强度按天求平均
- 天气数据取当天任意值
- 标准化处理

## 🏗️ 项目结构

```
src/
├── data/                   # 数据文件
├── models/                 # 模型定义
│   ├── base/              # 基础模型接口
│   ├── lstm/              # LSTM模型
│   ├── transformer/       # Transformer模型
│   └── model_factory.py   # 模型工厂
├── training/              # 训练模块
├── scripts/               # 运行脚本
├── output/                # 输出结果
├── saved_models/          # 保存的模型
├── core/                  # 核心配置
└── main.py                # 主程序
```

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- PyTorch
- CUDA (推荐)

### 2. 运行单个模型
```bash
cd src

bash scripts/run_lstm.sh
bash scripts/run_transformer.sh
bash scripts/run_improved.sh
```
