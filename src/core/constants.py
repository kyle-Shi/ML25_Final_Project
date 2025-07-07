# 项目常量定义
from typing import Literal

# 输入窗口大小（过去90天用于预测）
INPUT_WINDOW = 90

# 输出窗口大小
OUTPUT_WINDOW_SHORT = 90  # 短期预测（90天）
OUTPUT_WINDOW_LONG = 365  # 长期预测（365天）

# 实验轮数
NUM_EXPERIMENTS = 5

# 数据特征列定义
# 功率特征（按天求和）
POWER_FEATURES = [
    "Global_active_power",
    "Global_reactive_power", 
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
    # 注意：Sub_metering_remainder 不在这里，因为它需要在聚合后重新计算
]

# 所有特征列（包括计算得出的特征）
ALL_FEATURES = [
    "Global_active_power",
    "Global_reactive_power", 
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Sub_metering_remainder",
    "Voltage",
    "Global_intensity",
    "RR",
    "NBJRR1", 
    "NBJRR5",
    "NBJRR10",
    "NBJBROU"
]

AVERAGE_FEATURES = [
    "Voltage",
    "Global_intensity"
]

WEATHER_FEATURES = [
    "RR",
    "NBJRR1", 
    "NBJRR5",
    "NBJRR10",
    "NBJBROU"
]

# 预测目标
TARGET_COLUMN = "Global_active_power"

# 随机种子
RANDOM_SEED = 42 