import pandas as pd
import numpy as np
from typing import Tuple, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .constants import (
    POWER_FEATURES, AVERAGE_FEATURES, WEATHER_FEATURES, ALL_FEATURES,
    INPUT_WINDOW, OUTPUT_WINDOW_SHORT, OUTPUT_WINDOW_LONG
)

class PowerDataProcessor:
    """电力消费数据处理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """加载原始数据"""
        file_path = self.data_dir / filename
        print(f"正在加载数据: {file_path}")
        
        # 读取CSV文件，将"?"标记为缺失值
        data = pd.read_csv(file_path, na_values=['?', ''])
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据列名: {list(data.columns)}")
        
        # 处理DateTime列
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        
        # 将所有数值列转换为float类型
        numeric_cols = [col for col in data.columns if col != 'DateTime']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # 检查缺失值
        print(f"缺失值统计:\n{data.isnull().sum()}")
        
        # 填充缺失值
        for col in numeric_cols:
            if data[col].isnull().any():
                # 用前一个有效值填充
                data[col] = data[col].ffill()
                # 如果仍有缺失值，用后一个有效值填充
                data[col] = data[col].bfill()
                # 如果仍有缺失值，用均值填充
                data[col] = data[col].fillna(data[col].mean())
        
        print(f"填充后缺失值统计:\n{data.isnull().sum()}")
        
        return data
    
    def calculate_sub_metering_remainder_daily(self, data: pd.DataFrame) -> pd.DataFrame:
        """在按天聚合后计算sub_metering_remainder"""
        # 注意：此时data已经是按天聚合的数据
        # Global_active_power: 每天的总功率消耗 (kW*天)
        # Sub_metering_1/2/3: 每天的总能耗 (Wh*天)
        # 计算公式: sub_metering_remainder = (global_active_power * 1000) - (sub_metering_1 + sub_metering_2 + sub_metering_3)
        data = data.copy()
        
        # 将每天的功率转换为每天的总能耗 (kW*天 -> Wh*天)
        # 1kW = 1000W, 1天 = 24*60 = 1440分钟
        # 但这里数据已经是按天聚合的，所以直接转换单位
        data['Sub_metering_remainder'] = (
            data['Global_active_power'] * 1000  # kW -> W，然后乘以天数(已聚合)
        ) - (
            data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3']
        )
        
        print(f"已计算Sub_metering_remainder(按天聚合后)，统计信息:")
        print(f"均值: {data['Sub_metering_remainder'].mean():.3f}")
        print(f"标准差: {data['Sub_metering_remainder'].std():.3f}")
        print(f"最小值: {data['Sub_metering_remainder'].min():.3f}")
        print(f"最大值: {data['Sub_metering_remainder'].max():.3f}")
        
        return data
    
    def aggregate_daily_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """按天聚合数据"""
        # 提取日期
        data['Date'] = data['DateTime'].dt.date
        
        # 创建聚合字典
        agg_dict = {}
        
        # 功率数据按天取总和
        for col in POWER_FEATURES:
            if col in data.columns:
                agg_dict[col] = 'sum'
        
        # 电压和强度按天取平均
        for col in AVERAGE_FEATURES:
            if col in data.columns:
                agg_dict[col] = 'mean'
        
        # 降雨量数据取当天第一个值
        for col in WEATHER_FEATURES:
            if col in data.columns:
                agg_dict[col] = 'first'
        
        # 按日期聚合
        daily_data = data.groupby('Date').agg(agg_dict).reset_index()
        
        # 转换Date为datetime
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        
        print(f"聚合后数据形状: {daily_data.shape}")
        print(f"聚合后数据日期范围: {daily_data['Date'].min()} 到 {daily_data['Date'].max()}")
        
        return daily_data
    
    def create_sequences(self, data: pd.DataFrame, target_col: str, 
                        input_window: int, output_window: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建用于时间序列模型的序列数据"""
        # 选择特征列（使用所有可用特征）
        feature_cols = []
        for col in ALL_FEATURES:
            if col in data.columns:
                feature_cols.append(col)
        
        # 准备数据
        features = data[feature_cols].values
        target = data[target_col].values
        
        X, y = [], []
        
        # 创建序列
        for i in range(len(data) - input_window - output_window + 1):
            # 输入序列
            X.append(features[i:i+input_window])
            # 目标序列
            y.append(target[i+input_window:i+input_window+output_window])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"序列数据形状: X={X.shape}, y={y.shape}")
        print(f"特征列: {feature_cols}")
        
        return X, y
    
    def normalize_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """标准化数据"""
        # 选择需要标准化的列（使用所有可用特征）
        normalize_cols = []
        for col in ALL_FEATURES:
            if col in train_data.columns:
                normalize_cols.append(col)
        
        # 计算训练数据的统计信息
        stats = {}
        for col in normalize_cols:
            stats[col] = {
                'mean': train_data[col].mean(),
                'std': train_data[col].std()
            }
        
        # 标准化训练数据
        train_normalized = train_data.copy()
        for col in normalize_cols:
            train_normalized[col] = (train_data[col] - stats[col]['mean']) / stats[col]['std']
        
        # 标准化测试数据
        test_normalized = test_data.copy()
        for col in normalize_cols:
            test_normalized[col] = (test_data[col] - stats[col]['mean']) / stats[col]['std']
        
        print(f"已标准化 {len(normalize_cols)} 个特征列")
        
        return train_normalized, test_normalized, stats
    
    def inverse_normalize_target(self, normalized_values: np.ndarray, 
                               target_col: str, stats: dict) -> np.ndarray:
        """反标准化目标值"""
        if target_col in stats:
            return normalized_values * stats[target_col]['std'] + stats[target_col]['mean']
        return normalized_values
    
    def process_data(self, train_file: str = "train.csv", 
                    test_file: str = "test.csv", 
                    output_window: str = "short") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """完整的数据处理流程"""
        print("=== 开始数据处理 ===")
        
        # 步骤1：加载数据
        train_data = self.load_data(train_file)
        test_data = self.load_data(test_file)
        
        # 步骤2：按天聚合（先聚合，后计算衍生特征）
        train_daily = self.aggregate_daily_data(train_data)
        test_daily = self.aggregate_daily_data(test_data)
        
        # 步骤3：计算sub_metering_remainder（在聚合后计算）
        train_daily = self.calculate_sub_metering_remainder_daily(train_daily)
        test_daily = self.calculate_sub_metering_remainder_daily(test_daily)
        
        # 步骤4：标准化
        train_normalized, test_normalized, stats = self.normalize_data(train_daily, test_daily)
        
        # 步骤5：创建序列
        window_size = OUTPUT_WINDOW_SHORT if output_window == "short" else OUTPUT_WINDOW_LONG
        X_train, y_train = self.create_sequences(
            train_normalized, 'Global_active_power', INPUT_WINDOW, window_size
        )
        X_test, y_test = self.create_sequences(
            test_normalized, 'Global_active_power', INPUT_WINDOW, window_size
        )
        
        print("=== 数据处理完成 ===")
        
        return X_train, y_train, X_test, y_test, stats 