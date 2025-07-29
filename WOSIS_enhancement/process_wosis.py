
#!/usr/bin/env python3
"""
处理WOSIS数据 - 创建包含所有深度的大数据集
不筛选表层土壤，但保持其他处理逻辑相同
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn import model_selection
import argparse

def clean_value_column(value_str):
    """Extract numerical value from the value column"""
    if pd.isna(value_str):
        return np.nan
    # Remove curly braces and extract the number
    value_str = str(value_str).strip('{}')
    try:
        return float(value_str)
    except:
        return np.nan

def process_date_column(date_str):
    """Process date column to extract year"""
    if pd.isna(date_str):
        return 'Unknown'
    
    date_str = str(date_str)
    
    # Extract year from various date formats
    if '/' in date_str:
        # Format like "1993/11/24" or "1971-??-??"
        year = date_str.split('/')[0]
    elif '-' in date_str:
        # Format like "1962-11-1"
        year = date_str.split('-')[0]
    else:
        return 'Unknown'
    
    try:
        year_int = int(year)
        if 1900 <= year_int <= 2025:  # Reasonable year range
            return str(year_int)
        else:
            return 'Unknown'
    except:
        return 'Unknown'

def create_wosis_large(dataset_path):
    """创建包含所有深度的WOSIS大数据集"""
    print("=== WOSIS大数据集处理 ===")
    
    # 1. 读取原始数据
    raw_data_path = dataset_path
    print(f"读取原始数据: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 2. 清理value列
    df['value_clean'] = df['value'].apply(clean_value_column)
    df['date_year'] = df['date'].apply(process_date_column)
    # 用处理后的年份替换原date列
    df['date'] = df['date_year']
    
    # 3. 选择相关特征（与wosis.csv相同的列）
    selected_columns = [
        'X',                    # 经度 (数值)
        'Y',                    # 纬度 (数值)
        'upper_depth',          # 上边界深度 (数值)
        'lower_depth',          # 下边界深度 (数值)
        'value_avg',            # 平均土壤属性值 (目标变量, 数值)
        'dataset_id',           # 数据集ID (分类)
        'region',               # 地区 (分类)
        'continent',            # 大洲 (分类)
        'date'                  # 采样日期 (分类，已处理为年份)
    ]
    
    # 4. 选择列并处理缺失值
    df_selected = df[selected_columns].copy()
    
    # 去除关键列有缺失值的记录
    critical_cols = ['X', 'Y', 'upper_depth', 'lower_depth', 'value_avg']
    df_clean = df_selected.dropna(subset=critical_cols)
    print(f"去除缺失值后: {df_clean.shape}")
    
    # 5. 数据类型转换
    # 数值列转换
    numerical_cols = ['X', 'Y', 'upper_depth', 'lower_depth', 'value_avg']
    for col in numerical_cols:
        df_clean[col] = df_clean[col].astype(float)
    
    # 分类列转换
    categorical_cols = ['dataset_id', 'region', 'continent', 'date']
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str)
    
    print(f"处理后数据形状: {df_clean.shape}")
    
    # 6. 深度分布分析
    print("\n深度分布分析:")
    print(f"上边界深度范围: {df_clean['upper_depth'].min():.1f} - {df_clean['upper_depth'].max():.1f}")
    print(f"下边界深度范围: {df_clean['lower_depth'].min():.1f} - {df_clean['lower_depth'].max():.1f}")
    
    depth_ranges = df_clean.groupby(['upper_depth', 'lower_depth']).size().reset_index(name='count')
    print(f"不同深度层组合数: {len(depth_ranges)}")
    print("\n最常见的深度层:")
    print(depth_ranges.sort_values('count', ascending=False).head(10))
    
    # 7. 分割训练集和测试集
    print("\n数据集分割...")
    train_df, test_df = model_selection.train_test_split(
        df_clean, 
        test_size=0.1,  # 90% 训练，10% 测试
        random_state=42
    )
    
    print(f"训练集大小: {train_df.shape}")
    print(f"测试集大小: {test_df.shape}")
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    # 8. 保存数据
    output_dir = f'data/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, f'{dataset_name}.csv')
    test_path = os.path.join(output_dir, f'{dataset_name}_test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n数据已保存:")
    print(f"  训练集: {train_path}")
    print(f"  测试集: {test_path}")
    
    # 9. 创建info.json
    info_dir = 'data/Info'
    os.makedirs(info_dir, exist_ok=True)
    
    info = {
        "name": dataset_name,
        "task_type": "regression",
        "header": "infer",
        "column_names": [
            "X", "Y", "upper_depth", "lower_depth", "value_avg",
            "dataset_id", "region", "continent", "date"
        ],
        "num_col_idx": [0, 1, 2, 3],  # X, Y, upper_depth, lower_depth
        "cat_col_idx": [5, 6, 7, 8],  # dataset_id, region, continent, date
        "target_col_idx": [4],  # value_avg
        "file_type": "csv",
        "data_path": f"data/{dataset_name}/{dataset_name}.csv",
        "test_path": f"data/{dataset_name}/{dataset_name}_test.csv",
        "column_info": {
            "X": "float",
            "Y": "float",
            "upper_depth": "float",
            "lower_depth": "float",
            "value_avg": "float",
            "dataset_id": "str",
            "region": "str",
            "continent": "str",
            "date": "str"
        }
    }
    
    info_path = os.path.join(info_dir, f'{dataset_name}.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\n配置文件已保存: {info_path}")
    
    # 10. 统计分析
    print("\n=== 数据统计 ===")
    print("大洲分布:")
    print(train_df['continent'].value_counts())
    
    print("\n数据集分布:")
    print(train_df['dataset_id'].value_counts())
    
    print("\n土壤属性值统计:")
    print(train_df['value_avg'].describe())
    
    return train_df, test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理WOSIS大数据集')
    parser.add_argument('--run', action='store_true', help='运行数据处理')
    parser.add_argument('--dataset', default='wosis_large', help='运行数据处理')
    args = parser.parse_args()
    
    if args.run:
        create_wosis_large(args.dataset)
    else:
        print("使用 --run 参数来运行数据处理") 