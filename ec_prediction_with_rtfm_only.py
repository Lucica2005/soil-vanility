#!/usr/bin/env python3
"""
仅使用 RTFM 端到端（rtfm进 → rtfm出）的 TabuLa-8B EC 预测脚本备份
- 保留原有：序列化 + few-shot 组装 + 生成式分类 + 类别映射为数值
- 不包含：方案B（文本向量 + 数值向量 连续回归）的任何代码
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# 添加rtfm模块到路径
sys.path.append('soil_vanility/rtfm')

# 导入 rtfm 模块
from rtfm.inference_utils import infer_on_example
from rtfm.serialization.serializers import BasicSerializerV2
from rtfm.task_config import TLMConfig
from rtfm.arguments import DataArguments


class ECPredictor:
    """基于TabuLa-8B的EC预测器（RTFM 端到端版本）"""

    def __init__(self, model_path="/root/autodl-tmp/tabula-8b", device="auto"):
        self.device = self._setup_device(device)
        self.model_path = model_path

        print(f"🚀 初始化TabuLa-8B EC预测器 (RTFM 端到端)...")
        print(f"模型路径: {model_path}")
        print(f"计算设备: {self.device}")

        # 加载模型与分词器
        self._load_model()

        # 初始化序列化器
        from rtfm.configs import SerializerConfig
        serializer_config = SerializerConfig()
        self.serializer = BasicSerializerV2(config=serializer_config)

        print("✅ EC预测器初始化完成!")

    def _setup_device(self, device):
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        """加载模型和分词器"""
        try:
            if os.path.exists(self.model_path):
                print(f"📁 从本地加载模型: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )
            else:
                print(f"🌐 从HuggingFace加载模型...")
                model_name = "mlfoundations/tabula-8b"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("🔧 已设置pad_token以避免警告")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 提示: 请确保模型文件存在或网络连接正常")
            raise

    def prepare_ec_data(self, csv_path, ec_column='EC', sample_size=None):
        print(f"📂 加载EC数据: {csv_path}")
        df = pd.read_csv(csv_path)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 随机采样 {sample_size} 个样本")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        if ec_column not in df.columns:
            print(f"⚠️ 未找到EC列 '{ec_column}'，尝试查找相似列名...")
            possible_ec_cols = [col for col in df.columns if 'ec' in col.lower()]
            if possible_ec_cols:
                ec_column = possible_ec_cols[0]
                print(f"✅ 使用列: {ec_column}")
            else:
                raise ValueError(f"未找到EC相关列，可用列: {list(df.columns)}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != ec_column]

        print(f"🎯 目标变量: {ec_column}")
        print(f"📊 特征变量 ({len(feature_cols)}个): {feature_cols}")

        X = df[feature_cols].copy()
        y = df[ec_column].copy()

        X = X.fillna(X.median())
        y = y.fillna(y.median())

        valid_mask = np.isfinite(y) & (y > 0)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"📈 有效数据: {len(X)} 样本")
        print(f"EC统计: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
        return X, y, feature_cols

    def discretize_ec(self, y, num_buckets=4):
        print(f"🔢 将EC离散化为 {num_buckets} 个类别...")
        quantiles = []
        for i in range(1, num_buckets):
            q = np.quantile(y, i / num_buckets)
            quantiles.append(q)

        def get_category(value):
            for i, threshold in enumerate(quantiles):
                if value <= threshold:
                    if i == 0:
                        return f"low (≤{threshold:.2f})"
                    else:
                        return f"medium-{i} ({quantiles[i-1]:.2f}-{threshold:.2f}]"
            return f"high (>{quantiles[-1]:.2f})"

        categories = y.apply(get_category)

        all_categories = []
        all_categories.append(f"low (≤{quantiles[0]:.2f})")
        for i in range(1, len(quantiles)):
            all_categories.append(f"medium-{i} ({quantiles[i-1]:.2f}-{quantiles[i]:.2f}]")
        all_categories.append(f"high (>{quantiles[-1]:.2f})")

        bucket_info = {
            'quantiles': quantiles,
            'categories': all_categories,
            'num_buckets': num_buckets,
        }

        print(f"📊 分类统计:")
        print(categories.value_counts())
        return categories, bucket_info

    def predict_ec_sample(self, target_sample, training_data, ec_column, bucket_info, num_shots=16):
        few_shot_examples = training_data.sample(n=min(num_shots, len(training_data)), random_state=42)
        cfg = TLMConfig(
            prefix=f"Predict the {ec_column} category",
            suffix=f"What is the {ec_column} category?",
            label_values=bucket_info['categories'],
        )
        try:
            prediction = infer_on_example(
                model=self.model,
                tokenizer=self.tokenizer,
                serializer=self.serializer,
                target_example=target_sample,
                target_colname=ec_column,
                target_choices=bucket_info['categories'],
                labeled_examples=few_shot_examples,
                cfg=cfg,
                handle_invalid_predictions="warn",
            )
            return prediction
        except Exception as e:
            print(f"⚠️ 预测失败: {e}")
            return bucket_info['categories'][0]

    def batch_predict(self, X_test, X_train, y_train_cat, ec_column, bucket_info, max_samples=100):
        train_data = X_train.copy()
        train_data[ec_column] = y_train_cat
        if len(X_test) > max_samples:
            X_test = X_test.head(max_samples)
            print(f"📊 限制预测样本数为: {max_samples}")
        predictions = []
        total = len(X_test)
        print(f"🔮 开始批量预测 {total} 个样本...")
        for idx, (_, row) in enumerate(X_test.iterrows()):
            if idx % 10 == 0:
                print(f"进度: {idx+1}/{total}")
            target_sample = pd.DataFrame([row])
            pred = self.predict_ec_sample(target_sample, train_data, ec_column, bucket_info)
            predictions.append(pred)
        print("✅ 批量预测完成!")
        return predictions

    def convert_predictions_to_numeric(self, predictions, bucket_info):
        """将分类预测转换为数值EC值（按区间中点/倍数等规则）。"""
        numeric_values = []
        quantiles = bucket_info['quantiles']
        for pred in predictions:
            if isinstance(pred, str) and "low" in pred:
                value = quantiles[0] / 2
            elif isinstance(pred, str) and "high" in pred:
                value = quantiles[-1] * 1.5
            elif isinstance(pred, str) and "medium" in pred:
                try:
                    import re
                    m = re.search(r"\(([-+]?\d*\.?\d+)-([-+]?\d*\.?\d+)\]", pred)
                    if m:
                        lower = float(m.group(1))
                        upper = float(m.group(2))
                        value = (lower + upper) / 2.0
                    else:
                        value = float(np.mean(quantiles))
                except Exception:
                    value = float(np.mean(quantiles))
            else:
                value = float(np.mean(quantiles))
            numeric_values.append(value)
        return np.array(numeric_values)


def main():
    print("🌟 基于RTFM的TabuLa-8B EC预测（端到端备份版）")
    print("=" * 60)

    try:
        predictor = ECPredictor()

        # 数据路径（可按需修改）
        csv_path = "/root/soil_vanility/drive-download/drive-download/4d_0m_Static.csv"
        try:
            X, y, feature_names = predictor.prepare_ec_data(
                csv_path,
                ec_column='EC',
                sample_size=1000,
            )
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("📝 创建模拟数据进行演示...")
            np.random.seed(42)
            n_samples = 500
            data = {
                'temperature': np.random.normal(15, 5, n_samples),
                'precipitation': np.random.normal(300, 100, n_samples),
                'clay_content': np.random.uniform(5, 50, n_samples),
                'ph_value': np.random.normal(7, 1, n_samples),
                'sand_content': np.random.uniform(20, 80, n_samples),
                'elevation': np.random.uniform(50, 1500, n_samples),
            }
            X = pd.DataFrame(data)
            y = pd.Series(
                0.01 * X['clay_content'] +
                0.005 * X['temperature'] +
                0.001 * X['precipitation'] +
                0.1 * np.abs(X['ph_value'] - 7) +
                np.random.normal(0, 0.5, n_samples)
            )
            y = np.abs(y) + 0.1
            feature_names = list(X.columns)

        # 切分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 离散化训练标签
        y_train_cat, bucket_info = predictor.discretize_ec(y_train, num_buckets=3)

        # 批量预测（生成式分类）
        max_test_samples = min(20, len(X_test))
        predictions = predictor.batch_predict(
            X_test.head(max_test_samples),
            X_train,
            y_train_cat,
            ec_column='EC',
            bucket_info=bucket_info,
            max_samples=max_test_samples,
        )

        # 转数值并评估
        numeric_predictions = predictor.convert_predictions_to_numeric(predictions, bucket_info)
        y_test_subset = y_test.head(max_test_samples)
        mse = mean_squared_error(y_test_subset, numeric_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_subset, numeric_predictions)
        print(f"\n📈 预测性能评估 (分类→数值):")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        # 分类报告
        y_test_cat, _ = predictor.discretize_ec(y_test_subset, num_buckets=3)
        print("\n📊 分类预测报告:")
        print(classification_report(y_test_cat, predictions, zero_division=0))
        print("🎉 运行完成!")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 