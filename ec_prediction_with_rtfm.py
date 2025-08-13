#!/usr/bin/env python3
"""
基于RTFM框架的TabuLa-8B EC预测脚本
使用现有的rtfm代码进行电导率(EC)预测
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import warnings
import logging
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

# 添加soil_vanility目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TEC_Fusion.ECPredictor import ECPredictor

import argparse
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置环境变量以避免tokenizers警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置日志级别以减少警告
logging.getLogger().setLevel(logging.ERROR)

# 添加rtfm模块到路径
sys.path.append('soil_vanility/rtfm')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TabuLa-8B EC预测训练/测试脚本')
    
    # 基本配置
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                       help='运行模式: train(训练), test(测试), both(训练+测试)')
    parser.add_argument('--data_path', type=str, 
                       default='/root/soil_vanility/drive-download/drive-download/4d_0m_Static.csv',
                       help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/tabula-8b',
                       help='TabuLa-8B模型路径')
    
    # GPU配置
    parser.add_argument('--num_gpus', type=int, default=-1,
                       help='使用的GPU数量 (-1表示使用所有可用GPU)')
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='指定使用的GPU ID，用逗号分隔 (例: 0,1,2)')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小(每个GPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--sample_size', type=int, default=5000,
                       help='使用的样本数量')
    parser.add_argument('--max_train_samples', type=int, default=-1,
                       help='最大训练样本数 (-1表示使用全部)')
    
    # 模型保存/加载
    parser.add_argument('--save_path', type=str, default='./checkpoints/ec_model.pth',
                       help='模型保存路径')
    parser.add_argument('--load_path', type=str, default=None,
                       help='预训练模型加载路径')
    
    # 其他配置
    parser.add_argument('--use_fp32', action='store_true',
                       help='使用float32精度(默认使用混合精度)')
    parser.add_argument('--fewshot_k', type=int, default=0,
                       help='Few-shot学习的样本数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def setup_ddp(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train_with_ddp(rank, world_size, args):
    """分布式训练函数"""
    # 设置分布式环境
    setup_ddp(rank, world_size)
    
    try:
        # 设置随机种子
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # 初始化预测器
        predictor = ECPredictor(
            model_path=args.model_path,
            use_fp32=args.use_fp32, 
            local_rank=rank
        )
        
        # 准备数据（只在主进程加载和处理数据）
        if rank == 0:
            print("🌟 主进程加载数据...")
            X, y, feature_names = predictor.prepare_ec_data(
                csv_path=args.data_path, 
                ec_column='EC',
                sample_size=args.sample_size
            )
            
            # 数据划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # 数据清洗
            if X_train.isnull().any().any():
                X_train = X_train.fillna(0)
                X_test = X_test.fillna(0)
            
            if y_train.isnull().any():
                mask = ~y_train.isnull()
                X_train = X_train[mask]
                y_train = y_train[mask]
                
            # 保存数据供其他进程使用
            data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'feature_names': feature_names
            }
            torch.save(data, '/tmp/ec_data.pt')
        
        # 同步所有进程
        dist.barrier()
        
        # 所有进程加载数据
        data = torch.load('/tmp/ec_data.pt')
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        feature_names = data['feature_names']
        
        if rank == 0:
            print(f"\n🚀 开始分布式训练 (使用 {world_size} 个GPU)")
        
        # 训练融合回归头
        max_train_samples = len(X_train) if args.max_train_samples == -1 else args.max_train_samples
        predictor.fit_fusion_regressor(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            ec_column='EC',
            fewshot_k=args.fewshot_k,
            max_train_samples=max_train_samples,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=0,  # 避免多进程警告
            use_ddp=True
        )
        
        # 只在主进程进行预测和评估
        if rank == 0:
            print("\n📊 开始评估...")
            max_pred = min(100, len(X_test))
            y_pred_cont = predictor.predict_ec_continuous(
                X=X_test.head(max_pred),
                feature_names=feature_names,
                ec_column='EC',
                fewshot_k=0,
                max_samples=max_pred,
            )
            y_true_cont = y_test.head(max_pred).to_numpy()
            
            mse = mean_squared_error(y_true_cont, y_pred_cont)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_cont, y_pred_cont)
            
            print(f"\n✅ 分布式训练完成!")
            print(f"📈 性能指标:")
            print(f"  - MSE:  {mse:.4f}")
            print(f"  - RMSE: {rmse:.4f}")
            print(f"  - R²:   {r2:.4f}")
            
            # 保存模型
            if args.save_path:
                predictor.save_model(args.save_path)
                
                # 保存训练结果
                results = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'args': vars(args),
                    'timestamp': datetime.now().isoformat()
                }
                results_path = args.save_path.replace('.pth', '_results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"📊 训练结果已保存到: {results_path}")
            
            # 清理临时文件
            if os.path.exists('/tmp/ec_data.pt'):
                os.remove('/tmp/ec_data.pt')
                
    except Exception as e:
        print(f"❌ Rank {rank} 错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        cleanup_ddp()

def test_model(args):
    """测试模式：加载模型并进行预测"""
    print("🧪 测试模式")
    print("=" * 40)
    
    if not args.load_path:
        raise ValueError("测试模式需要指定 --load_path 参数")
    
    # 初始化预测器
    predictor = ECPredictor(
        model_path=args.model_path,
        use_fp32=args.use_fp32
    )
    
    # 加载模型
    predictor.load_model(args.load_path)
    
    # 准备测试数据
    X, y, feature_names = predictor.prepare_ec_data(
        args.data_path,
        ec_column='EC',
        sample_size=args.sample_size
    )
    
    # 数据划分
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed
    )
    
    print(f"\n🔮 开始预测 {len(X_test)} 个测试样本...")
    
    # 预测
    y_pred = predictor.predict_ec_continuous(
        X=X_test,
        feature_names=feature_names,
        ec_column='EC',
        fewshot_k=args.fewshot_k,
        max_samples=len(X_test)
    )
    
    # 评估
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 测试结果:")
    print(f"  - MSE:  {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²:   {r2:.4f}")
    
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

def main():
    """主函数 - EC预测训练/测试"""
    args = parse_args()
    
    print("🌟 基于RTFM的TabuLa-8B EC预测")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"数据路径: {args.data_path}")
    print(f"样本数量: {args.sample_size}")
    
 
    # 确定使用的GPU数量
    available_gpus = torch.cuda.device_count()
    if args.num_gpus == -1:
        world_size = available_gpus
    else:
        world_size = min(args.num_gpus, available_gpus)
    
    print(f"可用GPU: {available_gpus}, 使用GPU: {world_size}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 根据模式执行相应操作
    if args.mode == 'test':
        test_model(args)
        return
    
    # 训练模式
    use_distributed = world_size >= 2 and args.mode in ['train', 'both']
    
    if use_distributed:
        print(f"🚀 启用分布式训练模式 ({world_size} 个GPU)")
        
        # 启动分布式训练
        mp.spawn(train_with_ddp, 
                args=(world_size, args),
                nprocs=world_size,
                join=True)
        
        print("\n🎉 分布式训练完成!")
        
        # 如果是both模式，继续测试
        if args.mode == 'both' and args.save_path:
            print("\n" + "="*60)
            args.load_path = args.save_path  # 使用刚训练的模型
            test_model(args)
        
        return
    
    # 单GPU或CPU训练
    try:
        # 1. 初始化预测器
        # 使用 float32 以避免数值稳定性问题
        predictor = ECPredictor(use_fp32=True)
        
        # 2. 准备数据
        
        try:
            X, y, feature_names = predictor.prepare_ec_data(
                args.data_path, 
                ec_column='EC',  # 根据您的数据调整列名
                sample_size=5000  # 增加样本数
            )
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
        # 3. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 检查数据是否有 NaN 值
        if X_train.isnull().any().any():
            print("⚠️ 训练数据中存在 NaN 值，进行填充...")
            X_train = X_train.fillna(0)
            X_test = X_test.fillna(0)
        
        if y_train.isnull().any():
            print("⚠️ 训练标签中存在 NaN 值，进行过滤...")
            # 过滤掉标签为 NaN 的样本
            mask = ~y_train.isnull()
            X_train = X_train[mask]
            y_train = y_train[mask]
        
        # 方案B：连续回归（使用 rtfm 序列化 + 文本/数值融合）
        print("\n🧩 连续回归（文本向量 + 数值向量 后期融合）")
        feature_names = list(X_train.columns)
        # 训练小头（冻结大模型）
        predictor.fit_fusion_regressor(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            ec_column='EC',
            fewshot_k=0,              # 如需few-shot，可改小数，例如4
            max_train_samples=len(X_train),  # 使用全部训练样本
            epochs=10,                # 增加epoch数
            lr=1e-4,
            batch_size=32,            # 增加批次大小
            num_workers=0,            # 数据加载线程数（设为0避免多进程警告）
            use_ddp=False             # 单GPU模式
        )
        # 预测
        max_pred = min(100, len(X_test))
        y_pred_cont = predictor.predict_ec_continuous(
            X=X_test.head(max_pred),
            feature_names=feature_names,
            ec_column='EC',
            fewshot_k=0,
            max_samples=max_pred,
        )
        y_true_cont = y_test.head(max_pred).to_numpy()
        mse_b = mean_squared_error(y_true_cont, y_pred_cont)
        rmse_b = np.sqrt(mse_b)
        r2_b = r2_score(y_true_cont, y_pred_cont)
        print(f"方案B -> MSE: {mse_b:.4f}  RMSE: {rmse_b:.4f}  R²: {r2_b:.4f}")
        '''
        # ===== 以下保留原有离散分类流程，便于对比 =====
        print("\n—— 以下为原有的离散分类流程（用于对比）——")
        # 4. EC离散化
        y_train_cat, bucket_info = predictor.discretize_ec(y_train, num_buckets=3)
        # 5. 执行预测
        max_test_samples = min(20, len(X_test))
        predictions = predictor.batch_predict(
            X_test.head(max_test_samples),
            X_train,
            y_train_cat,
            ec_column='EC',
            bucket_info=bucket_info,
            max_samples=max_test_samples
        )
        # 6. 转换为数值并评估
        numeric_predictions = predictor.convert_predictions_to_numeric(predictions, bucket_info)
        y_test_subset = y_test.head(max_test_samples)
        # 计算评估指标
        mse = mean_squared_error(y_test_subset, numeric_predictions)
        r2 = r2_score(y_test_subset, numeric_predictions)
        rmse = np.sqrt(mse)
        print(f"\n📈 预测性能评估 (分类→数值):")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        # 7. 显示预测示例
        print(f"\n🔍 预测示例:")
        print("-" * 80)
        for i in range(min(5, len(predictions))):
            print(f"样本 {i+1}:")
            print(f"  真实EC值:  {y_test_subset.iloc[i]:.3f}")
            print(f"  预测类别:  {predictions[i]}")
            print(f"  预测EC值:  {numeric_predictions[i]:.3f}")
            print(f"  误差:      {abs(y_test_subset.iloc[i] - numeric_predictions[i]):.3f}")
            print()
        # 8. 类别预测准确率
        y_test_cat, _ = predictor.discretize_ec(y_test_subset, num_buckets=3)
        print(f"📊 分类预测报告:")
        print(classification_report(y_test_cat, predictions, zero_division=0))
        print("🎉 EC预测演示完成!")
        '''
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    main() 