#!/usr/bin/env python3
"""
基于RTFM框架的TabuLa-8B EC预测脚本 - 支持大模型微调版本
使用LoRA方法微调大模型以获得更好的EC预测性能
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import logging
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp

# 添加soil_vanility目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from TEC_Fusion.ECPredictorWithFinetune import ECPredictorWithFinetune

import argparse
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置环境变量以避免tokenizers警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='TabuLa-8B EC预测训练/测试脚本 - 微调版')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='train',
                       help='运行模式: train(训练), test(测试), both(训练+测试)')
    parser.add_argument('--data_path', type=str, 
                       default='/root/soil_vanility/drive-download/drive-download/4d_0m_Static.csv',
                       help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/tabula-8b',
                       help='TabuLa-8B模型路径')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='使用的GPU数量')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='指定使用的GPU ID，如 "0,1"')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小（微调时建议使用较小的batch size）')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='学习率（微调时建议使用较小的学习率）')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='采样数据大小（用于快速测试）')
    parser.add_argument('--max_train_samples', type=int, default=-1,
                       help='最大训练样本数，-1表示使用全部')
    
    # 微调相关参数
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='是否使用LoRA微调（默认启用）')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA的秩')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA的缩放系数')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA的dropout率')
    parser.add_argument('--freeze_embeddings', action='store_true', default=False,
                       help='是否冻结embedding层')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    
    # 模型保存/加载
    parser.add_argument('--save_path', type=str, 
                       default='soil_vanility/TEC_Fusion/checkpoints/ec_model_finetuned.pth',
                       help='保存训练好的模型路径')
    parser.add_argument('--load_path', type=str, default=None,
                       help='加载预训练模型路径')
    
    # 其他参数
    parser.add_argument('--use_fp32', action='store_true',
                       help='使用 float32 而不是半精度（更稳定但更慢）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def train_model_distributed(rank, world_size, args):
    """分布式训练函数"""
    setup_distributed(rank, world_size)
    
    # 设置随机种子
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    # 初始化模型
    if rank == 0:
        print(f"🚀 初始化EC预测器 (GPU {rank}/{world_size})")
    
    predictor = ECPredictorWithFinetune(
        model_path=args.model_path,
        device="auto",
        use_fp32=args.use_fp32,
        local_rank=rank,
        use_lora=args.use_lora,
        lora_config={
            'r': args.lora_rank,
            'lora_alpha': args.lora_alpha, 
            'lora_dropout': args.lora_dropout,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj']  # TabuLa使用的注意力层
        }
    )
    
    # 加载已有模型（如果提供）
    if args.load_path:
        if rank == 0:
            print(f"📂 加载预训练模型: {args.load_path}")
        predictor.load_model(args.load_path)
    
    # 准备数据
    if rank == 0:
        print(f"📊 加载数据: {args.data_path}")
    
    # 在主进程中准备数据
    if rank == 0:
        X_train, X_test, y_train, y_test, feature_names = predictor.prepare_ec_data(
            args.data_path, 
            ec_column='EC',
            sample_size=args.sample_size
        )
        
        # 从训练集中分出验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        print(f"数据分割: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
        
        # 创建临时文件保存数据供其他进程读取
        import pickle
        temp_data = {
            'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
            'y_train': y_train, 'y_test': y_test, 'y_val': y_val,
            'feature_names': feature_names
        }
        with open('/tmp/ec_data_temp.pkl', 'wb') as f:
            pickle.dump(temp_data, f)
    
    # 同步所有进程
    dist.barrier()
    
    # 所有进程读取数据
    if rank != 0:
        import pickle
        with open('/tmp/ec_data_temp.pkl', 'rb') as f:
            temp_data = pickle.load(f)
        X_train = temp_data['X_train']
        X_test = temp_data['X_test']
        X_val = temp_data['X_val']
        y_train = temp_data['y_train']
        y_test = temp_data['y_test']
        y_val = temp_data['y_val']
        feature_names = temp_data['feature_names']
    
    # 打印数据信息
    if rank == 0:
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")
        print(f"特征数量: {len(feature_names)}")
        print(f"样本数量: {args.sample_size}")
    
    # 训练模型
    if rank == 0:
        print(f"🚀 开始分布式微调训练 (使用 {world_size} 个GPU)")
    
    history = predictor.fit_with_finetune(
        X_train, y_train, 
        feature_names=feature_names,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else len(X_train),
        use_ddp=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        freeze_embeddings=args.freeze_embeddings,
        X_val=X_val,
        y_val=y_val,
        save_checkpoint_every=5,
        checkpoint_dir=os.path.dirname(args.save_path)
    )
    
    # 仅在主进程中进行评估和保存
    if rank == 0:
        print(f"\\n📊 开始评估...")
        
        # 评估模型
        predictions = predictor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"\\n✅ 分布式微调训练完成!")
        print(f"📈 性能指标:")
        print(f"  - MSE:  {mse:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²:   {r2:.4f}")
        
        # 保存模型
        predictor.save_model(args.save_path)
        print(f"✅ 模型已保存到: {args.save_path}")
        
        # 保存训练结果
        results = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "args": vars(args),
            "timestamp": datetime.now().isoformat(),
            "training_history": history,
            "best_epoch": history.get('best_epoch', 0),
            "best_val_loss": history.get('best_val_loss', float('inf'))
        }
        
        results_path = args.save_path.replace('.pth', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 训练结果已保存到: {results_path}")
        
        # 清理临时文件
        if os.path.exists('/tmp/ec_data_temp.pkl'):
            os.remove('/tmp/ec_data_temp.pkl')
    
    cleanup_distributed()

def train_model_single_gpu(args):
    """单GPU训练函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("🚀 初始化EC预测器 (单GPU模式)")
    
    # 初始化模型
    predictor = ECPredictorWithFinetune(
        model_path=args.model_path,
        device="auto",
        use_fp32=args.use_fp32,
        use_lora=args.use_lora,
        lora_config={
            'r': args.lora_rank,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        }
    )
    
    # 加载已有模型（如果提供）
    if args.load_path:
        print(f"📂 加载预训练模型: {args.load_path}")
        predictor.load_model(args.load_path)
    
    # 准备数据
    print(f"📊 加载数据: {args.data_path}")
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_ec_data(
        args.data_path, 
        ec_column='EC',
        sample_size=args.sample_size
    )
    
    # 从训练集中分出验证集
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"特征数量: {len(feature_names)}")
    
    # 训练模型
    print("🚀 开始微调训练...")
    history = predictor.fit_with_finetune(
        X_train, y_train,
        feature_names=feature_names,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples if args.max_train_samples > 0 else len(X_train),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        freeze_embeddings=args.freeze_embeddings,
        X_val=X_val,
        y_val=y_val,
        save_checkpoint_every=5,
        checkpoint_dir=os.path.dirname(args.save_path)
    )
    
    # 评估模型
    print("\\n📊 开始评估...")
    predictions = predictor.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"\\n✅ 训练完成!")
    print(f"📈 性能指标:")
    print(f"  - MSE:  {mse:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R²:   {r2:.4f}")
    
    # 保存模型
    predictor.save_model(args.save_path)
    print(f"✅ 模型已保存到: {args.save_path}")
    
    # 保存训练结果
    results = {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
        "training_history": history,
        "best_epoch": history.get('best_epoch', 0),
        "best_val_loss": history.get('best_val_loss', float('inf'))
    }
    
    results_path = args.save_path.replace('.pth', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 训练结果已保存到: {results_path}")

def main():
    args = parse_args()
    
    print("🌟 基于RTFM的TabuLa-8B EC预测 - 微调版")
    print(f"模式: {args.mode}")
    print(f"使用LoRA: {args.use_lora}")
    
    if args.mode in ['train', 'both']:
        if args.num_gpus > 1:
            # 分布式训练
            mp.spawn(
                train_model_distributed,
                args=(args.num_gpus, args),
                nprocs=args.num_gpus,
                join=True
            )
        else:
            # 单GPU训练
            train_model_single_gpu(args)
    
    if args.mode in ['test', 'both']:
        print("\\n🧪 开始测试模式...")
        # TODO: 实现纯测试模式
        pass

if __name__ == "__main__":
    main()