import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import logging
from tqdm import tqdm
import torch.distributed as dist
# 添加当前目录到Python路径，以便导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ECFusion import MeanPooler, NumericalEncoder
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from rtfm.arguments import DataArguments
from rtfm.inference_utils import prepare_dataframe
from rtfm.configs import SerializerConfig

from datetime import datetime
warnings.filterwarnings('ignore')

# 设置环境变量以避免tokenizers警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 设置日志级别以减少警告
logging.getLogger().setLevel(logging.ERROR)

# 添加rtfm模块到路径
sys.path.append('soil_vanility/rtfm')
from TEC_Fusion.rtfm.inference_utils import infer_on_example
# 导入rtfm模块
from rtfm.inference_utils import infer_on_example
from rtfm.serialization.serializers import BasicSerializerV2
from rtfm.task_config import TLMConfig
from rtfm.arguments import DataArguments
import torch.nn as nn
from rtfm.data import (
    serialize_dataset_fn,
    add_qa_and_eoc_tokens_to_example,
    tokenize_ds_dict,
    build_formatted_df,
    prepare_hf_dataset_from_formatted_df,
    make_few_shot_sample,
    DataCollatorForSupervisedDataset,
)
from TEC_Fusion.ECDataset import ECDataset
def ec_collate_fn(batch):
    """自定义collate函数，将批次数据转换为DataFrame和目标列表"""
    rows = []
    targets = []
    
    for row_dict, target in batch:
        rows.append(row_dict)
        targets.append(target)
    
    # 将字典列表转换为DataFrame
    rows_df = pd.DataFrame(rows)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return rows_df, targets_tensor
class ECPredictor:
    """基于TabuLa-8B的EC预测器"""
    
    def __init__(self, model_path="/root/autodl-tmp/tabula-8b", device="auto", use_fp32=False, local_rank=-1):
        """
        初始化EC预测器
        
        Args:
            model_path: 模型路径 (本地或HuggingFace模型名)
            device: 计算设备
            use_fp32: 是否使用 float32 精度（更稳定但更慢）
            local_rank: 分布式训练的本地rank
        """
        self.local_rank = local_rank
        self.device = self._setup_device(device)
        self.model_path = model_path
        # 选择数据类型
        if use_fp32:
            self.dtype = torch.float32
        elif self.device == "cuda":
            # 优先使用 bfloat16，如果不支持则使用 float16
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        print(f"🚀 初始化TabuLa-8B EC预测器...")
        print(f"模型路径: {model_path}")
        print(f"计算设备: {self.device}")
        print(f"数据类型: {self.dtype}")
        
        # ===== 方案B：连续回归部件（文本向量 + 数值向量 后期融合）=====
        self.text_pool = MeanPooler()
        self.text_proj = None  # 延迟初始化：等加载模型后知道hidden_size再建
        self.num_encoder = None
        self.fusion_head = None
        
        # 加载模型和分词器
        self._load_model()
        
        # 初始化序列化器
        
        serializer_config = SerializerConfig()
        self.serializer = BasicSerializerV2(config=serializer_config)
        
        print("✅ EC预测器初始化完成!")
    
    def _setup_device(self, device):
        """设置计算设备"""
        if self.local_rank != -1:
            # 分布式训练模式
            return f"cuda:{self.local_rank}"
        elif device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """加载模型和分词器"""
        try:
            # 尝试从本地加载
            if os.path.exists(self.model_path):
                print(f"从本地加载模型: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            else:
                # 从HuggingFace加载
                print(f"从HuggingFace加载模型...")
                model_name = "mlfoundations/tabula-8b"  # 官方模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            
            # 设置分词器参数以避免警告
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("已设置pad_token以避免警告")
            
            # 初始化方案B需要的小模块（基于模型hidden_size）
            try:
                hidden = self.model.config.hidden_size
            except Exception:
                hidden = 4096  # 合理兜底
            proj_dim = 256
            
            # 使用与模型相同的数据类型
            dtype = self.dtype
            
            self.text_proj = nn.Sequential(
                nn.Linear(hidden, proj_dim, dtype=dtype), 
                nn.ReLU()
            ).to(self.device)
            
            # 使用较小的权重初始化，避免梯度爆炸
            for module in self.text_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            self.num_encoder = NumericalEncoder(hidden_dim=proj_dim, dtype=dtype, device=self.device)
            
            self.fusion_head = nn.Sequential(
                nn.Linear(proj_dim * 2, proj_dim, dtype=dtype), 
                nn.ReLU(), 
                nn.Linear(proj_dim, 1, dtype=dtype)
            ).to(self.device)
            
            # 使用较小的权重初始化，避免梯度爆炸
            for module in self.fusion_head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 提示: 请确保模型文件存在或网络连接正常")
            raise


    def prepare_ec_data(self, csv_path, ec_column='EC', sample_size=None):
        """
        准备EC预测数据
        
        Args:
            csv_path: CSV文件路径
            ec_column: EC列名
            sample_size: 采样大小 (None表示使用全部数据)
        
        Returns:
            X: 特征数据
            y: EC目标数据
            feature_names: 特征名称列表
        """
        print(f"📂 加载EC数据: {csv_path}")
        
        # 读取数据
        df = pd.read_csv(csv_path)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"📊 随机采样 {sample_size} 个样本")
        
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 检查EC列是否存在
        if ec_column not in df.columns:
            print(f"⚠️ 未找到EC列 '{ec_column}'，尝试查找相似列名...")
            possible_ec_cols = [col for col in df.columns if 'ec' in col.lower()]
            if possible_ec_cols:
                ec_column = possible_ec_cols[0]
                print(f"✅ 使用列: {ec_column}")
            else:
                raise ValueError(f"未找到EC相关列，可用列: {list(df.columns)}")
        
        # 选择数值特征
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != ec_column]
        
        print(f"🎯 目标变量: {ec_column}")
        print(f"📊 特征变量 ({len(feature_cols)}个): {feature_cols}")
        
        # 提取特征和目标
        X = df[feature_cols].copy()
        y = df[ec_column].copy()
        
        # 处理缺失值
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # 移除无效值
        valid_mask = np.isfinite(y) & (y > 0)  # EC应该为正值
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"📈 有效数据: {len(X)} 样本")
        print(f"EC统计: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
        
        return X, y, feature_cols
    
    def discretize_ec(self, y, num_buckets=4):
        """
        将连续EC值离散化为类别
        
        Args:
            y: EC连续值
            num_buckets: 分桶数量
        
        Returns:
            categories: 离散化类别
            bucket_info: 桶信息
        """
        print(f"🔢 将EC离散化为 {num_buckets} 个类别...")
        
        # 计算分位数阈值
        quantiles = []
        for i in range(1, num_buckets):
            q = np.quantile(y, i / num_buckets)
            quantiles.append(q)
        
        # 定义类别标签
        def get_category(value):
            for i, threshold in enumerate(quantiles):
                if value <= threshold:
                    if i == 0:
                        return f"low (≤{threshold:.2f})"
                    else:
                        return f"medium-{i} ({quantiles[i-1]:.2f}-{threshold:.2f}]"
            return f"high (>{quantiles[-1]:.2f})"
        
        # 应用分类
        categories = y.apply(get_category)
        
        # 生成所有可能的类别
        all_categories = []
        all_categories.append(f"low (≤{quantiles[0]:.2f})")
        for i in range(1, len(quantiles)):
            all_categories.append(f"medium-{i} ({quantiles[i-1]:.2f}-{quantiles[i]:.2f}]")
        all_categories.append(f"high (>{quantiles[-1]:.2f})")
        
        bucket_info = {
            'quantiles': quantiles,
            'categories': all_categories,
            'num_buckets': num_buckets
        }
        
        print(f"📊 分类统计:")
        print(categories.value_counts())
        
        return categories, bucket_info
    
    def _build_rtfm_tokenized_batch(self, row_df: pd.DataFrame, ec_column: str, labeled_examples: pd.DataFrame = None, cfg: TLMConfig = None):
        """基于 rtfm 的序列化/组装，返回包含 input_ids/attention_mask 的 batch（单样本）。"""
        
        is_fewshot = labeled_examples is not None
        data_arguments = DataArguments(
            use_config=True,
            feature_value_handling="none",
            feature_name_handling="none",
            targets_handling="none",
        )
        if cfg is None:
            cfg = TLMConfig(
                prefix=f"Predict the {ec_column}",
                suffix=f"What is the value of {ec_column}?",
                label_values=None,
            )
        if is_fewshot:
            ds_dict = {
                "train": prepare_dataframe(labeled_examples, ec_column, data_arguments),
                "test": prepare_dataframe(row_df, ec_column, data_arguments),
            }
        else:
            ds_dict = {"test": prepare_dataframe(row_df, ec_column, data_arguments)}
        ds_dict = {split: serialize_dataset_fn(ds, data_args=data_arguments, serializer=self.serializer, cfg=cfg) for split, ds in ds_dict.items()}
        ds_dict = {split: ds.map(add_qa_and_eoc_tokens_to_example) for split, ds in ds_dict.items()}
        tokenized_ds_dict = tokenize_ds_dict(ds_dict, tokenizer=self.tokenizer, data_arguments=data_arguments)
        if is_fewshot:
            ds_train = tokenized_ds_dict["train"]
            shots = list(ds_train.select_columns(["input_ids", "labels"]).with_format("torch").take(min(len(labeled_examples), 8)))
            shots = [(x["input_ids"], x["labels"]) for x in shots]
        else:
            shots = None
        ds_test = tokenized_ds_dict["test"]
        target_sample = list(ds_test.select_columns(["input_ids", "labels"]).with_format("torch").take(1))[0]
        target_sample = (target_sample["input_ids"], target_sample["labels"])
        data_collator = DataCollatorForSupervisedDataset(self.tokenizer)
        input_ids, labels = make_few_shot_sample(shots=shots, target_sample=target_sample, max_len=self.tokenizer.model_max_length)
        batch = data_collator([{"input_ids": input_ids, "labels": labels}])
        return batch

    def _encode_text_with_backbone(self, batch: dict) -> torch.Tensor:
        """使用已加载的 CausalLM 的 backbone 提取句向量。"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # 检查输入
        if torch.isnan(input_ids.float()).any():
            print("⚠️ NaN in input_ids!")
            
        # 获取模型的隐藏状态
        with torch.no_grad():
            # 如果是PEFT模型，先获取base_model
            if hasattr(self.model, 'base_model'):
                # PEFT包装的模型
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                # 对于CausalLM，hidden_states在outputs.hidden_states中
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden = outputs.hidden_states[-1]  # 最后一层的hidden states
                else:
                    # 备用方案：尝试获取decoder的hidden states
                    last_hidden = outputs.logits  # 这会导致问题，但至少不会崩溃
                    print("⚠️ Warning: Using logits instead of hidden states, results may be incorrect")
            else:
                # 原始模型，使用backbone
                backbone = getattr(self.model, "model", self.model)
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                last_hidden = outputs.last_hidden_state  # [B, T, H]
            
            # 检查 backbone 输出
            if torch.isnan(last_hidden).any():
                print(f"⚠️ NaN in backbone output! Shape: {last_hidden.shape}")
                print(f"   Hidden stats: min={last_hidden.min().item():.4f}, max={last_hidden.max().item():.4f}")
                
        pooled = self.text_pool(last_hidden, attention_mask)
        
        # 检查池化输出
        if torch.isnan(pooled).any():
            print(f"⚠️ NaN after pooling! Shape: {pooled.shape}")
            print(f"   Pooled stats: min={pooled.min().item():.4f}, max={pooled.max().item():.4f}")
            print(f"   Attention mask sum: {attention_mask.sum().item()}")
            
        result = self.text_proj(pooled)
        
        # 检查投影输出
        if torch.isnan(result).any():
            print(f"⚠️ NaN after projection! Shape: {result.shape}")
            
        return result

    def _build_numeric_tensors(self, row: pd.Series, feature_names: list) -> tuple:
        values = []
        mask01 = []
        for name in feature_names:
            val = row.get(name, None)
            if pd.isna(val):
                values.append(0.0)
                mask01.append(0.0)
            else:
                values.append(float(val))
                mask01.append(1.0)
        values = torch.tensor([values], dtype=self.dtype, device=self.device)
        mask01 = torch.tensor([mask01], dtype=self.dtype, device=self.device)
        return values, mask01

    def fit_fusion_regressor(self, X_train: pd.DataFrame, y_train: pd.Series, feature_names: list, ec_column: str = "EC", 
                            fewshot_k: int = 0, max_train_samples: int = 256, epochs: int = 2, lr: float = 1e-3,
                            batch_size: int = 16, num_workers: int = 0, use_ddp: bool = False):
        """训练融合回归头（冻结大模型，仅训练数值编码与融合头）。"""
        # 构建数值编码器结构
        if self.num_encoder.net is None:
            self.num_encoder.build(len(feature_names))
        
        # 冻结大模型
        for p in self.model.parameters():
            p.requires_grad = False
            
        # 如果使用DDP，包装模型
        if use_ddp and self.local_rank != -1:
            self.text_proj = DDP(self.text_proj, device_ids=[self.local_rank])
            self.num_encoder = DDP(self.num_encoder, device_ids=[self.local_rank])
            self.fusion_head = DDP(self.fusion_head, device_ids=[self.local_rank])
            # 获取基础模块用于参数访问
            text_proj_module = self.text_proj.module
            num_encoder_module = self.num_encoder.module
            fusion_head_module = self.fusion_head.module
        else:
            text_proj_module = self.text_proj
            num_encoder_module = self.num_encoder
            fusion_head_module = self.fusion_head
            
        params = list(text_proj_module.parameters()) + list(num_encoder_module.parameters()) + list(fusion_head_module.parameters())
        optim = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        loss_fn = nn.L1Loss()
        
        self.fusion_head.train()
        self.num_encoder.train()
        self.text_proj.train()
        
        # 准备数据集
        if len(X_train) > max_train_samples:
            indices = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_train_sample = X_train.iloc[indices]
            y_train_sample = y_train.iloc[indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
            
        dataset = ECDataset(X_train_sample, y_train_sample, ec_column)
        
        # 创建数据加载器
        if use_ddp and self.local_rank != -1:
            sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), 
                                        rank=self.local_rank, shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, 
                                  collate_fn=ec_collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  collate_fn=ec_collate_fn)
        
        # 准备few-shot示例
        shots_df = None
        if fewshot_k > 0:
            shots_indices = np.random.choice(len(X_train_sample), 
                                           min(fewshot_k, len(X_train_sample)), 
                                           replace=False)
            shots_df = X_train_sample.iloc[shots_indices].copy()
            shots_df[ec_column] = y_train_sample.iloc[shots_indices]
        
        # 训练循环
        for epoch in range(epochs):
            if use_ddp and self.local_rank != -1:
                sampler.set_epoch(epoch)
                
            total_loss = 0.0
            valid_count = 0
            
            # 创建进度条
            if self.local_rank in [-1, 0]:  # 只在主进程显示进度条
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
            else:
                pbar = dataloader
            
            for batch_idx, (rows, targets) in enumerate(pbar):
                batch_loss = 0.0
                batch_valid = 0
                
                # 处理批次中的每个样本
                for i in range(len(rows)):
                    try:
                        row = rows.iloc[i]
                        row_df = pd.DataFrame([row])
                        
                        # 构建RTFM批次
                        batch = self._build_rtfm_tokenized_batch(
                            row_df, ec_column=ec_column, labeled_examples=shots_df,
                            cfg=TLMConfig(prefix=f"Predict the {ec_column}", 
                                        suffix=f"What is the value of {ec_column}?", 
                                        label_values=None)
                        )
                        
                        # 提取特征
                        text_vec = self._encode_text_with_backbone(batch)
                        vals, mask01 = self._build_numeric_tensors(row.drop(labels=[ec_column]), feature_names)
                        num_vec = self.num_encoder(vals, mask01)
                        
                        # 融合并预测
                        fused = torch.cat([text_vec, num_vec], dim=-1)
                        pred = self.fusion_head(fused).squeeze(-1)
                        target = torch.tensor([targets[i]], dtype=self.dtype, device=self.device)
                        
                        # 检查NaN
                        if torch.isnan(pred) or torch.isnan(target):
                            continue
                            
                        loss = loss_fn(pred, target)
                        
                        if torch.isnan(loss):
                            continue
                            
                        # 累积梯度
                        loss = loss / len(rows)  # 归一化批次损失
                        loss.backward()
                        
                        batch_loss += loss.item() * len(rows)
                        batch_valid += 1
                        
                    except Exception as e:
                        if self.local_rank in [-1, 0]:
                            print(f"⚠️ 样本处理失败: {e}")
                        continue
                
                # 更新参数
                if batch_valid > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    optim.step()
                    optim.zero_grad()
                    
                    total_loss += batch_loss
                    valid_count += batch_valid
                    
                    # 更新进度条
                    if self.local_rank in [-1, 0] and isinstance(pbar, tqdm):
                        pbar.set_postfix({"loss": f"{batch_loss/batch_valid:.4f}", 
                                         "valid": f"{batch_valid}/{len(rows)}"})
            
            # 同步所有进程
            if use_ddp and self.local_rank != -1:
                dist.barrier()
            
            # 更新学习率
            scheduler.step()
            
            # 计算并打印平均损失
            if valid_count > 0:
                avg_loss = total_loss / valid_count
                current_lr = scheduler.get_last_lr()[0]
                if self.local_rank in [-1, 0]:
                    print(f"🧪 Epoch {epoch+1}/{epochs} - train MAE: {avg_loss:.4f} " +
                          f"(valid samples: {valid_count}/{len(dataset)}) - LR: {current_lr:.2e}")
            else:
                if self.local_rank in [-1, 0]:
                    print(f"🧪 Epoch {epoch+1}/{epochs} - No valid samples processed!")
                    
        # 恢复为评估模式
        self.fusion_head.eval()
        self.num_encoder.eval() 
        self.text_proj.eval()
    
    def save_model(self, save_path: str):
        """保存训练好的小头模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 获取基础模块（处理DDP包装）
        text_proj_state = self.text_proj.module.state_dict() if hasattr(self.text_proj, 'module') else self.text_proj.state_dict()
        num_encoder_state = self.num_encoder.module.state_dict() if hasattr(self.num_encoder, 'module') else self.num_encoder.state_dict()
        fusion_head_state = self.fusion_head.module.state_dict() if hasattr(self.fusion_head, 'module') else self.fusion_head.state_dict()
        
        checkpoint = {
            'text_proj_state_dict': text_proj_state,
            'num_encoder_state_dict': num_encoder_state,
            'fusion_head_state_dict': fusion_head_state,
            'model_config': {
                'dtype': str(self.dtype),
                'device': str(self.device),
                'model_path': self.model_path
            },
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        print(f"✅ 模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载训练好的小头模型"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载状态字典
        if hasattr(self.text_proj, 'module'):
            self.text_proj.module.load_state_dict(checkpoint['text_proj_state_dict'])
            self.num_encoder.module.load_state_dict(checkpoint['num_encoder_state_dict'])
            self.fusion_head.module.load_state_dict(checkpoint['fusion_head_state_dict'])
        else:
            self.text_proj.load_state_dict(checkpoint['text_proj_state_dict'])
            self.num_encoder.load_state_dict(checkpoint['num_encoder_state_dict'])
            self.fusion_head.load_state_dict(checkpoint['fusion_head_state_dict'])
        
        print(f"✅ 模型已从 {load_path} 加载")
        print(f"📅 保存时间: {checkpoint.get('timestamp', 'Unknown')}")
        
        return checkpoint.get('model_config', {})

    def predict_ec_continuous(self, X: pd.DataFrame, feature_names: list, ec_column: str = "EC", fewshot_k: int = 0, max_samples: int = 128) -> np.ndarray:
        """使用融合回归头预测连续EC。"""
        preds = []
        shots_df = None
        # 构造少量 few-shot 库（可选）
        if fewshot_k > 0 and ec_column in X.columns:
            # 如果传入的X包含目标列，可以采集few-shot；否则跳过
            shots_df = X.sample(n=min(fewshot_k, len(X)), random_state=321)
        count = 0
        for _, row in X.iterrows():
            row_df = pd.DataFrame([row])
            # 为预测添加一个虚拟的 EC 值（实际值会被忽略）
            row_df[ec_column] = 0.0
            batch = self._build_rtfm_tokenized_batch(row_df, ec_column=ec_column, labeled_examples=shots_df, cfg=TLMConfig(prefix=f"Predict the {ec_column}", suffix=f"What is the value of {ec_column}?", label_values=None))
            text_vec = self._encode_text_with_backbone(batch)
            vals, mask01 = self._build_numeric_tensors(row, feature_names)
            num_vec = self.num_encoder(vals, mask01)
            fused = torch.cat([text_vec, num_vec], dim=-1)
            with torch.no_grad():
                pred = self.fusion_head(fused).squeeze(-1).item()
            preds.append(pred)
            count += 1
            if count >= max_samples:
                break
        return np.array(preds)
    
    def predict_ec_sample(self, target_sample, training_data, ec_column, bucket_info, num_shots=16):
        """
        预测单个样本的EC值
        
        Args:
            target_sample: 目标样本 (单行DataFrame)
            training_data: 训练数据 (包含EC列)
            ec_column: EC列名
            bucket_info: 分桶信息
            num_shots: 少样本学习的样本数
        
        Returns:
            prediction: 预测类别
        """
        # 准备少样本示例
        few_shot_examples = training_data.sample(
            n=min(num_shots, len(training_data)), 
            random_state=42
        )
        
        # 配置任务
        cfg = TLMConfig(
            prefix=f"Predict the {ec_column} category",
            suffix=f"What is the {ec_column} category?",
            label_values=bucket_info['categories']
        )
        
        try:
            # 执行推理
            prediction = infer_on_example(
                model=self.model,
                tokenizer=self.tokenizer,
                serializer=self.serializer,
                target_example=target_sample,
                target_colname=ec_column,
                target_choices=bucket_info['categories'],
                labeled_examples=few_shot_examples,
                cfg=cfg,
                handle_invalid_predictions="warn"
            )
            return prediction
        
        except Exception as e:
            print(f"⚠️ 预测失败: {e}")
            # 返回最常见的类别作为默认值
            return bucket_info['categories'][0]
    
    def batch_predict(self, X_test, X_train, y_train_cat, ec_column, bucket_info, max_samples=100):
        """
        批量预测EC类别
        
        Args:
            X_test: 测试特征
            X_train: 训练特征
            y_train_cat: 训练标签(分类)
            ec_column: EC列名
            bucket_info: 分桶信息
            max_samples: 最大预测样本数
        
        Returns:
            predictions: 预测结果列表
        """
        # 准备训练数据
        train_data = X_train.copy()
        train_data[ec_column] = y_train_cat
        
        # 限制样本数
        if len(X_test) > max_samples:
            X_test = X_test.head(max_samples)
            print(f"📊 限制预测样本数为: {max_samples}")
        
        predictions = []
        total = len(X_test)
        
        print(f"🔮 开始批量预测 {total} 个样本...")
        
        for idx, (_, row) in enumerate(X_test.iterrows()):
            if idx % 10 == 0:
                print(f"进度: {idx+1}/{total}")
            
            # 准备单个样本
            target_sample = pd.DataFrame([row])
            
            # 执行预测
            pred = self.predict_ec_sample(
                target_sample, train_data, ec_column, bucket_info
            )
            predictions.append(pred)
        
        print("✅ 批量预测完成!")
        return predictions
    
    def convert_predictions_to_numeric(self, predictions, bucket_info):
        """
        将分类预测转换为数值EC值
        
        Args:
            predictions: 预测的类别列表
            bucket_info: 桶信息
        
        Returns:
            numeric_values: 数值EC值
        """
        numeric_values = []
        quantiles = bucket_info['quantiles']
        
        for pred in predictions:
            if "low" in pred:
                # 使用第一个分位数的中点
                value = quantiles[0] / 2
            elif "high" in pred:
                # 使用最后一个分位数的1.5倍
                value = quantiles[-1] * 1.5
            elif "medium" in pred:
                # 提取范围并取中点
                try:
                    # 解析类似 "medium-1 (1.23-2.45]" 的格式
                    import re
                    match = re.search(r'\(([\d.]+)-([\d.]+)\]', pred)
                    if match:
                        lower = float(match.group(1))
                        upper = float(match.group(2))
                        value = (lower + upper) / 2
                    else:
                        value = np.mean(quantiles)
                except:
                    value = np.mean(quantiles)
            else:
                value = np.mean(quantiles)
            
            numeric_values.append(value)
        
        return np.array(numeric_values)