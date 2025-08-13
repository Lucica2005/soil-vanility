import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Make sure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from TEC_Fusion.ECPredictor import (
    ECPredictor,  # base class
    ec_collate_fn,
)
from TEC_Fusion.ECFusion import MeanPooler, NumericalEncoder
from TEC_Fusion.ECDataset import ECDataset

# rtfm utilities (used via the base class methods such as _build_rtfm_tokenized_batch)
from rtfm.task_config import TLMConfig

# LoRA / PEFT
from peft import LoraConfig, get_peft_model
from peft.utils import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sklearn.model_selection import train_test_split


class ECPredictorWithFinetune(ECPredictor):
    """
    EC predictor with LoRA-based finetuning on top of TabuLa-8B backbone, while
    retaining the late-fusion head (text projection + numerical encoder + fusion head).

    Exposes the API expected by ec_prediction_with_finetune.py:
      - prepare_ec_data(csv_path, ec_column='EC', sample_size=None) -> X_train, X_test, y_train, y_test, feature_names
      - fit_with_finetune(X_train, y_train, feature_names, epochs, lr, batch_size, ...)
      - predict(X: pd.DataFrame) -> np.ndarray
      - save_model(path), load_model(path)
    """

    def __init__(
        self,
        model_path: str = "/root/autodl-tmp/tabula-8b",
        device: str = "auto",
        use_fp32: bool = False,
        local_rank: int = -1,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        # Initialize base class (loads model/tokenizer and builds small heads)
        super().__init__(model_path=model_path, device=device, use_fp32=use_fp32, local_rank=local_rank)

        # LoRA flags/config
        self.use_lora: bool = use_lora
        self.lora_config_dict: Dict = lora_config or {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        self.lora_applied: bool = False

        # Will be set during training
        self.finetune_feature_names: Optional[List[str]] = None

        # Apply LoRA adapters if requested
        if self.use_lora:
            self._apply_lora_if_needed()

    # -------------------- Data prep --------------------
    def prepare_ec_data(
        self,
        csv_path: str,
        ec_column: str = "EC",
        sample_size: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Load CSV and return train/test split with feature names.
        Matches the usage in ec_prediction_with_finetune.py.
        """
        print(f"📂 加载EC数据: {csv_path}")
        df = pd.read_csv(csv_path)

        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
            print(f"📊 随机采样 {sample_size} 个样本")

        if ec_column not in df.columns:
            print(f"⚠️ 未找到EC列 '{ec_column}'，尝试查找相似列名...")
            possible_ec_cols = [col for col in df.columns if "ec" in col.lower()]
            if possible_ec_cols:
                ec_column = possible_ec_cols[0]
                print(f"✅ 使用列: {ec_column}")
            else:
                raise ValueError(f"未找到EC相关列，可用列: {list(df.columns)}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != ec_column]

        X_all = df[feature_cols].copy()
        y_all = df[ec_column].copy()

        # Basic cleaning
        X_all = X_all.fillna(X_all.median())
        y_all = y_all.fillna(y_all.median())

        valid_mask = np.isfinite(y_all)
        X_all = X_all[valid_mask]
        y_all = y_all[valid_mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=random_state
        )

        print(f"训练集大小: {len(X_train)} | 测试集大小: {len(X_test)} | 特征数量: {len(feature_cols)}")
        return X_train, X_test, y_train, y_test, feature_cols

    # -------------------- LoRA helpers --------------------
    def _apply_lora_if_needed(self):
        if self.lora_applied:
            return
        
        # 如果不使用LoRA，直接返回
        if not self.use_lora:
            print("🔄 跳过LoRA应用（use_lora=False）")
            return
            
        try:
            print(f"🔧 正在应用LoRA配置: {self.lora_config_dict}")
            cfg = LoraConfig(
                r=int(self.lora_config_dict.get("r", 16)),
                lora_alpha=int(self.lora_config_dict.get("lora_alpha", 32)),
                lora_dropout=float(self.lora_config_dict.get("lora_dropout", 0.1)),
                target_modules=self.lora_config_dict.get(
                    "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
                task_type="CAUSAL_LM",
            )
            
            # 检查模型是否已经是PEFT模型
            if hasattr(self.model, 'peft_config'):
                print("⚠️ 模型已经是PEFT模型，跳过重复应用")
                self.lora_applied = True
                return
                
            # Wrap the model with PEFT LoRA
            self.model = get_peft_model(self.model, cfg)
            # Ensure model is on the right device and dtype
            self.model.to(self.device)
            self.lora_applied = True
            print("✅ 已成功应用LoRA适配器!")
            
            # 打印可训练参数统计
            try:
                self.model.print_trainable_parameters()
                
                # 验证LoRA参数确实存在
                lora_param_count = 0
                for name, param in self.model.named_parameters():
                    if "lora" in name.lower() and param.requires_grad:
                        lora_param_count += param.numel()
                print(f"🎯 LoRA可训练参数数量: {lora_param_count:,}")
                
            except Exception as e:
                print(f"⚠️ 打印参数统计失败: {e}")
                
        except ImportError as e:
            print(f"❌ LoRA依赖导入失败: {e}")
            print("💡 建议: pip install peft")
            self.lora_applied = False
            self.use_lora = False  # 禁用LoRA
        except Exception as exc:
            print(f"❌ 应用LoRA失败: {exc}")
            print(f"错误类型: {type(exc).__name__}")
            self.lora_applied = False
            self.use_lora = False  # 禁用LoRA

    def _encode_text_with_backbone_trainable(self, batch: dict) -> torch.Tensor:
        """训练时使用的文本编码方法，允许梯度流向LoRA参数"""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # 训练时不使用no_grad，让LoRA参数能够更新
        if hasattr(self.model, 'base_model'):
            # PEFT包装的模型
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]  # 最后一层的hidden states
            else:
                # 备用方案
                last_hidden = outputs.logits
                print("⚠️ Warning: Using logits instead of hidden states")
        else:
            # 原始模型，使用backbone
            backbone = getattr(self.model, "model", self.model)
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = outputs.last_hidden_state
            
        pooled = self.text_pool(last_hidden, attention_mask)
        result = self.text_proj(pooled)
        return result

    # -------------------- Training --------------------
    def fit_with_finetune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        epochs: int = 5,
        lr: float = 5e-5,
        batch_size: int = 4,
        max_train_samples: int = -1,
        use_ddp: bool = False,
        gradient_accumulation_steps: int = 1,
        freeze_embeddings: bool = False,
        ec_column: str = "EC",
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        save_checkpoint_every: int = 5,
        checkpoint_dir: str = "soil_vanility/TEC_Fusion/checkpoints",
    ) -> Dict:
        """
        Finetune with LoRA along with small fusion heads. The backbone parameters are
        frozen except LoRA adapters if enabled.
        
        Returns:
            Dict: 训练历史，包含loss和验证指标
        """
        # 创建checkpoint目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 初始化训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }
        
        # 检查LoRA状态
        print(f"🔍 训练前LoRA状态检查:")
        print(f"  - use_lora: {self.use_lora}")
        print(f"  - lora_applied: {self.lora_applied}")
        
        if self.use_lora and not self.lora_applied:
            print("⚠️ LoRA未成功应用，将仅训练融合头")
        elif self.use_lora and self.lora_applied:
            print("✅ LoRA已应用，将同时训练LoRA适配器和融合头")
        else:
            print("🔄 未使用LoRA，将仅训练融合头")
            
        # 验证集检查
        has_validation = X_val is not None and y_val is not None
        if has_validation:
            print(f"📊 验证集大小: {len(X_val)}")
        else:
            print("⚠️ 未提供验证集，将无法进行最佳模型选择")
        
        # Optional embedding freeze
        if freeze_embeddings and hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            print("--> 冻结embedding层")
            self.model.model.embed_tokens.weight.requires_grad_(False)

        self.finetune_feature_names = list(feature_names)

        # Ensure NumericalEncoder is built
        if self.num_encoder is None:
            self.num_encoder = NumericalEncoder(hidden_dim=self.text_proj[-2].out_features if isinstance(self.text_proj, nn.Sequential) else 256,
                                                dtype=self.dtype, device=self.device)
        if self.num_encoder.net is None:
            self.num_encoder.build(len(feature_names))

        # Freeze all base model params; LoRA adapters remain trainable
        for param in self.model.parameters():
            param.requires_grad = False
        # Re-enable LoRA params if applied
        if self.use_lora and self.lora_applied:
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

        # Small heads trainable
        for module in [self.text_proj, self.num_encoder, self.fusion_head]:
            for param in module.parameters():
                param.requires_grad = True

        # DDP wrapping for small heads if requested
        if use_ddp and self.local_rank != -1:
            self.text_proj = DDP(self.text_proj, device_ids=[self.local_rank])
            self.num_encoder = DDP(self.num_encoder, device_ids=[self.local_rank])
            self.fusion_head = DDP(self.fusion_head, device_ids=[self.local_rank])
            text_proj_module = self.text_proj.module
            num_encoder_module = self.num_encoder.module
            fusion_head_module = self.fusion_head.module
        else:
            text_proj_module = self.text_proj
            num_encoder_module = self.num_encoder
            fusion_head_module = self.fusion_head

        # Collect trainable parameters: LoRA + heads
        trainable_params = []
        if self.use_lora and self.lora_applied:
            lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
            trainable_params += lora_params
        trainable_params += list(text_proj_module.parameters())
        trainable_params += list(num_encoder_module.parameters())
        trainable_params += list(fusion_head_module.parameters())

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        loss_fn = nn.L1Loss()

        # Build dataset
        if max_train_samples is not None and max_train_samples > 0 and len(X_train) > max_train_samples:
            sampled_indices = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_used = X_train.iloc[sampled_indices]
            y_used = y_train.iloc[sampled_indices]
        else:
            X_used = X_train
            y_used = y_train

        dataset = ECDataset(X_used, y_used, ec_column)

        # Sampler / DataLoader
        if use_ddp and self.local_rank != -1:
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=ec_collate_fn,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=ec_collate_fn,
            )

        self.fusion_head.train()
        self.num_encoder.train()
        self.text_proj.train()
        if hasattr(self.model, "train"):
            self.model.train()

        for epoch in range(epochs):
            if use_ddp and self.local_rank != -1:
                sampler.set_epoch(epoch)

            running_loss = 0.0
            running_count = 0
            step_in_accum = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") if self.local_rank in [-1, 0] else dataloader

            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (rows_df, targets_tensor) in enumerate(progress):
                # iterate each row inside the batch because we construct individual RTFM batches
                batch_loss_sum = 0.0
                batch_valid = 0

                for i in range(len(rows_df)):
                    try:
                        row = rows_df.iloc[i]
                        single_df = pd.DataFrame([row])
                        # Provide a placeholder value for target column if not present for serialization
                        if ec_column not in single_df.columns:
                            single_df[ec_column] = 0.0

                        tokenized_batch = self._build_rtfm_tokenized_batch(
                            row_df=single_df,
                            ec_column=ec_column,
                            labeled_examples=None,
                            cfg=TLMConfig(
                                prefix=f"Predict the {ec_column}",
                                suffix=f"What is the value of {ec_column}?",
                                label_values=None,
                            ),
                        )

                        text_vec = self._encode_text_with_backbone_trainable(tokenized_batch)
                        vals, mask01 = self._build_numeric_tensors(row.drop(labels=[ec_column]) if ec_column in row else row, self.finetune_feature_names)
                        num_vec = self.num_encoder(vals, mask01)
                        fused = torch.cat([text_vec, num_vec], dim=-1)
                        pred = self.fusion_head(fused).squeeze(-1)

                        target_value = targets_tensor[i].to(self.device, dtype=self.dtype)
                        loss = loss_fn(pred, target_value)
                        if torch.isnan(loss):
                            continue

                        # gradient accumulation
                        loss = loss / max(1, gradient_accumulation_steps)
                        loss.backward()
                        step_in_accum += 1

                        batch_loss_sum += loss.item() * max(1, gradient_accumulation_steps)
                        batch_valid += 1

                        if step_in_accum % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    except Exception as exc:
                        if self.local_rank in [-1, 0]:
                            print(f"⚠️ 样本处理失败: {exc}")
                        continue

                if batch_valid > 0:
                    running_loss += batch_loss_sum
                    running_count += batch_valid
                    if self.local_rank in [-1, 0] and isinstance(progress, tqdm):
                        progress.set_postfix({"loss": f"{batch_loss_sum / max(1, batch_valid):.4f}", "valid": f"{batch_valid}"})

            scheduler.step()
            
            # 记录训练损失
            epoch_train_loss = running_loss / max(1, running_count) if running_count > 0 else float('inf')
            history['train_loss'].append(epoch_train_loss)
            
            # 验证阶段
            val_loss, val_rmse, val_r2 = float('inf'), float('inf'), -float('inf')
            if has_validation and self.local_rank in [-1, 0]:
                val_loss, val_rmse, val_r2 = self._validate(X_val, y_val, ec_column)
                history['val_loss'].append(val_loss)
                history['val_rmse'].append(val_rmse)
                history['val_r2'].append(val_r2)
                
                # 检查是否是最佳模型
                if val_loss < history['best_val_loss']:
                    history['best_val_loss'] = val_loss
                    history['best_epoch'] = epoch + 1
                    # 保存最佳模型
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                    self.save_model(best_model_path)
                    print(f"🏆 新的最佳模型! 验证损失: {val_loss:.4f}, 已保存到: {best_model_path}")
            
            # 打印epoch结果
            if self.local_rank in [-1, 0]:
                if running_count > 0:
                    log_msg = f"🧪 Epoch {epoch + 1}/{epochs} - train MAE: {epoch_train_loss:.4f}"
                    if has_validation:
                        log_msg += f", val MAE: {val_loss:.4f}, val RMSE: {val_rmse:.4f}, val R²: {val_r2:.4f}"
                    log_msg += f" (valid samples: {running_count})"
                    print(log_msg)
                else:
                    print(f"🧪 Epoch {epoch + 1}/{epochs} - No valid samples processed!")
            
            # 定期保存checkpoint
            if (epoch + 1) % save_checkpoint_every == 0 and self.local_rank in [-1, 0]:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
                self.save_model(checkpoint_path)
                print(f"💾 Checkpoint已保存: {checkpoint_path}")

        # eval modes
        self.fusion_head.eval()
        self.num_encoder.eval()
        self.text_proj.eval()
        if hasattr(self.model, "eval"):
            self.model.eval()
            
        return history

    def _validate(self, X_val: pd.DataFrame, y_val: pd.Series, ec_column: str = "EC") -> Tuple[float, float, float]:
        """验证模型性能"""
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 设置为评估模式
        self.fusion_head.eval()
        self.num_encoder.eval()
        self.text_proj.eval()
        if hasattr(self.model, "eval"):
            self.model.eval()
            
        # 预测
        with torch.no_grad():
            predictions = self.predict(X_val, ec_column=ec_column, max_samples=min(100, len(X_val)))  # 限制验证样本数
            
        # 计算指标
        y_val_subset = y_val.iloc[:len(predictions)]  # 匹配预测数量
        val_loss = np.mean(np.abs(predictions - y_val_subset))  # MAE
        val_rmse = np.sqrt(mean_squared_error(y_val_subset, predictions))
        val_r2 = r2_score(y_val_subset, predictions)
        
        # 恢复训练模式
        self.fusion_head.train()
        self.num_encoder.train()
        self.text_proj.train()
        if hasattr(self.model, "train"):
            self.model.train()
            
        return val_loss, val_rmse, val_r2

    # -------------------- Inference --------------------
    def predict(self, X: pd.DataFrame, ec_column: str = "EC", max_samples: Optional[int] = None) -> np.ndarray:
        """
        Predict continuous EC values using the trained fusion head.
        """
        if self.finetune_feature_names is None:
            # Fallback: infer from X
            self.finetune_feature_names = list(X.columns)

        preds: List[float] = []
        count = 0
        for _, row in X.iterrows():
            single_df = pd.DataFrame([row])
            if ec_column not in single_df.columns:
                single_df[ec_column] = 0.0

            batch = self._build_rtfm_tokenized_batch(
                row_df=single_df,
                ec_column=ec_column,
                labeled_examples=None,
                cfg=TLMConfig(prefix=f"Predict the {ec_column}", suffix=f"What is the value of {ec_column}?", label_values=None),
            )
            text_vec = self._encode_text_with_backbone(batch)
            vals, mask01 = self._build_numeric_tensors(row, self.finetune_feature_names)
            num_vec = self.num_encoder(vals, mask01)
            fused = torch.cat([text_vec, num_vec], dim=-1)
            with torch.no_grad():
                pred = self.fusion_head(fused).squeeze(-1).item()
            preds.append(float(pred))
            count += 1
            if max_samples is not None and count >= max_samples:
                break
        return np.array(preds)

    # -------------------- Save / Load --------------------
    def save_model(self, save_path: str) -> None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        text_proj_state = self.text_proj.module.state_dict() if hasattr(self.text_proj, "module") else self.text_proj.state_dict()
        num_encoder_state = self.num_encoder.module.state_dict() if hasattr(self.num_encoder, "module") else self.num_encoder.state_dict()
        fusion_head_state = self.fusion_head.module.state_dict() if hasattr(self.fusion_head, "module") else self.fusion_head.state_dict()

        checkpoint = {
            "text_proj_state_dict": text_proj_state,
            "num_encoder_state_dict": num_encoder_state,
            "fusion_head_state_dict": fusion_head_state,
            "model_config": {
                "dtype": str(self.dtype),
                "device": str(self.device),
                "model_path": self.model_path,
            },
            "use_lora": self.use_lora,
            "lora_config": self.lora_config_dict if self.use_lora else None,
            "lora_state_dict": None,
            "feature_names": self.finetune_feature_names,
        }

        # Save LoRA adapter weights into the checkpoint if available
        if self.use_lora and self.lora_applied:
            try:
                lora_state = get_peft_model_state_dict(self.model)
                checkpoint["lora_state_dict"] = {k: v.cpu() for k, v in lora_state.items()}
            except Exception as exc:
                print(f"⚠️ 保存LoRA权重失败: {exc}")

        torch.save(checkpoint, save_path)
        print(f"✅ 模型已保存到: {save_path}")

    def load_model(self, load_path: str) -> Dict:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # Restore heads
        if hasattr(self.text_proj, "module"):
            self.text_proj.module.load_state_dict(checkpoint["text_proj_state_dict"]) 
            self.num_encoder.module.load_state_dict(checkpoint["num_encoder_state_dict"]) 
            self.fusion_head.module.load_state_dict(checkpoint["fusion_head_state_dict"]) 
        else:
            self.text_proj.load_state_dict(checkpoint["text_proj_state_dict"]) 
            self.num_encoder.load_state_dict(checkpoint["num_encoder_state_dict"]) 
            self.fusion_head.load_state_dict(checkpoint["fusion_head_state_dict"]) 

        # Restore LoRA (re-apply if needed, then load weights)
        self.finetune_feature_names = checkpoint.get("feature_names", self.finetune_feature_names)
        self.use_lora = bool(checkpoint.get("use_lora", self.use_lora))
        self.lora_config_dict = checkpoint.get("lora_config", self.lora_config_dict)

        if self.use_lora:
            self._apply_lora_if_needed()
            lora_state = checkpoint.get("lora_state_dict", None)
            if lora_state:
                try:
                    set_peft_model_state_dict(self.model, lora_state)
                except Exception as exc:
                    print(f"⚠️ 加载LoRA权重失败: {exc}")

        print(f"✅ 模型已从 {load_path} 加载")
        return checkpoint.get("model_config", {})


