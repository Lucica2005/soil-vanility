import torch
import torch.nn as nn

class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

class NumericalEncoder(nn.Module):
    def __init__(self, num_features: int = None, hidden_dim: int = 256, dtype=None, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device if device is not None else torch.device('cpu')
        self.net = None
        if num_features is not None:
            self.build(num_features)
    
    def build(self, num_features: int):
        self.net = nn.Sequential(
            nn.Linear(num_features * 2, self.hidden_dim, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
            nn.ReLU(),
        ).to(self.device)
        
        # 使用较小的权重初始化
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, values: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
        assert self.net is not None, "NumericalEncoder 未初始化：请先调用 build(num_features)"
        x = torch.cat([values, mask01], dim=-1)
        
        # 检查输入
        if torch.isnan(x).any():
            print(f"⚠️ NaN in NumericalEncoder input! Shape: {x.shape}")
            print(f"   Values shape: {values.shape}, has NaN: {torch.isnan(values).any()}")
            print(f"   Mask shape: {mask01.shape}, has NaN: {torch.isnan(mask01).any()}")
            
        result = self.net(x)
        
        # 检查输出
        if torch.isnan(result).any():
            print(f"⚠️ NaN in NumericalEncoder output! Shape: {result.shape}")
            
        return result
    
    def to(self, *args, **kwargs):
        # 更新设备信息
        device = None
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device = torch.device(arg) if isinstance(arg, str) else arg
                break
        if device is not None:
            self.device = device
        return super().to(*args, **kwargs)