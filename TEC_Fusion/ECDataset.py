from torch.utils.data import Dataset
import pandas as pd

class ECDataset(Dataset):
    """数据集类，用于批量加载EC数据"""
    def __init__(self, X: pd.DataFrame, y: pd.Series, ec_column: str = "EC"):
        self.X = X
        self.y = y
        self.ec_column = ec_column
        self.data = X.copy()
        self.data[ec_column] = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # 返回字典而不是Series，以便DataLoader可以处理
        return row.to_dict(), float(row[self.ec_column])
