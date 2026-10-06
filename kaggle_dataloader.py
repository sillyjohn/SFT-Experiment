import pandas as pd
import glob
import kagglehub

class KaggleDataLoader:
    def __init__(self, path: str):
        files = glob.glob(f"{path}/*.parquet")
        if not files:
            raise FileNotFoundError(
                f"No parquet files found in {path}."
            )
        df = pd.concat([pd.read_parquet(f) for f in files])
        self.data = df.reset_index(drop=True) # Reset index to ensure it's sequential and starts from 0
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "question": self.data.iloc[index,0],
            "answer": self.data.iloc[index, 1]
        }
    
    def get_columns_name(self):
        return self.data.columns