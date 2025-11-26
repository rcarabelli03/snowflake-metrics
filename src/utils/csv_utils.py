import os
import numpy as np
import pandas as pd
from typing import Optional



def get_csv_filename_from_path(snowflake_path: str) -> Optional[str]:
    """Generates the CSV filename for snowflake metrics based on the snowflake path."""
    assert os.path.exists(snowflake_path), f"Provided path does not exist: {snowflake_path}"
    for file in os.listdir(snowflake_path):
        if file.endswith(".csv"):
            print(os.path.join(snowflake_path, file))
            return os.path.join(snowflake_path, file)
    return None


def read_csv_metrics(csv_path: str) -> pd.DataFrame:
    """Reads the CSV file containing snowflake image metrics and returns it as a pandas DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def read_single_img_metrics(csv_path: str, idx: int) -> np.ndarray:
    """Reads the CSV file and returns the metrics of image {idx} as a NumPy array."""
    df = read_csv_metrics(csv_path)
    if idx < 0 or idx >= len(df):
        raise IndexError(f"Index {idx} is out of bounds for DataFrame with length {len(df)}")
    metrics_array = df.iloc[idx].to_numpy()
    return metrics_array
    