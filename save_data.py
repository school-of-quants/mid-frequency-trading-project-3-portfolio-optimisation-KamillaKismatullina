import pandas as pd
import numpy as np
from pathlib import Path

project_path = Path(__file__).parent

# Импортируем нужные функции из get_data
import sys
sys.path.insert(0, str(project_path))

from equity_project.src.get_data import get_raw_data
from equity_project.src.utils import load_config

def save_raw_data():
    """Сохраняет только сырые OHLC данные без вычисления лейблов"""
    
    data, first_appearance_dict = get_raw_data()

    raw_path = project_path / "equity_project/data/raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    
    output_file = raw_path / "full_data.parquet"
    data.to_parquet(output_file)
    print(f"Данные сохранены в {output_file}")
    
    first_appearance_df = pd.DataFrame(
        list(first_appearance_dict.items()),
        columns=["ticker", "first_appearance"]
    )
    first_appearance_df.to_csv(raw_path / "first_appearance.csv", index=False)
    print(f"Список тикеров сохранён в {raw_path / 'first_appearance.csv'}")
    
    return data

if __name__ == "__main__":
    save_raw_data()