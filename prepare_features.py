#каюсь, этот код был сгенерирован, потому что за час до сдачи возникала проблема с закачкой данных
import pandas as pd
import numpy as np
from pathlib import Path

project_path = Path(__file__).parent

def main():
    print("Загрузка сырых данных...")
    data = pd.read_parquet(project_path / "equity_project/data/raw/full_data.parquet")
    print(f"Данные загружены: {data.shape}")
    print(f"Колонки: {data.columns}")
    
    # Проверяем структуру
    print(f"Тип колонок: {type(data.columns)}")
    
    # Получаем цены закрытия
    if 'Close' in data.columns:
        close = data['Close']
        print(f"Close shape: {close.shape}")
    else:
        # Если мультииндекс
        close = data.xs('Close', axis=1, level=0)
        print(f"Close shape (from MultiIndex): {close.shape}")
    
    # Создаём простые признаки без stack
    tickers = close.columns
    print(f"Количество тикеров: {len(tickers)}")
    
    # Результаты будем собирать в списки
    X_list = []
    
    # Доходности
    mom5 = close.pct_change(5)
    mom5 = mom5.stack()
    mom5.name = 'mom5'
    X_list.append(mom5)
    
    mom21 = close.pct_change(21)
    mom21 = mom21.stack()
    mom21.name = 'mom21'
    X_list.append(mom21)
    
    # Скользящие средние
    ma20 = close.rolling(20).mean()
    dev_ma20 = (close - ma20) / close
    dev_ma20 = dev_ma20.stack()
    dev_ma20.name = 'dev_ma20'
    X_list.append(dev_ma20)
    
    ma50 = close.rolling(50).mean()
    dev_ma50 = (close - ma50) / close
    dev_ma50 = dev_ma50.stack()
    dev_ma50.name = 'dev_ma50'
    X_list.append(dev_ma50)
    
    ma200 = close.rolling(200).mean()
    dev_ma200 = (close - ma200) / close
    dev_ma200 = dev_ma200.stack()
    dev_ma200.name = 'dev_ma200'
    X_list.append(dev_ma200)
    
    # Волатильность
    returns = close.pct_change()
    vol10 = returns.rolling(10).std()
    vol10 = vol10.stack()
    vol10.name = 'vol10'
    X_list.append(vol10)
    
    vol21 = returns.rolling(21).std()
    vol21 = vol21.stack()
    vol21.name = 'vol21'
    X_list.append(vol21)
    
    # Объединяем
    X = pd.concat(X_list, axis=1)
    X = X.dropna()
    
    print(f"Признаки созданы: {X.shape}")
    print(f"Колонки: {list(X.columns)}")
    
    # Целевая переменная
    future_return = close.pct_change(5).shift(-5)
    future_return = future_return.stack()
    future_return = future_return.dropna()
    
    threshold = 0.02
    y = pd.Series(1, index=future_return.index)  # нейтрально
    y[future_return > threshold] = 2  # рост
    y[future_return < -threshold] = 0  # падение
    y.name = 'target'
    
    # Обрезаем до одинаковых индексов
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    print(f"Распределение классов: {y.value_counts().to_dict()}")
    
    # Разделение на train/test
    train_end = '2022-12-31'
    
    X_train = X[X.index.get_level_values('Date') <= train_end]
    X_test = X[X.index.get_level_values('Date') > train_end]
    
    y_train = y[y.index.get_level_values('Date') <= train_end]
    y_test = y[y.index.get_level_values('Date') > train_end]
    
    # Сохраняем
    processed_path = project_path / "equity_project/data/processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_parquet(processed_path / "X_train.parquet")
    X_test.to_parquet(processed_path / "X_backtest.parquet")
    y_train.to_frame().to_parquet(processed_path / "y_train.parquet")
    y_test.to_frame().to_parquet(processed_path / "y_backtest.parquet")
    
    print(f"X_train: {X_train.shape}")
    print(f"X_backtest: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_backtest: {y_test.shape}")

if __name__ == "__main__":
    main()