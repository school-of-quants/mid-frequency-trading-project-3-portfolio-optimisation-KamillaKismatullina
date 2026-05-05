import os
import joblib
import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path

project_path = Path(__file__).parent.parent

def run_backtest():
    X_backtest = pd.read_parquet(project_path / "data/processed/X_backtest.parquet")
    full_data = pd.read_parquet(project_path / "data/raw/full_data.parquet")
    
    model = joblib.load(project_path / "models/model.joblib")
    
    preds_proba = model.predict_proba(X_backtest)
    preds = pd.DataFrame(preds_proba, index=X_backtest.index, columns=['p_down', 'p_neutral', 'p_up'])
    
    dates = X_backtest.index.get_level_values('Date').unique()
    tickers = X_backtest.index.get_level_values('Ticker').unique()
    
    signal_df = pd.DataFrame(index=dates, columns=tickers)
    
    for ticker in tickers:
        mask = X_backtest.index.get_level_values('Ticker') == ticker
        ticker_preds = preds[mask]
        if len(ticker_preds) > 0:
            signal_df[ticker] = ticker_preds['p_up'].values - ticker_preds['p_down'].values
    
    signal_df = signal_df.fillna(0)
    weights = signal_df.where(signal_df > 0, 0)
    row_sums = weights.sum(axis=1)
    weights = weights.div(row_sums, axis=0).fillna(0)
    
    close = full_data['Close']
    close_backtest = close.loc[dates]
    close_backtest = close_backtest[tickers]
    close_backtest = close_backtest.dropna(axis=1, how='all')
    weights = weights[close_backtest.columns]
    
    init_cash = 20000000
    fees = 0.001
    
    pf = vbt.Portfolio.from_orders(
        close=close_backtest,
        size=weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq='1d',
        init_cash=init_cash,
        fees=fees
    )
    
    stats = pf.stats()
    total_return = stats['Total Return [%]']
    sharpe = stats['Sharpe Ratio']
    max_dd = stats['Max Drawdown [%]']
    max_dd_days = stats['Max Drawdown Duration']
    
    os.makedirs(project_path / "artifacts/metrics", exist_ok=True)
    
    metrics = {
        "total_return_pct": float(total_return),
        "sharpe_ratio_annualized": float(sharpe * np.sqrt(252)),
        "max_drawdown_pct": float(max_dd),
        "max_drawdown_duration_days": int(max_dd_days),
        "constraints_satisfied": total_return > 0 and max_dd < 20 and max_dd_days < 180
    }
    
    import json
    with open(project_path / "artifacts/metrics/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    run_backtest()
