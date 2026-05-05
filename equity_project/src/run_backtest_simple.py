import os
import joblib
import pandas as pd
import numpy as np
import json
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
    
    signal_dict = {}
    for ticker in tickers:
        mask = X_backtest.index.get_level_values('Ticker') == ticker
        ticker_preds = preds[mask]
        if len(ticker_preds) > 0 and len(ticker_preds) == len(dates):
            signal_dict[ticker] = (ticker_preds['p_up'].values - ticker_preds['p_down'].values)
        elif len(ticker_preds) > 0:
            aligned = np.zeros(len(dates))
            aligned[:len(ticker_preds)] = (ticker_preds['p_up'].values - ticker_preds['p_down'].values)
            signal_dict[ticker] = aligned
    
    signal_df = pd.DataFrame(signal_dict, index=dates)
    signal_df = signal_df.fillna(0)
    
    weights = signal_df.where(signal_df > 0, 0)
    row_sums = weights.sum(axis=1)
    row_sums = row_sums.replace(0, 1)
    weights = weights.div(row_sums, axis=0).fillna(0)
    
    close = full_data['Close']
    close_backtest = close.loc[dates]
    close_backtest = close_backtest[weights.columns]
    close_backtest = close_backtest.dropna(axis=1, how='all')
    weights = weights[close_backtest.columns]
    
    returns = close_backtest.pct_change()
    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)
    
    init_cash = 20000000
    fees = 0.001
    turnover = weights.diff().abs().sum(axis=1)
    portfolio_returns = portfolio_returns - fees * turnover
    
    portfolio_returns = portfolio_returns.dropna()
    
    if len(portfolio_returns) == 0:
        metrics = {
            "total_return_pct": 0.0,
            "sharpe_ratio_annualized": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_duration_days": 0,
            "constraints_satisfied": False
        }
    else:
        portfolio_value = (1 + portfolio_returns).cumprod() * init_cash
        total_return = float((portfolio_value.iloc[-1] / init_cash - 1) * 100)
        
        sharpe = float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0.0
        
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min() * 100) if len(drawdown) > 0 else 0.0
        max_dd_days = int((drawdown == drawdown.min()).sum()) if drawdown.min() < 0 else 0
        
        constraints_satisfied = bool(total_return > 0 and max_dd > -20 and max_dd_days < 180)
        
        metrics = {
            "total_return_pct": total_return,
            "sharpe_ratio_annualized": sharpe,
            "max_drawdown_pct": max_dd,
            "max_drawdown_duration_days": max_dd_days,
            "constraints_satisfied": constraints_satisfied
        }
    
    os.makedirs(project_path / "artifacts/metrics", exist_ok=True)
    
    with open(project_path / "artifacts/metrics/backtest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(json.dumps(metrics, indent=4))
    
    try:
        os.makedirs(project_path / "artifacts/plots", exist_ok=True)
        import matplotlib.pyplot as plt
        if 'portfolio_value' in locals() and len(portfolio_value) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_value.index, portfolio_value.values)
            plt.title('Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.grid(True)
            plt.savefig(project_path / "artifacts/plots/pnl.png")
            plt.close()
    except:
        pass

if __name__ == "__main__":
    run_backtest()
