import os
import joblib
import pandas as pd
import lightgbm as lgb
from pathlib import Path

project_path = Path(__file__).parent.parent

def train():
    X_train = pd.read_parquet(project_path / "data/processed/X_train.parquet")
    y_train = pd.read_parquet(project_path / "data/processed/y_train.parquet")
    
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Distribution: {y_train.value_counts().sort_index().to_dict()}")
    
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=7,
        num_leaves=31,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=42,
        verbose=100
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs(project_path / "models", exist_ok=True)
    joblib.dump(model, project_path / "models/model.joblib")
    
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature importance:")
    print(importance)

if __name__ == "__main__":
    train()
