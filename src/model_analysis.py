"""
Model analysis and visualization tools.
"""
import os
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

def analyze_model(track_name: str):
    """Analyze a trained model for a specific track."""
    # Load model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, f'pitstop_model_{track_name}.txt')
    if not os.path.exists(model_path):
        raise ValueError(f"No trained model found for {track_name}")
    
    model = lgb.Booster(model_file=model_path)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=10)
    plt.title(f'Feature Importance - {track_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{track_name}_feature_importance.png'))
    plt.close()
    
    # Plot first tree
    plt.figure(figsize=(20, 10))
    lgb.plot_tree(model, tree_index=0)
    plt.title(f'Decision Tree Structure - {track_name} (Tree 0)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{track_name}_tree_0.png'))
    plt.close()
    
    # Print model summary
    print(f"\nModel Analysis for {track_name}:")
    print("=" * 50)
    print(f"Number of trees: {model.num_trees()}")
    print(f"Feature names: {model.feature_name()}")
    
    # Get feature importance scores
    importance = model.feature_importance()
    feature_names = model.feature_name()
    importance_dict = dict(zip(feature_names, importance))
    print("\nFeature Importance Scores:")
    for feat, score in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{feat}: {score}")
    
    return model

if __name__ == '__main__':
    analyze_model('Japan')
