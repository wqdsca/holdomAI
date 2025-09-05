#!/usr/bin/env python3
"""
Standalone evaluation script for trained poker models
"""

import sys
import argparse
from pathlib import Path
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.poker_transformer import PokerTransformer
from src.features.feature_extractor import FeatureExtractor
from src.training.trainer import PokerHandDataset
from src.evaluation.evaluator import PokerModelEvaluator, StrategyAnalyzer


def main(args):
    print("\n" + "="*60)
    print("POKER MODEL EVALUATION")
    print("="*60)
    
    # Load model
    model_path = Path(args.model) if args.model else Path("models/best_model.pt")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using scripts/train.py")
        return
    
    print(f"Loading model from {model_path}")
    
    # Initialize model (assuming transformer by default)
    model = PokerTransformer(
        input_dim=800,
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_actions=6
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")
    
    # Initialize evaluator
    evaluator = PokerModelEvaluator(model)
    feature_extractor = FeatureExtractor()
    
    # Load test data
    test_data_path = Path(args.data) if args.data else Path("data/processed/test.json")
    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        print("Using validation data instead...")
        test_data_path = Path("data/processed/val.json")
        
        if not test_data_path.exists():
            print("No data found for evaluation. Please run preprocessing first.")
            return
    
    print(f"Loading test data from {test_data_path}")
    test_dataset = PokerHandDataset(test_data_path, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"✓ Loaded {len(test_dataset)} test examples")
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_dataset(test_loader, verbose=True)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_dict = metrics.to_dict()
    with open(results_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n✓ Saved metrics to {results_dir / 'evaluation_metrics.json'}")
    
    # Position-based evaluation
    if args.position_analysis:
        print("\nAnalyzing performance by position...")
        position_df = evaluator.evaluate_position_play(test_loader)
        position_df.to_csv(results_dir / "position_analysis.csv", index=False)
        print(f"✓ Saved position analysis to {results_dir / 'position_analysis.csv'}")
        print("\nPosition Performance:")
        print(position_df.to_string())
    
    # Strategy analysis
    if args.strategy_analysis:
        print("\nAnalyzing learned strategies...")
        analyzer = StrategyAnalyzer(model, feature_extractor)
        
        # Preflop strategy
        preflop_df = analyzer.analyze_preflop_strategy()
        preflop_df.to_csv(results_dir / "preflop_strategy.csv", index=False)
        print(f"✓ Saved preflop strategy to {results_dir / 'preflop_strategy.csv'}")
        
        # Show top hands
        print("\nTop 10 most aggressive preflop hands:")
        aggressive_hands = preflop_df.nlargest(10, 'Raise_Prob')[['Hand', 'Raise_Prob', 'Call_Prob', 'Fold_Prob']]
        print(aggressive_hands.to_string())
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        generate_visualizations(metrics_dict, results_dir)
        print(f"✓ Saved visualizations to {results_dir}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


def generate_visualizations(metrics: dict, output_dir: Path):
    """Generate evaluation visualizations"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Action distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Action precision
    actions = list(metrics['action_precision'].keys())
    precision_values = list(metrics['action_precision'].values())
    
    axes[0].bar(actions, precision_values, color='steelblue')
    axes[0].set_title('Action Precision', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Action')
    axes[0].set_ylabel('Precision')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(precision_values):
        axes[0].text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # Action recall
    recall_values = list(metrics['action_recall'].values())
    
    axes[1].bar(actions, recall_values, color='coral')
    axes[1].set_title('Action Recall', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Action')
    axes[1].set_ylabel('Recall')
    axes[1].set_ylim([0, 1])
    for i, v in enumerate(recall_values):
        axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'action_performance.png', dpi=150)
    plt.close()
    
    # 2. Playing style metrics
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    style_metrics = {
        'Aggression Factor': metrics['aggression_factor'],
        'VPIP': metrics['vpip'] * 100,
        'PFR': metrics['pfr'] * 100,
        'Accuracy': metrics['action_accuracy'] * 100
    }
    
    bars = ax.bar(style_metrics.keys(), style_metrics.values(), 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    ax.set_title('Playing Style Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_ylim([0, max(style_metrics.values()) * 1.2])
    
    # Add value labels
    for bar, (name, value) in zip(bars, style_metrics.items()):
        height = bar.get_height()
        if 'Factor' in name:
            label = f'{value:.2f}'
        else:
            label = f'{value:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               label, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'playing_style.png', dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained poker model")
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, help='Path to test data')
    parser.add_argument('--position-analysis', action='store_true',
                       help='Analyze performance by position')
    parser.add_argument('--strategy-analysis', action='store_true',
                       help='Analyze learned strategies')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Default to all analyses if none specified
    if not any([args.position_analysis, args.strategy_analysis, args.visualize]):
        args.position_analysis = True
        args.strategy_analysis = True
        args.visualize = True
    
    main(args)