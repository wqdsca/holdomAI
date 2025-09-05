#!/usr/bin/env python3
"""
Main training script for Poker Imitation Learning
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.poker_transformer import PokerTransformer, PokerCNN
from src.features.feature_extractor import FeatureExtractor, FeatureNormalizer
from src.training.trainer import PokerHandDataset, ImitationLearningTrainer
from src.evaluation.evaluator import PokerModelEvaluator


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    # Load configuration
    config_path = Path(args.config) if args.config else Path("configs/training_config.yaml")
    if not config_path.exists():
        # Create default config
        config_path.parent.mkdir(exist_ok=True)
        default_config = {
            'data': {
                'train_path': 'data/processed/train.json',
                'val_path': 'data/processed/val.json',
                'test_path': 'data/processed/test.json',
                'max_hands': None
            },
            'model': {
                'type': 'transformer',  # 'transformer' or 'cnn'
                'input_dim': 800,
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'n_actions': 6,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'num_epochs': 100,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_workers': 4,
                'use_wandb': False
            },
            'paths': {
                'checkpoint_dir': 'models/checkpoints',
                'best_model_path': 'models/best_model.pt'
            }
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config at {config_path}")
        config = default_config
    else:
        config = load_config(config_path)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    feature_normalizer = FeatureNormalizer()
    
    # Load datasets
    print("Loading datasets...")
    train_data_path = Path(config['data']['train_path'])
    
    if not train_data_path.exists():
        print(f"Training data not found at {train_data_path}")
        print("Please download the PHH dataset and run preprocessing first.")
        return
    
    # Create dataset
    full_dataset = PokerHandDataset(
        train_data_path,
        feature_extractor,
        max_hands=config['data']['max_hands']
    )
    
    # Split into train/val if val_path not provided
    if 'val_path' in config['data'] and Path(config['data']['val_path']).exists():
        val_dataset = PokerHandDataset(
            Path(config['data']['val_path']),
            feature_extractor,
            max_hands=config['data']['max_hands']
        )
        train_dataset = full_dataset
    else:
        # 80/20 split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Initialize model
    print(f"Initializing {config['model']['type']} model...")
    
    if config['model']['type'] == 'transformer':
        model = PokerTransformer(
            input_dim=config['model']['input_dim'],
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers'],
            n_actions=config['model']['n_actions'],
            dropout=config['model']['dropout']
        )
    else:  # CNN
        model = PokerCNN(
            input_dim=config['model']['input_dim'],
            hidden_dims=[512, 256, 128],
            n_actions=config['model']['n_actions'],
            dropout=config['model']['dropout']
        )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ImitationLearningTrainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        use_wandb=config['training']['use_wandb']
    )
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            trainer.load_model(checkpoint_path)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    
    # Train model
    print("\nStarting training...")
    save_path = Path(config['paths']['best_model_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_path=save_path
    )
    
    print("\n✓ Training complete!")
    print(f"Best model saved to: {save_path}")
    
    # Final evaluation
    if args.evaluate:
        print("\nRunning final evaluation...")
        evaluator = PokerModelEvaluator(model)
        
        # Load test data if available
        test_path = Path(config['data']['test_path'])
        if test_path.exists():
            test_dataset = PokerHandDataset(test_path, feature_extractor)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            metrics = evaluator.evaluate_dataset(test_loader)
            print(f"Test Accuracy: {metrics.action_accuracy:.2%}")
        else:
            # Evaluate on validation set
            metrics = evaluator.evaluate_dataset(val_loader)
            print(f"Validation Accuracy: {metrics.action_accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Poker Imitation Learning Model")
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    main(args)