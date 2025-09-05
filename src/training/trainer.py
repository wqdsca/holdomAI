"""
Training Pipeline for Poker Imitation Learning
Includes data loading, training loops, and optimization strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import pandas as pd


class PokerHandDataset(Dataset):
    """Dataset for poker hand histories"""
    
    def __init__(
        self,
        data_path: Path,
        feature_extractor,
        max_hands: Optional[int] = None,
        transform=None
    ):
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.transform = transform
        
        # Load and process data
        self.hands_data = []
        self.load_data(max_hands)
        
    def load_data(self, max_hands: Optional[int] = None):
        """Load poker hands from processed files"""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                if max_hands:
                    data = data[:max_hands]
                self.hands_data = data
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            if max_hands:
                df = df.head(max_hands)
            self.hands_data = df.to_dict('records')
        
    def __len__(self):
        return len(self.hands_data)
    
    def __getitem__(self, idx):
        """Get a single training example"""
        hand = self.hands_data[idx]
        
        # Extract features from hand
        features = self.extract_hand_features(hand)
        
        # Get target action (what the expert player did)
        target_action = hand.get('action', 0)
        target_bet_size = hand.get('bet_size', 0.0)
        
        # Apply transformations if any
        if self.transform:
            features = self.transform(features)
        
        return {
            'features': torch.FloatTensor(features),
            'action': torch.LongTensor([target_action]),
            'bet_size': torch.FloatTensor([target_bet_size])
        }
    
    def extract_hand_features(self, hand: Dict) -> np.ndarray:
        """Extract features from a hand dictionary"""
        # This would use the feature_extractor to convert hand data to features
        # Simplified version for demonstration
        from src.features.feature_extractor import GameState
        
        game_state = GameState(
            pot_size=hand.get('pot', 0),
            stack_sizes=hand.get('stacks', [100, 100]),
            current_bet=hand.get('current_bet', 0),
            min_raise=hand.get('min_raise', 0),
            players_remaining=hand.get('players_remaining', 2),
            street=hand.get('street', 'preflop'),
            board_cards=hand.get('board', []),
            hole_cards=hand.get('hole_cards', []),
            betting_history=hand.get('betting_history', []),
            position=hand.get('position', 0),
            num_active_players=hand.get('num_players', 2)
        )
        
        return self.feature_extractor.extract_features(game_state)


class ImitationLearningTrainer:
    """Trainer for poker imitation learning models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizers
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.bet_size_loss_fn = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        action_loss_sum = 0
        value_loss_sum = 0
        bet_loss_sum = 0
        correct_actions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            features = batch['features'].to(self.device)
            target_actions = batch['action'].squeeze(-1).to(self.device)
            target_bet_sizes = batch['bet_size'].to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate losses
            action_loss = self.action_loss_fn(outputs['action_logits'], target_actions)
            
            # Value loss (if we have value targets)
            value_loss = torch.tensor(0.0).to(self.device)
            if 'value' in batch:
                target_values = batch['value'].to(self.device)
                value_loss = self.value_loss_fn(outputs['value'], target_values)
            
            # Bet sizing loss (only for bet/raise actions)
            bet_mask = (target_actions == 3) | (target_actions == 4)  # bet or raise
            if bet_mask.any():
                pred_bet_sizes = outputs['bet_size'][bet_mask]
                target_bet_sizes_masked = target_bet_sizes[bet_mask]
                bet_size_loss = self.bet_size_loss_fn(pred_bet_sizes, target_bet_sizes_masked)
            else:
                bet_size_loss = torch.tensor(0.0).to(self.device)
            
            # Combined loss
            loss = action_loss + 0.1 * value_loss + 0.5 * bet_size_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            action_loss_sum += action_loss.item()
            value_loss_sum += value_loss.item()
            bet_loss_sum += bet_size_loss.item()
            
            # Calculate accuracy
            pred_actions = torch.argmax(outputs['action_logits'], dim=-1)
            correct_actions += (pred_actions == target_actions).sum().item()
            total_samples += len(target_actions)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_actions/total_samples:.2%}"
            })
        
        # Calculate epoch metrics
        metrics = {
            'total_loss': total_loss / len(dataloader),
            'action_loss': action_loss_sum / len(dataloader),
            'value_loss': value_loss_sum / len(dataloader),
            'bet_loss': bet_loss_sum / len(dataloader),
            'action_accuracy': correct_actions / total_samples
        }
        
        return metrics
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct_actions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                features = batch['features'].to(self.device)
                target_actions = batch['action'].squeeze(-1).to(self.device)
                target_bet_sizes = batch['bet_size'].to(self.device)
                
                outputs = self.model(features)
                
                # Calculate losses
                action_loss = self.action_loss_fn(outputs['action_logits'], target_actions)
                
                # Bet sizing loss
                bet_mask = (target_actions == 3) | (target_actions == 4)
                if bet_mask.any():
                    pred_bet_sizes = outputs['bet_size'][bet_mask]
                    target_bet_sizes_masked = target_bet_sizes[bet_mask]
                    bet_size_loss = self.bet_size_loss_fn(pred_bet_sizes, target_bet_sizes_masked)
                else:
                    bet_size_loss = torch.tensor(0.0)
                
                loss = action_loss + 0.5 * bet_size_loss
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_actions = torch.argmax(outputs['action_logits'], dim=-1)
                correct_actions += (pred_actions == target_actions).sum().item()
                total_samples += len(target_actions)
        
        metrics = {
            'val_loss': total_loss / len(dataloader),
            'val_accuracy': correct_actions / total_samples
        }
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int = 100,
        save_path: Path = Path('models/best_model.pt')
    ):
        """Full training loop"""
        
        if self.use_wandb:
            wandb.init(project="poker-imitation-learning")
            wandb.watch(self.model)
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_metrics = self.train_epoch(train_dataloader, epoch)
            self.train_losses.append(train_metrics['total_loss'])
            
            # Validation
            val_metrics = self.validate(val_dataloader)
            self.val_losses.append(val_metrics['val_loss'])
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Train Acc: {train_metrics['action_accuracy']:.2%}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.2%}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'train_accuracy': train_metrics['action_accuracy'],
                    'val_loss': val_metrics['val_loss'],
                    'val_accuracy': val_metrics['val_accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_model(save_path)
                print(f"✓ Saved best model with val_loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
    
    def save_model(self, path: Path):
        """Save model checkpoint"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


class AdaptiveLearningRateScheduler:
    """Custom learning rate scheduler with warmup and decay"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    # Example training setup
    from src.models.poker_transformer import PokerTransformer
    from src.features.feature_extractor import FeatureExtractor
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    model = PokerTransformer(input_dim=800, d_model=256, n_heads=8, n_layers=4)
    
    # Create dummy dataset for testing
    dummy_data = Path("data/processed/train.json")
    if not dummy_data.exists():
        dummy_data.parent.mkdir(parents=True, exist_ok=True)
        # Create dummy data
        dummy_hands = [
            {
                'pot': 10.0,
                'stacks': [95.0, 100.0],
                'current_bet': 3.0,
                'action': 2,  # call
                'bet_size': 0.0,
                'street': 'preflop',
                'board': [],
                'hole_cards': ['As', 'Ks']
            }
        ] * 100
        with open(dummy_data, 'w') as f:
            json.dump(dummy_hands, f)
    
    # Create dataset and dataloader
    dataset = PokerHandDataset(dummy_data, feature_extractor, max_hands=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize trainer
    trainer = ImitationLearningTrainer(model)
    
    # Train for one epoch (demonstration)
    metrics = trainer.train_epoch(dataloader, epoch=1)
    print(f"Training metrics: {metrics}")