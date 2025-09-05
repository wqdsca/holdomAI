"""
Transformer-based Poker Imitation Learning Model
Uses self-attention to learn complex poker strategies from expert play
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from einops import rearrange
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PokerAttentionBlock(nn.Module):
    """Multi-head attention block for poker sequences"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class PokerTransformer(nn.Module):
    """
    Transformer model for poker action prediction
    Learns from expert play through imitation learning
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_actions: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_actions = n_actions
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            PokerAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_actions)
        )
        
        # Value head (for estimating expected value)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Bet sizing head (for continuous bet amounts)
        self.bet_sizing_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1 (fraction of pot)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            x: Input features [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            mask: Optional attention mask
            
        Returns:
            Dictionary with action logits, value estimates, and bet sizing
        """
        # Handle both sequential and single-step inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len = x.shape[:2]
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Pool sequence (take last position for current decision)
        if seq_len > 1:
            x = x[:, -1, :]  # Take last position
        else:
            x = x.squeeze(1)
        
        # Get outputs from different heads
        action_logits = self.action_head(x)
        value = self.value_head(x)
        bet_size = self.bet_sizing_head(x)
        
        return {
            'action_logits': action_logits,
            'action_probs': F.softmax(action_logits, dim=-1),
            'value': value,
            'bet_size': bet_size
        }
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action from model output
        
        Args:
            x: Input features
            deterministic: If True, return argmax; if False, sample from distribution
            
        Returns:
            (action_idx, bet_size_fraction)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            if deterministic:
                action = torch.argmax(outputs['action_logits'], dim=-1)
            else:
                # Sample from action distribution
                probs = outputs['action_probs']
                action = torch.multinomial(probs, 1).squeeze(-1)
            
            bet_size = outputs['bet_size'].squeeze(-1)
            
        return action.item(), bet_size.item()


class PokerCNN(nn.Module):
    """
    Convolutional neural network for poker (alternative to transformer)
    Faster inference but less expressive
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256, 128],
        n_actions: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.action_head = nn.Linear(prev_dim, n_actions)
        self.value_head = nn.Linear(prev_dim, 1)
        self.bet_sizing_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        return {
            'action_logits': self.action_head(features),
            'action_probs': F.softmax(self.action_head(features), dim=-1),
            'value': self.value_head(features),
            'bet_size': self.bet_sizing_head(features)
        }


class PokerModelEnsemble(nn.Module):
    """
    Ensemble of multiple poker models for robust decision making
    """
    
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Average predictions from all models"""
        all_outputs = [model(x) for model in self.models]
        
        # Average the outputs
        avg_outputs = {}
        for key in all_outputs[0].keys():
            if key == 'action_probs':
                # Average probabilities
                probs = torch.stack([out[key] for out in all_outputs])
                avg_outputs[key] = probs.mean(dim=0)
                avg_outputs['action_logits'] = torch.log(avg_outputs[key] + 1e-8)
            else:
                # Average other outputs
                values = torch.stack([out[key] for out in all_outputs])
                avg_outputs[key] = values.mean(dim=0)
        
        return avg_outputs


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    input_dim = 800  # Feature dimension from feature extractor
    seq_len = 10
    
    # Create model
    model = PokerTransformer(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        n_actions=6
    )
    
    # Test with random input
    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model(x)
    
    print(f"Action logits shape: {outputs['action_logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    print(f"Bet size shape: {outputs['bet_size'].shape}")
    
    # Test action selection
    action, bet_size = model.get_action(x[0].unsqueeze(0))
    print(f"Selected action: {action}, Bet size: {bet_size:.2%} of pot")