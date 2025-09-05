"""
Feature Extraction for Poker Game States
Converts poker game states into numerical features for machine learning
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch
from enum import Enum


class PositionType(Enum):
    """Poker table positions"""
    SB = 0  # Small Blind
    BB = 1  # Big Blind
    UTG = 2  # Under the Gun
    MP = 3  # Middle Position
    CO = 4  # Cutoff
    BTN = 5  # Button


@dataclass
class GameState:
    """Current game state representation"""
    pot_size: float
    stack_sizes: List[float]
    current_bet: float
    min_raise: float
    players_remaining: int
    street: str
    board_cards: List[str]
    hole_cards: List[str]
    betting_history: List[Dict]
    position: int
    num_active_players: int


class FeatureExtractor:
    """Extract features from poker game states for ML models"""
    
    def __init__(self):
        self.card_ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                          '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        self.card_suits = {'s': 0, 'h': 1, 'd': 2, 'c': 3}
        self.streets = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        self.actions = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4, 'all-in': 5}
        
    def card_to_numeric(self, card: str) -> Tuple[int, int]:
        """Convert card string to numeric representation"""
        if len(card) != 2:
            return (0, 0)
        rank = self.card_ranks.get(card[0], 0)
        suit = self.card_suits.get(card[1], 0)
        return (rank, suit)
    
    def encode_cards(self, cards: List[str], max_cards: int = 7) -> np.ndarray:
        """Encode cards as one-hot vectors"""
        encoded = np.zeros((max_cards, 52))
        for i, card in enumerate(cards[:max_cards]):
            if card:
                rank, suit = self.card_to_numeric(card)
                if rank > 0:
                    card_idx = (rank - 2) * 4 + suit
                    encoded[i, card_idx] = 1
        return encoded.flatten()
    
    def extract_hand_strength_features(self, hole_cards: List[str], board: List[str]) -> np.ndarray:
        """Extract hand strength related features"""
        features = []
        
        if len(hole_cards) >= 2:
            # Hole card features
            card1_rank, card1_suit = self.card_to_numeric(hole_cards[0])
            card2_rank, card2_suit = self.card_to_numeric(hole_cards[1])
            
            # Basic features
            features.append(card1_rank / 14.0)  # Normalized rank
            features.append(card2_rank / 14.0)
            features.append(int(card1_suit == card2_suit))  # Suited
            features.append(abs(card1_rank - card2_rank) / 14.0)  # Gap
            features.append(int(card1_rank == card2_rank))  # Pair
            
            # Premium hand indicators
            is_premium = int((card1_rank >= 10 and card2_rank >= 10) or 
                           (card1_rank == card2_rank and card1_rank >= 8))
            features.append(is_premium)
        else:
            features.extend([0] * 6)
        
        # Board texture features (if board exists)
        if board:
            board_ranks = [self.card_to_numeric(card)[0] for card in board]
            board_suits = [self.card_to_numeric(card)[1] for card in board]
            
            # Board characteristics
            features.append(len(set(board_ranks)) / len(board) if board else 0)  # Rank diversity
            features.append(max(board_ranks) / 14.0 if board_ranks else 0)  # Highest card
            features.append(min(board_ranks) / 14.0 if board_ranks else 0)  # Lowest card
            
            # Flush possibilities
            suit_counts = [board_suits.count(s) for s in range(4)]
            features.append(max(suit_counts) / len(board) if board else 0)  # Flush draw potential
            
            # Straight possibilities
            sorted_ranks = sorted(set(board_ranks))
            straight_potential = 0
            for i in range(len(sorted_ranks) - 1):
                if sorted_ranks[i+1] - sorted_ranks[i] == 1:
                    straight_potential += 1
            features.append(straight_potential / max(len(board) - 1, 1) if len(board) > 1 else 0)
        else:
            features.extend([0] * 5)
        
        return np.array(features)
    
    def extract_position_features(self, position: int, num_players: int) -> np.ndarray:
        """Extract positional features"""
        features = np.zeros(6)  # One-hot for position types
        
        if num_players <= 2:
            # Heads-up
            features[PositionType.BTN.value if position == 0 else PositionType.BB.value] = 1
        elif num_players <= 6:
            # 6-max positions
            position_map = {
                0: PositionType.SB,
                1: PositionType.BB,
                2: PositionType.UTG,
                3: PositionType.MP,
                4: PositionType.CO,
                5: PositionType.BTN
            }
            if position in position_map:
                features[position_map[position].value] = 1
        else:
            # Full ring - simplified
            if position == 0:
                features[PositionType.SB.value] = 1
            elif position == 1:
                features[PositionType.BB.value] = 1
            elif position == num_players - 1:
                features[PositionType.BTN.value] = 1
            elif position == num_players - 2:
                features[PositionType.CO.value] = 1
            elif position <= 3:
                features[PositionType.UTG.value] = 1
            else:
                features[PositionType.MP.value] = 1
        
        return features
    
    def extract_betting_features(self, game_state: GameState) -> np.ndarray:
        """Extract betting related features"""
        features = []
        
        # Pot odds and stack features
        hero_stack = game_state.stack_sizes[0] if game_state.stack_sizes else 0
        avg_stack = np.mean(game_state.stack_sizes) if game_state.stack_sizes else 0
        
        features.append(game_state.pot_size / max(avg_stack, 1))  # Pot to stack ratio
        features.append(game_state.current_bet / max(game_state.pot_size, 1))  # Bet to pot ratio
        features.append(hero_stack / max(avg_stack, 1))  # Stack relative to average
        features.append(game_state.current_bet / max(hero_stack, 1))  # Bet to hero stack
        
        # Number of players
        features.append(game_state.num_active_players / 9.0)  # Normalized
        features.append(game_state.players_remaining / max(game_state.num_active_players, 1))
        
        # Street
        street_encoding = np.zeros(4)
        street_idx = self.streets.get(game_state.street, 0)
        street_encoding[street_idx] = 1
        features.extend(street_encoding)
        
        # Aggression metrics from betting history
        num_bets = sum(1 for action in game_state.betting_history 
                      if action.get('action') in ['bet', 'raise'])
        num_calls = sum(1 for action in game_state.betting_history 
                       if action.get('action') == 'call')
        total_actions = len(game_state.betting_history)
        
        features.append(num_bets / max(total_actions, 1))  # Aggression frequency
        features.append(num_calls / max(total_actions, 1))  # Call frequency
        
        return np.array(features)
    
    def extract_action_history_features(self, betting_history: List[Dict], max_actions: int = 20) -> np.ndarray:
        """Extract features from betting action history"""
        # Create sequence of actions
        action_sequence = np.zeros((max_actions, 6))  # 6 action types
        amount_sequence = np.zeros(max_actions)
        
        for i, action in enumerate(betting_history[-max_actions:]):
            if i < max_actions:
                action_type = action.get('action', 'fold')
                action_idx = self.actions.get(action_type, 0)
                action_sequence[i, action_idx] = 1
                amount_sequence[i] = action.get('amount', 0) / 100.0  # Normalize by 100 BB
        
        return np.concatenate([action_sequence.flatten(), amount_sequence])
    
    def extract_features(self, game_state: GameState) -> np.ndarray:
        """Extract all features from game state"""
        features = []
        
        # Hand strength features
        hand_features = self.extract_hand_strength_features(
            game_state.hole_cards, game_state.board_cards
        )
        features.append(hand_features)
        
        # Position features
        position_features = self.extract_position_features(
            game_state.position, game_state.num_active_players
        )
        features.append(position_features)
        
        # Betting features
        betting_features = self.extract_betting_features(game_state)
        features.append(betting_features)
        
        # Card encoding (for deep learning)
        hole_encoding = self.encode_cards(game_state.hole_cards, max_cards=2)
        board_encoding = self.encode_cards(game_state.board_cards, max_cards=5)
        
        # Action history
        history_features = self.extract_action_history_features(game_state.betting_history)
        
        # Concatenate all features
        all_features = np.concatenate([
            np.concatenate(features),
            hole_encoding,
            board_encoding,
            history_features
        ])
        
        return all_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features for interpretability"""
        names = []
        
        # Hand strength features
        names.extend(['card1_rank', 'card2_rank', 'suited', 'gap', 'pocket_pair', 'premium_hand'])
        names.extend(['board_diversity', 'board_high', 'board_low', 'flush_potential', 'straight_potential'])
        
        # Position features
        names.extend([f'position_{p.name}' for p in PositionType])
        
        # Betting features
        names.extend(['pot_to_stack', 'bet_to_pot', 'stack_relative', 'bet_to_stack'])
        names.extend(['players_normalized', 'players_remaining_ratio'])
        names.extend(['street_preflop', 'street_flop', 'street_turn', 'street_river'])
        names.extend(['aggression_freq', 'call_freq'])
        
        # Card encodings
        for i in range(2):
            for j in range(52):
                names.append(f'hole_card_{i}_idx_{j}')
        for i in range(5):
            for j in range(52):
                names.append(f'board_card_{i}_idx_{j}')
        
        # Action history
        for i in range(20):
            for action in ['fold', 'check', 'call', 'bet', 'raise', 'all-in']:
                names.append(f'action_{i}_{action}')
        for i in range(20):
            names.append(f'action_{i}_amount')
        
        return names


class FeatureNormalizer:
    """Normalize features for neural network training"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, features: np.ndarray):
        """Compute normalization parameters"""
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0) + 1e-8  # Avoid division by zero
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before transform")
        return (features - self.mean) / self.std
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(features)
        return self.transform(features)


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Create sample game state
    game_state = GameState(
        pot_size=15.0,
        stack_sizes=[95.0, 100.0, 120.0],
        current_bet=5.0,
        min_raise=10.0,
        players_remaining=3,
        street='flop',
        board_cards=['Ah', '7c', '2d'],
        hole_cards=['As', 'Ks'],
        betting_history=[
            {'action': 'raise', 'amount': 3.0},
            {'action': 'call', 'amount': 3.0},
            {'action': 'fold', 'amount': 0}
        ],
        position=1,
        num_active_players=3
    )
    
    # Extract features
    features = extractor.extract_features(game_state)
    print(f"Feature vector shape: {features.shape}")
    print(f"Number of features: {len(features)}")