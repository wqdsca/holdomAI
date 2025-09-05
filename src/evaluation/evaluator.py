"""
Evaluation Framework for Poker Imitation Learning Models
Includes metrics, performance analysis, and strategy evaluation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    action_accuracy: float
    action_precision: Dict[str, float]
    action_recall: Dict[str, float]
    bet_sizing_mae: float
    bet_sizing_mse: float
    expected_value: float
    win_rate: float
    aggression_factor: float
    vpip: float  # Voluntarily Put money In Pot
    pfr: float  # Pre-Flop Raise
    
    def to_dict(self) -> Dict:
        return {
            'action_accuracy': self.action_accuracy,
            'action_precision': self.action_precision,
            'action_recall': self.action_recall,
            'bet_sizing_mae': self.bet_sizing_mae,
            'bet_sizing_mse': self.bet_sizing_mse,
            'expected_value': self.expected_value,
            'win_rate': self.win_rate,
            'aggression_factor': self.aggression_factor,
            'vpip': self.vpip,
            'pfr': self.pfr
        }


class PokerModelEvaluator:
    """Comprehensive evaluator for poker AI models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.action_names = ['fold', 'check', 'call', 'bet', 'raise', 'all-in']
        
    def evaluate_dataset(
        self,
        dataloader,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """Evaluate model on entire dataset"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_bet_predictions = []
        all_bet_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
                features = batch['features'].to(self.device)
                target_actions = batch['action'].squeeze(-1).to(self.device)
                target_bet_sizes = batch['bet_size'].to(self.device)
                
                outputs = self.model(features)
                
                # Get predictions
                pred_actions = torch.argmax(outputs['action_logits'], dim=-1)
                pred_bet_sizes = outputs['bet_size']
                
                all_predictions.extend(pred_actions.cpu().numpy())
                all_targets.extend(target_actions.cpu().numpy())
                all_bet_predictions.extend(pred_bet_sizes.cpu().numpy())
                all_bet_targets.extend(target_bet_sizes.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_targets,
            all_bet_predictions,
            all_bet_targets
        )
        
        if verbose:
            self._print_evaluation_report(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: List[int],
        targets: List[int],
        bet_predictions: List[float],
        bet_targets: List[float]
    ) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        
        # Action classification metrics
        accuracy = np.mean(np.array(predictions) == np.array(targets))
        
        # Per-action precision and recall
        report = classification_report(
            targets, predictions,
            target_names=self.action_names,
            output_dict=True,
            zero_division=0
        )
        
        precision_dict = {name: report[name]['precision'] for name in self.action_names}
        recall_dict = {name: report[name]['recall'] for name in self.action_names}
        
        # Bet sizing metrics
        bet_predictions = np.array(bet_predictions).flatten()
        bet_targets = np.array(bet_targets).flatten()
        
        # Only calculate for bet/raise actions
        bet_mask = np.isin(targets, [3, 4])  # bet or raise
        if bet_mask.any():
            masked_pred = bet_predictions[bet_mask]
            masked_target = bet_targets[bet_mask]
            bet_mae = np.mean(np.abs(masked_pred - masked_target))
            bet_mse = np.mean((masked_pred - masked_target) ** 2)
        else:
            bet_mae = 0.0
            bet_mse = 0.0
        
        # Playing style metrics
        predictions = np.array(predictions)
        aggression_factor = self._calculate_aggression_factor(predictions)
        vpip = self._calculate_vpip(predictions)
        pfr = self._calculate_pfr(predictions)
        
        # Placeholder for expected value and win rate
        # These would need to be calculated from actual game results
        expected_value = 0.0
        win_rate = 0.5
        
        return EvaluationMetrics(
            action_accuracy=accuracy,
            action_precision=precision_dict,
            action_recall=recall_dict,
            bet_sizing_mae=bet_mae,
            bet_sizing_mse=bet_mse,
            expected_value=expected_value,
            win_rate=win_rate,
            aggression_factor=aggression_factor,
            vpip=vpip,
            pfr=pfr
        )
    
    def _calculate_aggression_factor(self, actions: np.ndarray) -> float:
        """Calculate aggression factor (bets + raises) / calls"""
        bets_raises = np.sum(np.isin(actions, [3, 4]))  # bet, raise
        calls = np.sum(actions == 2)
        
        if calls > 0:
            return bets_raises / calls
        return bets_raises
    
    def _calculate_vpip(self, actions: np.ndarray) -> float:
        """Calculate VPIP (% of hands where money voluntarily put in pot)"""
        voluntary_actions = np.isin(actions, [2, 3, 4])  # call, bet, raise
        return np.mean(voluntary_actions)
    
    def _calculate_pfr(self, actions: np.ndarray) -> float:
        """Calculate PFR (% of hands with preflop raise)"""
        raises = actions == 4
        return np.mean(raises)
    
    def _print_evaluation_report(self, metrics: EvaluationMetrics):
        """Print formatted evaluation report"""
        print("\n" + "="*60)
        print("POKER MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Action Accuracy: {metrics.action_accuracy:.2%}")
        
        print("\nPer-Action Performance:")
        print("-"*40)
        for action in self.action_names:
            prec = metrics.action_precision[action]
            rec = metrics.action_recall[action]
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            print(f"{action:8s} | Prec: {prec:.2%} | Rec: {rec:.2%} | F1: {f1:.2%}")
        
        print(f"\nBet Sizing Performance:")
        print("-"*40)
        print(f"MAE: {metrics.bet_sizing_mae:.3f}")
        print(f"MSE: {metrics.bet_sizing_mse:.3f}")
        
        print(f"\nPlaying Style Metrics:")
        print("-"*40)
        print(f"Aggression Factor: {metrics.aggression_factor:.2f}")
        print(f"VPIP: {metrics.vpip:.2%}")
        print(f"PFR: {metrics.pfr:.2%}")
        
        print("="*60)
    
    def compare_with_expert(
        self,
        model_actions: List[int],
        expert_actions: List[int],
        save_path: Optional[Path] = None
    ) -> Dict:
        """Compare model decisions with expert decisions"""
        
        # Create confusion matrix
        cm = confusion_matrix(expert_actions, model_actions)
        
        # Calculate agreement metrics
        agreement_rate = np.mean(np.array(model_actions) == np.array(expert_actions))
        
        # Action distribution comparison
        model_dist = np.bincount(model_actions, minlength=6) / len(model_actions)
        expert_dist = np.bincount(expert_actions, minlength=6) / len(expert_actions)
        
        # KL divergence
        kl_div = np.sum(expert_dist * np.log(expert_dist / (model_dist + 1e-10) + 1e-10))
        
        results = {
            'agreement_rate': agreement_rate,
            'confusion_matrix': cm,
            'model_distribution': model_dist,
            'expert_distribution': expert_dist,
            'kl_divergence': kl_div
        }
        
        if save_path:
            self._plot_confusion_matrix(cm, save_path)
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.action_names,
            yticklabels=self.action_names
        )
        plt.title('Model vs Expert Actions Confusion Matrix')
        plt.ylabel('Expert Action')
        plt.xlabel('Model Action')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_position_play(
        self,
        dataloader,
        positions: List[str] = ['BTN', 'CO', 'MP', 'UTG', 'BB', 'SB']
    ) -> pd.DataFrame:
        """Evaluate model performance by position"""
        position_stats = {pos: {'actions': [], 'correct': 0, 'total': 0} 
                         for pos in positions}
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                positions_batch = batch.get('position', [])
                target_actions = batch['action'].squeeze(-1).to(self.device)
                
                outputs = self.model(features)
                pred_actions = torch.argmax(outputs['action_logits'], dim=-1)
                
                for i, pos in enumerate(positions_batch):
                    if pos in position_stats:
                        position_stats[pos]['total'] += 1
                        if pred_actions[i] == target_actions[i]:
                            position_stats[pos]['correct'] += 1
                        position_stats[pos]['actions'].append(pred_actions[i].item())
        
        # Create summary DataFrame
        summary = []
        for pos in positions:
            stats = position_stats[pos]
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                action_dist = np.bincount(stats['actions'], minlength=6) / len(stats['actions'])
                
                summary.append({
                    'Position': pos,
                    'Accuracy': accuracy,
                    'Total_Hands': stats['total'],
                    'Fold_Rate': action_dist[0],
                    'Check_Rate': action_dist[1],
                    'Call_Rate': action_dist[2],
                    'Bet_Rate': action_dist[3],
                    'Raise_Rate': action_dist[4],
                    'AllIn_Rate': action_dist[5]
                })
        
        return pd.DataFrame(summary)


class StrategyAnalyzer:
    """Analyze learned poker strategies"""
    
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
        
    def analyze_preflop_strategy(self, num_samples: int = 1000) -> pd.DataFrame:
        """Analyze preflop playing strategy"""
        from src.features.feature_extractor import GameState
        
        # Generate all possible starting hands
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        
        results = []
        
        for r1_idx, rank1 in enumerate(ranks):
            for r2_idx, rank2 in enumerate(ranks):
                if r2_idx >= r1_idx:  # Avoid duplicates
                    # Suited
                    hole_cards = [rank1 + suits[0], rank2 + suits[0]]
                    
                    # Create game state
                    game_state = GameState(
                        pot_size=1.5,  # SB + BB
                        stack_sizes=[100.0, 100.0],
                        current_bet=1.0,
                        min_raise=2.0,
                        players_remaining=2,
                        street='preflop',
                        board_cards=[],
                        hole_cards=hole_cards,
                        betting_history=[],
                        position=1,  # Button
                        num_active_players=2
                    )
                    
                    # Get model decision
                    features = self.feature_extractor.extract_features(game_state)
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = self.model(features_tensor)
                        action_probs = outputs['action_probs'].squeeze(0).numpy()
                        recommended_action = np.argmax(action_probs)
                    
                    results.append({
                        'Hand': f"{rank1}{rank2}{'s' if r1_idx == r2_idx else 'o'}",
                        'Suited': r1_idx == r2_idx,
                        'Recommended_Action': self._action_to_string(recommended_action),
                        'Fold_Prob': action_probs[0],
                        'Check_Prob': action_probs[1],
                        'Call_Prob': action_probs[2],
                        'Bet_Prob': action_probs[3],
                        'Raise_Prob': action_probs[4],
                        'AllIn_Prob': action_probs[5]
                    })
        
        return pd.DataFrame(results)
    
    def _action_to_string(self, action_idx: int) -> str:
        """Convert action index to string"""
        actions = ['Fold', 'Check', 'Call', 'Bet', 'Raise', 'All-In']
        return actions[action_idx] if action_idx < len(actions) else 'Unknown'


if __name__ == "__main__":
    # Example evaluation
    from src.models.poker_transformer import PokerTransformer
    from src.features.feature_extractor import FeatureExtractor
    from src.training.trainer import PokerHandDataset
    from torch.utils.data import DataLoader
    
    # Initialize components
    model = PokerTransformer(input_dim=800)
    evaluator = PokerModelEvaluator(model)
    
    # Create dummy test data
    feature_extractor = FeatureExtractor()
    test_data = Path("data/processed/test.json")
    
    if test_data.exists():
        dataset = PokerHandDataset(test_data, feature_extractor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        metrics = evaluator.evaluate_dataset(dataloader)
        print(f"\nEvaluation complete! Accuracy: {metrics.action_accuracy:.2%}")