#!/usr/bin/env python3
"""
Preprocess PHH dataset for training
Downloads and processes poker hand histories
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm
import subprocess
import zipfile
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.parser.phh_parser import PHHParser
from src.features.feature_extractor import FeatureExtractor, GameState


def download_phh_dataset(output_dir: Path):
    """Download PHH dataset from GitHub"""
    print("Downloading PHH dataset...")
    
    # Create directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone the repository
    repo_url = "https://github.com/uoftcprg/phh-dataset.git"
    repo_path = output_dir / "phh-dataset"
    
    if not repo_path.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        print(f"✓ Downloaded PHH dataset to {repo_path}")
    else:
        print(f"PHH dataset already exists at {repo_path}")
    
    return repo_path


def process_phh_files(
    input_dir: Path,
    output_dir: Path,
    max_hands: int = None
) -> Dict:
    """Process PHH files and convert to training format"""
    
    parser = PHHParser()
    feature_extractor = FeatureExtractor()
    
    # Find all PHH files
    phh_files = list(input_dir.glob("**/*.txt")) + list(input_dir.glob("**/*.phh"))
    print(f"Found {len(phh_files)} PHH files")
    
    all_processed_hands = []
    stats = {
        'total_files': len(phh_files),
        'total_hands': 0,
        'total_actions': 0,
        'games_by_type': {},
        'actions_by_type': {}
    }
    
    for file_path in tqdm(phh_files, desc="Processing files"):
        try:
            # Parse hands from file
            hands = parser.parse_file(file_path)
            
            for hand in hands:
                if max_hands and len(all_processed_hands) >= max_hands:
                    break
                
                # Process each action in the hand as a training example
                for i, action in enumerate(hand.actions):
                    # Create game state at this point
                    game_state = create_game_state_from_hand(hand, i)
                    
                    # Extract features
                    features = feature_extractor.extract_features(game_state)
                    
                    # Create training example
                    example = {
                        'hand_id': hand.hand_id,
                        'features': features.tolist(),
                        'action': action.action.value,
                        'action_name': action.action.name,
                        'amount': action.amount or 0.0,
                        'street': action.street.value,
                        'pot_before': action.pot_size_before,
                        'game_type': hand.game_type,
                        'stakes': list(hand.stakes),
                        'position': get_player_position(action.player, hand.players),
                        'num_players': len(hand.players)
                    }
                    
                    # Calculate bet size as fraction of pot
                    if action.amount and action.pot_size_before > 0:
                        example['bet_size'] = action.amount / action.pot_size_before
                    else:
                        example['bet_size'] = 0.0
                    
                    all_processed_hands.append(example)
                    
                    # Update statistics
                    stats['total_actions'] += 1
                    action_type = action.action.name
                    stats['actions_by_type'][action_type] = stats['actions_by_type'].get(action_type, 0) + 1
                
                stats['total_hands'] += 1
                stats['games_by_type'][hand.game_type] = stats['games_by_type'].get(hand.game_type, 0) + 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"\nProcessed {stats['total_hands']} hands with {stats['total_actions']} actions")
    
    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split into train/val/test (70/15/15)
    total_examples = len(all_processed_hands)
    train_size = int(0.7 * total_examples)
    val_size = int(0.15 * total_examples)
    
    train_data = all_processed_hands[:train_size]
    val_data = all_processed_hands[train_size:train_size + val_size]
    test_data = all_processed_hands[train_size + val_size:]
    
    # Save splits
    with open(output_dir / "train.json", 'w') as f:
        json.dump(train_data, f)
    print(f"✓ Saved {len(train_data)} training examples")
    
    with open(output_dir / "val.json", 'w') as f:
        json.dump(val_data, f)
    print(f"✓ Saved {len(val_data)} validation examples")
    
    with open(output_dir / "test.json", 'w') as f:
        json.dump(test_data, f)
    print(f"✓ Saved {len(test_data)} test examples")
    
    # Save statistics
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def create_game_state_from_hand(hand, action_idx: int) -> GameState:
    """Create a game state from a hand at a specific action index"""
    from src.features.feature_extractor import GameState
    
    # Get current action
    current_action = hand.actions[action_idx]
    
    # Get betting history up to this point
    betting_history = []
    for i in range(action_idx):
        prev_action = hand.actions[i]
        betting_history.append({
            'action': prev_action.action.name.lower(),
            'amount': prev_action.amount or 0,
            'street': prev_action.street.value
        })
    
    # Find current player
    current_player = None
    for player in hand.players:
        if player.name == current_action.player:
            current_player = player
            break
    
    # Create game state
    game_state = GameState(
        pot_size=current_action.pot_size_before,
        stack_sizes=[p.stack for p in hand.players],
        current_bet=0,  # Would need to calculate from history
        min_raise=hand.stakes[1] * 2,  # Simplified
        players_remaining=len([p for p in hand.players if p.stack > 0]),
        street=current_action.street.value,
        board_cards=hand.board[:get_board_size(current_action.street.value)],
        hole_cards=current_player.hole_cards if current_player else [],
        betting_history=betting_history,
        position=get_player_position(current_action.player, hand.players),
        num_active_players=len(hand.players)
    )
    
    return game_state


def get_board_size(street: str) -> int:
    """Get number of board cards visible at each street"""
    street_cards = {
        'preflop': 0,
        'flop': 3,
        'turn': 4,
        'river': 5
    }
    return street_cards.get(street, 0)


def get_player_position(player_name: str, players: List) -> int:
    """Get player's position at table"""
    for i, player in enumerate(players):
        if player.name == player_name:
            return i
    return 0


def analyze_dataset(data_dir: Path):
    """Analyze the processed dataset"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Load training data
    with open(data_dir / "train.json", 'r') as f:
        train_data = json.load(f)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(train_data)
    
    print(f"\nTotal training examples: {len(df)}")
    print(f"Unique hands: {df['hand_id'].nunique()}")
    print(f"Game types: {df['game_type'].value_counts().to_dict()}")
    
    print("\nAction distribution:")
    action_counts = df['action_name'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action:10s}: {count:6d} ({count/len(df)*100:5.1f}%)")
    
    print("\nStreet distribution:")
    street_counts = df['street'].value_counts()
    for street, count in street_counts.items():
        print(f"  {street:10s}: {count:6d} ({count/len(df)*100:5.1f}%)")
    
    print("\nBet sizing statistics (for bet/raise actions):")
    bet_df = df[df['bet_size'] > 0]
    if len(bet_df) > 0:
        print(f"  Mean bet size: {bet_df['bet_size'].mean():.2f}x pot")
        print(f"  Median bet size: {bet_df['bet_size'].median():.2f}x pot")
        print(f"  Std bet size: {bet_df['bet_size'].std():.2f}")
    
    print("="*60)


def main(args):
    # Set up paths
    raw_data_dir = Path(args.input_dir) if args.input_dir else Path("data/raw")
    processed_data_dir = Path(args.output_dir) if args.output_dir else Path("data/processed")
    
    # Download dataset if requested
    if args.download:
        phh_path = download_phh_dataset(raw_data_dir)
    else:
        phh_path = raw_data_dir / "phh-dataset"
        if not phh_path.exists():
            print(f"PHH dataset not found at {phh_path}")
            print("Use --download flag to download the dataset")
            return
    
    # Process the data
    print("\nProcessing PHH files...")
    stats = process_phh_files(
        input_dir=phh_path,
        output_dir=processed_data_dir,
        max_hands=args.max_hands
    )
    
    # Analyze the dataset
    if args.analyze:
        analyze_dataset(processed_data_dir)
    
    print("\n✓ Data preprocessing complete!")
    print(f"Processed data saved to: {processed_data_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PHH dataset for training")
    parser.add_argument('--download', action='store_true', help='Download PHH dataset from GitHub')
    parser.add_argument('--input-dir', type=str, help='Input directory with PHH files')
    parser.add_argument('--output-dir', type=str, help='Output directory for processed data')
    parser.add_argument('--max-hands', type=int, help='Maximum number of hands to process')
    parser.add_argument('--analyze', action='store_true', help='Analyze the processed dataset')
    
    args = parser.parse_args()
    main(args)