"""
PHH (Poker Hand History) Format Parser
Parses poker hand histories from the standardized PHH format
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from pathlib import Path


class Action(Enum):
    """Poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all-in"
    POST_SB = "post_sb"
    POST_BB = "post_bb"
    POST_ANTE = "post_ante"
    SHOW = "show"
    MUCK = "muck"


class Street(Enum):
    """Poker betting rounds"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass
class Player:
    """Player information"""
    name: str
    seat: int
    stack: float
    position: Optional[str] = None
    hole_cards: List[str] = field(default_factory=list)
    is_hero: bool = False
    final_stack: Optional[float] = None
    winnings: float = 0.0


@dataclass
class HandAction:
    """Individual action in a hand"""
    player: str
    action: Action
    amount: Optional[float] = None
    street: Street = Street.PREFLOP
    pot_size_before: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "player": self.player,
            "action": self.action.value,
            "amount": self.amount,
            "street": self.street.value,
            "pot_size_before": self.pot_size_before
        }


@dataclass
class Hand:
    """Complete poker hand"""
    hand_id: str
    game_type: str
    stakes: Tuple[float, float]  # (small_blind, big_blind)
    players: List[Player]
    actions: List[HandAction]
    board: List[str] = field(default_factory=list)
    pot: float = 0.0
    rake: float = 0.0
    timestamp: Optional[str] = None
    table_name: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "hand_id": self.hand_id,
            "game_type": self.game_type,
            "stakes": self.stakes,
            "players": [{"name": p.name, "stack": p.stack, "cards": p.hole_cards} for p in self.players],
            "actions": [a.to_dict() for a in self.actions],
            "board": self.board,
            "pot": self.pot,
            "rake": self.rake
        }


class PHHParser:
    """Parser for PHH format poker hand histories"""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for parsing"""
        return {
            'hand_start': re.compile(r'Hand #(\d+)'),
            'game_type': re.compile(r'(Hold\'em|Omaha|Stud|Badugi)\s+No Limit'),
            'stakes': re.compile(r'\$(\d+(?:\.\d+)?)/\$(\d+(?:\.\d+)?)'),
            'seat': re.compile(r'Seat (\d+): (.+) \(\$(\d+(?:\.\d+)?)\)'),
            'hole_cards': re.compile(r'Dealt to (.+) \[(.*?)\]'),
            'action': re.compile(r'(.+): (folds|checks|calls|bets|raises|shows|mucks|all-in)(?:\s+\$(\d+(?:\.\d+)?))?'),
            'board': re.compile(r'\*\*\* (FLOP|TURN|RIVER) \*\*\* \[(.*?)\]'),
            'pot': re.compile(r'Total pot \$(\d+(?:\.\d+)?)'),
            'winner': re.compile(r'(.+) collected \$(\d+(?:\.\d+)?)')
        }
    
    def parse_file(self, filepath: Path) -> List[Hand]:
        """Parse a file containing multiple hands"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into individual hands
        hand_texts = re.split(r'\n\n+', content)
        hands = []
        
        for hand_text in hand_texts:
            if hand_text.strip():
                try:
                    hand = self.parse_hand(hand_text)
                    if hand:
                        hands.append(hand)
                except Exception as e:
                    print(f"Error parsing hand: {e}")
                    continue
        
        return hands
    
    def parse_hand(self, hand_text: str) -> Optional[Hand]:
        """Parse a single hand from text"""
        lines = hand_text.strip().split('\n')
        if not lines:
            return None
        
        # Extract hand ID
        hand_match = self.patterns['hand_start'].search(lines[0])
        if not hand_match:
            return None
        hand_id = hand_match.group(1)
        
        # Extract game type and stakes
        game_type = "No Limit Hold'em"  # Default
        stakes = (0.0, 0.0)
        
        for line in lines[:5]:
            game_match = self.patterns['game_type'].search(line)
            if game_match:
                game_type = game_match.group(0)
            
            stakes_match = self.patterns['stakes'].search(line)
            if stakes_match:
                stakes = (float(stakes_match.group(1)), float(stakes_match.group(2)))
        
        # Parse players
        players = []
        for line in lines:
            seat_match = self.patterns['seat'].match(line)
            if seat_match:
                player = Player(
                    name=seat_match.group(2),
                    seat=int(seat_match.group(1)),
                    stack=float(seat_match.group(3))
                )
                players.append(player)
        
        if not players:
            return None
        
        # Parse hole cards
        for line in lines:
            cards_match = self.patterns['hole_cards'].search(line)
            if cards_match:
                player_name = cards_match.group(1)
                cards = cards_match.group(2).split()
                for player in players:
                    if player.name == player_name:
                        player.hole_cards = cards
                        player.is_hero = True
                        break
        
        # Parse actions and board
        actions = []
        board = []
        current_street = Street.PREFLOP
        pot = 0.0
        
        for line in lines:
            # Check for street changes
            board_match = self.patterns['board'].search(line)
            if board_match:
                street_name = board_match.group(1)
                cards = board_match.group(2).split()
                board.extend(cards)
                
                if street_name == "FLOP":
                    current_street = Street.FLOP
                elif street_name == "TURN":
                    current_street = Street.TURN
                elif street_name == "RIVER":
                    current_street = Street.RIVER
            
            # Parse actions
            action_match = self.patterns['action'].search(line)
            if action_match:
                player_name = action_match.group(1).strip()
                action_str = action_match.group(2)
                amount = float(action_match.group(3)) if action_match.group(3) else None
                
                # Map action string to Action enum
                action_map = {
                    'folds': Action.FOLD,
                    'checks': Action.CHECK,
                    'calls': Action.CALL,
                    'bets': Action.BET,
                    'raises': Action.RAISE,
                    'shows': Action.SHOW,
                    'mucks': Action.MUCK,
                    'all-in': Action.ALL_IN
                }
                
                if action_str in action_map:
                    hand_action = HandAction(
                        player=player_name,
                        action=action_map[action_str],
                        amount=amount,
                        street=current_street,
                        pot_size_before=pot
                    )
                    actions.append(hand_action)
                    
                    if amount:
                        pot += amount
        
        # Parse final pot
        for line in reversed(lines):
            pot_match = self.patterns['pot'].search(line)
            if pot_match:
                pot = float(pot_match.group(1))
                break
        
        return Hand(
            hand_id=hand_id,
            game_type=game_type,
            stakes=stakes,
            players=players,
            actions=actions,
            board=board,
            pot=pot
        )
    
    def parse_directory(self, directory: Path) -> pd.DataFrame:
        """Parse all PHH files in a directory and return as DataFrame"""
        all_hands = []
        
        for filepath in directory.glob("*.txt"):
            hands = self.parse_file(filepath)
            all_hands.extend(hands)
        
        # Convert to DataFrame for easier analysis
        data = []
        for hand in all_hands:
            for action in hand.actions:
                data.append({
                    'hand_id': hand.hand_id,
                    'game_type': hand.game_type,
                    'stakes': f"{hand.stakes[0]}/{hand.stakes[1]}",
                    'num_players': len(hand.players),
                    'player': action.player,
                    'action': action.action.value,
                    'amount': action.amount,
                    'street': action.street.value,
                    'pot_before': action.pot_size_before,
                    'board': ' '.join(hand.board),
                    'final_pot': hand.pot
                })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    parser = PHHParser()
    
    # Test with sample hand
    sample_hand = """
Hand #1234567
No Limit Hold'em $0.50/$1.00
Seat 1: Player1 ($100.00)
Seat 2: Player2 ($150.00)
Seat 3: Player3 ($200.00)
Player1 posts small blind $0.50
Player2 posts big blind $1.00
*** HOLE CARDS ***
Dealt to Player1 [As Ks]
Player3: raises $3.00
Player1: calls $3.00
Player2: folds
*** FLOP *** [Ah 7c 2d]
Player1: checks
Player3: bets $5.00
Player1: calls $5.00
*** TURN *** [Ah 7c 2d] [Qh]
Player1: checks
Player3: checks
*** RIVER *** [Ah 7c 2d Qh] [3s]
Player1: bets $10.00
Player3: folds
Player1 collected $17.50
"""
    
    hand = parser.parse_hand(sample_hand)
    if hand:
        print(json.dumps(hand.to_dict(), indent=2))