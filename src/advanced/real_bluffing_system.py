"""
진짜 블러핑 & 스택별 전략 구현 시스템
현실적으로 가능한 고급 포커 AI 기능들
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class BluffingType(Enum):
    """블러핑 타입"""
    PURE_BLUFF = "pure_bluff"          # 순수 블러핑
    SEMI_BLUFF = "semi_bluff"          # 세미 블러핑  
    VALUE_THIN = "value_thin"          # 밸류 띤베팅
    POLARIZED = "polarized"            # 폴라라이즈드
    BALANCED = "balanced"              # 밸런스드


@dataclass
class GameContext:
    """게임 상황 정보"""
    pot_size: float
    hero_stack: float
    villain_stack: float
    board: List[str]
    position: str
    street: str
    betting_history: List[Dict]
    opponent_stats: Dict
    table_image: str  # 'tight', 'loose', 'aggressive', 'passive'


class OpponentModel:
    """상대방 모델링 시스템"""
    
    def __init__(self):
        self.stats = {
            'vpip': 0.25,        # Voluntarily Put money In Pot
            'pfr': 0.18,         # Pre-Flop Raise
            'af': 2.5,           # Aggression Factor
            'fold_to_cbet': 0.35, # Fold to Continuation Bet
            'fold_to_3bet': 0.65, # Fold to 3-bet
            'call_3bet': 0.25,    # Call 3-bet
            '4bet': 0.08,         # 4-bet frequency
        }
        
        # 포지션별 통계
        self.positional_stats = {
            'UTG': {'vpip': 0.12, 'pfr': 0.09},
            'MP': {'vpip': 0.16, 'pfr': 0.12}, 
            'CO': {'vpip': 0.25, 'pfr': 0.18},
            'BTN': {'vpip': 0.35, 'pfr': 0.28},
            'SB': {'vpip': 0.28, 'pfr': 0.15},
            'BB': {'vpip': 0.22, 'pfr': 0.08}
        }
        
        # 스택별 통계
        self.stack_stats = {
            'short': {'push_fold_threshold': 15, 'aggression_multiplier': 1.2},
            'medium': {'steal_frequency': 0.25, 'cbet_frequency': 0.65},
            'deep': {'implied_odds_weight': 1.4, 'bluff_frequency': 0.18}
        }
    
    def update_stats(self, action: str, amount: float, context: GameContext):
        """액션을 보고 상대방 통계 업데이트"""
        if action == 'raise':
            if context.street == 'preflop':
                self.stats['pfr'] = self.stats['pfr'] * 0.95 + 0.05  # 점진적 업데이트
        elif action == 'call':
            self.stats['vpip'] = self.stats['vpip'] * 0.95 + 0.05
        elif action == 'fold':
            self.stats['fold_to_cbet'] = self.stats['fold_to_cbet'] * 0.95 + 0.05
    
    def get_fold_probability(self, bet_size: float, context: GameContext) -> float:
        """베팅 사이즈에 따른 폴드 확률 예측"""
        base_fold = self.stats['fold_to_cbet']
        
        # 베팅 사이즈 조정
        pot_ratio = bet_size / context.pot_size
        size_adjustment = min(0.3, pot_ratio * 0.2)  # 큰 베팅일수록 높은 폴드율
        
        # 스택 깊이 조정
        stack_ratio = min(context.hero_stack, context.villain_stack) / context.pot_size
        if stack_ratio < 5:  # 숏스택이면 폴드 확률 감소
            size_adjustment *= 0.7
        
        return min(0.9, base_fold + size_adjustment)
    
    def get_bluff_catcher_range(self, context: GameContext) -> float:
        """블러프 캐처 레인지 강도 추정"""
        # 상대방이 얼마나 광범위하게 콜할지 예측
        calling_range = 1.0 - self.get_fold_probability(context.pot_size * 0.75, context)
        return calling_range


class AdvancedBluffingEngine:
    """고급 블러핑 엔진"""
    
    def __init__(self):
        self.gto_frequencies = self._load_gto_frequencies()
        
    def _load_gto_frequencies(self) -> Dict:
        """GTO 솔버 기반 블러핑 빈도"""
        return {
            'flop_cbet': 0.65,
            'turn_barrel': 0.35,
            'river_bluff': 0.25,
            'river_thin_value': 0.15,
            '3bet_bluff': 0.08,
            '4bet_bluff': 0.03
        }
    
    def should_bluff(
        self,
        context: GameContext,
        opponent_model: OpponentModel,
        bluff_type: BluffingType = BluffingType.PURE_BLUFF
    ) -> Tuple[bool, float]:
        """블러핑 여부와 베팅 사이즈 결정"""
        
        # 1. 기본 GTO 빈도 확인
        base_frequency = self._get_base_bluff_frequency(context, bluff_type)
        
        # 2. 상대방별 조정
        opponent_adjustment = self._adjust_for_opponent(context, opponent_model)
        
        # 3. 보드 텍스처 조정
        board_adjustment = self._adjust_for_board(context)
        
        # 4. 스택 깊이 조정
        stack_adjustment = self._adjust_for_stacks(context)
        
        # 최종 블러핑 빈도
        final_frequency = base_frequency * opponent_adjustment * board_adjustment * stack_adjustment
        final_frequency = np.clip(final_frequency, 0.05, 0.8)
        
        # 블러핑 결정
        should_bluff = np.random.random() < final_frequency
        
        # 베팅 사이즈 결정
        if should_bluff:
            bet_size = self._calculate_bluff_sizing(context, opponent_model, bluff_type)
        else:
            bet_size = 0
        
        return should_bluff, bet_size
    
    def _get_base_bluff_frequency(self, context: GameContext, bluff_type: BluffingType) -> float:
        """기본 GTO 블러핑 빈도"""
        if context.street == 'flop':
            return self.gto_frequencies['flop_cbet']
        elif context.street == 'turn':
            return self.gto_frequencies['turn_barrel']
        elif context.street == 'river':
            if bluff_type == BluffingType.PURE_BLUFF:
                return self.gto_frequencies['river_bluff']
            else:
                return self.gto_frequencies['river_thin_value']
        return 0.25
    
    def _adjust_for_opponent(self, context: GameContext, opponent_model: OpponentModel) -> float:
        """상대방 성향에 따른 조정"""
        fold_prob = opponent_model.get_fold_probability(context.pot_size * 0.75, context)
        
        # 폴드를 많이 하는 상대방에게는 블러핑 증가
        if fold_prob > 0.6:
            return 1.3  # 30% 증가
        elif fold_prob < 0.3:
            return 0.7  # 30% 감소
        return 1.0
    
    def _adjust_for_board(self, context: GameContext) -> float:
        """보드 텍스처에 따른 조정"""
        board = context.board
        
        if not board:  # 프리플랍
            return 1.0
            
        # 간단한 보드 분석 (실제로는 더 복잡)
        high_cards = sum(1 for card in board if card[0] in 'AKQJT')
        connected = self._is_connected_board(board)
        suited = self._is_suited_board(board)
        
        # 드라이한 보드에서 블러핑 증가
        if high_cards <= 1 and not connected and not suited:
            return 1.2  # 드라이한 보드
        elif high_cards >= 3 or (connected and suited):
            return 0.8  # 웨트한 보드
        return 1.0
    
    def _adjust_for_stacks(self, context: GameContext) -> float:
        """스택 깊이에 따른 조정"""
        effective_stack = min(context.hero_stack, context.villain_stack)
        stack_to_pot = effective_stack / context.pot_size
        
        if stack_to_pot < 2:  # 매우 짧은 스택
            return 0.5  # 블러핑 감소
        elif stack_to_pot > 10:  # 매우 깊은 스택
            return 1.1  # 약간 증가
        return 1.0
    
    def _calculate_bluff_sizing(
        self,
        context: GameContext,
        opponent_model: OpponentModel,
        bluff_type: BluffingType
    ) -> float:
        """블러핑 베팅 사이즈 계산"""
        
        # 상대방 폴드 확률에 기반한 사이징
        fold_prob_50 = opponent_model.get_fold_probability(context.pot_size * 0.5, context)
        fold_prob_75 = opponent_model.get_fold_probability(context.pot_size * 0.75, context)
        fold_prob_100 = opponent_model.get_fold_probability(context.pot_size * 1.0, context)
        
        # 효율성 계산 (폴드확률 / 베팅사이즈)
        efficiency_50 = fold_prob_50 / 0.5
        efficiency_75 = fold_prob_75 / 0.75
        efficiency_100 = fold_prob_100 / 1.0
        
        # 가장 효율적인 사이즈 선택
        if efficiency_50 >= efficiency_75 and efficiency_50 >= efficiency_100:
            return context.pot_size * 0.5
        elif efficiency_75 >= efficiency_100:
            return context.pot_size * 0.75
        else:
            return context.pot_size * 1.0
    
    def _is_connected_board(self, board: List[str]) -> bool:
        """연결된 보드인지 확인"""
        if len(board) < 3:
            return False
        
        ranks = [self._card_rank_value(card[0]) for card in board[:3]]
        ranks.sort()
        
        # 스트레이트 가능성 확인
        return (ranks[2] - ranks[0] <= 4) or (14 in ranks and 2 in ranks and 3 in ranks)
    
    def _is_suited_board(self, board: List[str]) -> bool:
        """수트가 많은 보드인지 확인"""
        if len(board) < 3:
            return False
        
        suits = [card[1] for card in board[:3]]
        return len(set(suits)) <= 2
    
    def _card_rank_value(self, rank: str) -> int:
        """카드 랭크를 숫자로 변환"""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return rank_map.get(rank, 0)


class StackBasedStrategy:
    """스택별 전략 엔진"""
    
    def __init__(self):
        self.stack_thresholds = {
            'push_fold': 15,      # 15BB 이하
            'short': 30,          # 30BB 이하
            'medium': 75,         # 75BB 이하
            'deep': float('inf')  # 75BB 초과
        }
    
    def get_stack_category(self, stack: float, big_blind: float) -> str:
        """스택 카테고리 분류"""
        bb_stack = stack / big_blind
        
        for category, threshold in self.stack_thresholds.items():
            if bb_stack <= threshold:
                return category
        return 'deep'
    
    def get_strategy_adjustments(
        self,
        context: GameContext,
        stack_category: str
    ) -> Dict:
        """스택별 전략 조정"""
        
        if stack_category == 'push_fold':
            return self._push_fold_strategy(context)
        elif stack_category == 'short':
            return self._short_stack_strategy(context)
        elif stack_category == 'medium':
            return self._medium_stack_strategy(context)
        else:
            return self._deep_stack_strategy(context)
    
    def _push_fold_strategy(self, context: GameContext) -> Dict:
        """푸시/폴드 전략"""
        effective_stack = min(context.hero_stack, context.villain_stack)
        bb = 2.0  # 빅블라인드 가정
        bb_stack = effective_stack / bb
        
        # 내시 균형 푸시/폴드 차트 기반
        push_range = self._get_push_range(bb_stack, context.position)
        
        return {
            'strategy_type': 'push_fold',
            'push_range': push_range,
            'betting_sizes': [0, effective_stack],  # 폴드 또는 올인만
            'bluff_frequency': 0.0,  # 푸시/폴드에서는 블러핑 개념 다름
            'call_threshold': self._get_call_threshold(bb_stack)
        }
    
    def _short_stack_strategy(self, context: GameContext) -> Dict:
        """숏스택 전략"""
        return {
            'strategy_type': 'short_stack',
            'preflop_aggression': 1.2,  # 프리플랍 어그레시브
            'postflop_commitment': 0.8,  # 포스트플랍에서 커밋 경향
            'bluff_frequency_multiplier': 0.7,  # 블러핑 감소
            'value_bet_sizing': 0.6,  # 밸류벳 사이징 작게
            'protection_bet_frequency': 1.3  # 프로텍션 베팅 증가
        }
    
    def _medium_stack_strategy(self, context: GameContext) -> Dict:
        """미디엄스택 전략"""  
        return {
            'strategy_type': 'medium_stack',
            'balanced_approach': True,
            'bluff_frequency_multiplier': 1.0,
            'value_bet_sizing': 0.75,
            'semi_bluff_frequency': 1.1,
            'pot_control_threshold': 0.4
        }
    
    def _deep_stack_strategy(self, context: GameContext) -> Dict:
        """딥스택 전략"""
        return {
            'strategy_type': 'deep_stack',
            'implied_odds_weight': 1.4,  # 임플라이드 오즈 중시
            'reverse_implied_odds_weight': 1.3,  # 리버스 임플라이드 오즈 고려
            'bluff_frequency_multiplier': 1.2,  # 블러핑 증가
            'thin_value_betting': 1.3,  # 띤 밸류 베팅 증가
            'multi_street_planning': True,  # 멀티 스트리트 계획
            'pot_building_frequency': 1.1
        }
    
    def _get_push_range(self, bb_stack: float, position: str) -> float:
        """포지션별 푸시 레인지 (간단한 버전)"""
        base_ranges = {
            'UTG': 0.08,   # 8% 레인지
            'MP': 0.12,    # 12% 레인지
            'CO': 0.18,    # 18% 레인지  
            'BTN': 0.35,   # 35% 레인지
            'SB': 0.25,    # 25% 레인지
            'BB': 0.15     # 15% 레인지 (콜 레인지)
        }
        
        base_range = base_ranges.get(position, 0.15)
        
        # 스택이 작을수록 레인지 확장
        stack_adjustment = max(1.0, (20 - bb_stack) * 0.1)
        
        return min(0.5, base_range * stack_adjustment)
    
    def _get_call_threshold(self, bb_stack: float) -> float:
        """숏스택에서의 콜 임계값"""
        # 스택이 작을수록 더 광범위하게 콜
        return max(0.1, 0.25 - (15 - bb_stack) * 0.01)


class RealPokerAI:
    """실제 블러핑과 스택별 전략이 가능한 포커 AI"""
    
    def __init__(self):
        self.bluffing_engine = AdvancedBluffingEngine()
        self.stack_strategy = StackBasedStrategy()
        self.opponent_models = {}  # 상대방별 모델
        
    def get_action(self, context: GameContext, opponent_id: str) -> Tuple[str, float]:
        """최종 액션 결정"""
        
        # 상대방 모델 초기화/업데이트
        if opponent_id not in self.opponent_models:
            self.opponent_models[opponent_id] = OpponentModel()
        
        opponent_model = self.opponent_models[opponent_id]
        
        # 스택 카테고리 결정
        stack_category = self.stack_strategy.get_stack_category(context.hero_stack, 2.0)
        
        # 스택별 전략 조정
        strategy_adjustments = self.stack_strategy.get_strategy_adjustments(context, stack_category)
        
        # 블러핑 결정
        should_bluff, bluff_size = self.bluffing_engine.should_bluff(
            context, opponent_model, BluffingType.PURE_BLUFF
        )
        
        # 최종 액션 결정
        if should_bluff and strategy_adjustments['strategy_type'] != 'push_fold':
            return 'bet', bluff_size
        elif self._should_value_bet(context, strategy_adjustments):
            value_size = self._calculate_value_bet_size(context, strategy_adjustments)
            return 'bet', value_size
        elif self._should_call(context, opponent_model):
            return 'call', context.pot_size  # 현재 베팅 액수에 콜
        else:
            return 'fold', 0
    
    def _should_value_bet(self, context: GameContext, strategy: Dict) -> bool:
        """밸류 베팅 여부 판단 (간단한 버전)"""
        # 실제로는 핸드 스트렝스와 상대방 콜링 레인지 비교
        return random.random() < 0.4  # 임시
    
    def _calculate_value_bet_size(self, context: GameContext, strategy: Dict) -> float:
        """밸류 베팅 사이즈 계산"""
        base_size = context.pot_size * strategy.get('value_bet_sizing', 0.75)
        return base_size
    
    def _should_call(self, context: GameContext, opponent_model: OpponentModel) -> bool:
        """콜 여부 판단"""
        # 팟 오즈 기반 간단한 결정
        pot_odds = context.pot_size / (context.pot_size + context.pot_size * 0.75)  # 임시
        return random.random() < pot_odds


# 사용 예시
if __name__ == "__main__":
    # 실제 포커 AI 테스트
    ai = RealPokerAI()
    
    # 게임 상황 설정
    context = GameContext(
        pot_size=15.0,
        hero_stack=85.0,  # 42.5 BB (미디엄스택)
        villain_stack=120.0,
        board=['Ah', '7c', '2d'],  # 플랍
        position='BTN',
        street='flop',
        betting_history=[],
        opponent_stats={},
        table_image='tight'
    )
    
    # 액션 결정
    action, amount = ai.get_action(context, 'opponent_1')
    print(f"AI 결정: {action} ${amount:.1f}")
    
    print("\n" + "="*50)
    print("🎯 현실적 구현 가능성:")
    print("• 기본적인 GTO 블러핑 빈도: ✅ 가능")
    print("• 상대방별 블러핑 조정: ✅ 기본 수준 가능")  
    print("• 스택별 전략 차별화: ✅ 가능")
    print("• 보드 텍스처별 조정: ✅ 기본 수준 가능")
    print("• 복잡한 심리전: ❌ 여전히 어려움")
    print("• 멀티 스트리트 스토리: ❌ 제한적")
    print("="*50)