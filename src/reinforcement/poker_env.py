"""
포커 강화학습 환경
OpenAI Gym 호환 포커 게임 환경
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import random
from dataclasses import dataclass
import copy


class PokerAction(Enum):
    FOLD = 0
    CHECK = 1  
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5


@dataclass
class PokerPlayer:
    """포커 플레이어 정보"""
    player_id: int
    stack: float
    position: int
    hole_cards: List[str] = None
    is_folded: bool = False
    is_all_in: bool = False
    total_invested: float = 0
    current_bet: float = 0


class PokerEnvironment(gym.Env):
    """
    Multi-Agent 포커 환경
    6인 테이블 No-Limit Hold'em
    """
    
    def __init__(
        self,
        num_players: int = 6,
        small_blind: float = 1.0,
        big_blind: float = 2.0,
        starting_stack: float = 200.0,
        max_raises_per_street: int = 4
    ):
        super().__init__()
        
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        self.max_raises_per_street = max_raises_per_street
        
        # 액션 공간: [fold, check, call, bet_size_1, bet_size_2, ..., all_in]
        # bet_size는 pot의 비율로 표현 (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        self.bet_sizes = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.action_space = spaces.Discrete(3 + len(self.bet_sizes) + 1)  # fold/check/call + bet_sizes + all_in
        
        # 관찰 공간: [hole_cards, community_cards, player_info, betting_history, pot_info]
        # 784차원 (기존 feature_extractor와 호환)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(800,), dtype=np.float32
        )
        
        # 게임 상태
        self.reset()
        
        # 카드 덱 초기화
        self.ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        self.suits = ['s', 'h', 'd', 'c']
        self.deck = [rank + suit for rank in self.ranks for suit in self.suits]
    
    def reset(self) -> np.ndarray:
        """게임 리셋"""
        # 플레이어 초기화
        self.players = [
            PokerPlayer(
                player_id=i,
                stack=self.starting_stack,
                position=i
            ) for i in range(self.num_players)
        ]
        
        # 게임 상태 초기화
        self.pot = 0.0
        self.community_cards = []
        self.current_bet = 0.0
        self.street = 'preflop'  # preflop, flop, turn, river
        self.active_player = 0
        self.dealer_position = 0
        self.betting_round_complete = False
        self.game_over = False
        self.winner_id = None
        
        # 베팅 히스토리
        self.betting_history = []
        self.street_actions = []
        self.raises_this_street = 0
        
        # 블라인드 포스팅
        self._post_blinds()
        
        # 카드 딜링
        self._deal_hole_cards()
        
        return self._get_observation(self.active_player)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """액션 실행"""
        if self.game_over:
            return self._get_observation(self.active_player), 0.0, True, {}
        
        # 액션 실행
        reward = self._execute_action(self.active_player, action)
        
        # 베팅 라운드 완료 체크
        if self._is_betting_round_complete():
            self._complete_betting_round()
        
        # 게임 종료 체크
        done = self._check_game_over()
        
        # 다음 플레이어로 이동
        if not done and not self.betting_round_complete:
            self._next_player()
        
        info = self._get_info()
        
        return self._get_observation(self.active_player), reward, done, info
    
    def _post_blinds(self):
        """블라인드 포스팅"""
        sb_pos = (self.dealer_position + 1) % self.num_players
        bb_pos = (self.dealer_position + 2) % self.num_players
        
        # 스몰 블라인드
        sb_bet = min(self.small_blind, self.players[sb_pos].stack)
        self.players[sb_pos].stack -= sb_bet
        self.players[sb_pos].current_bet = sb_bet
        self.players[sb_pos].total_invested += sb_bet
        self.pot += sb_bet
        
        # 빅 블라인드
        bb_bet = min(self.big_blind, self.players[bb_pos].stack)
        self.players[bb_pos].stack -= bb_bet
        self.players[bb_pos].current_bet = bb_bet
        self.players[bb_pos].total_invested += bb_bet
        self.pot += bb_bet
        self.current_bet = bb_bet
        
        # 첫 액션 플레이어 (UTG)
        self.active_player = (self.dealer_position + 3) % self.num_players
    
    def _deal_hole_cards(self):
        """홀 카드 딜링"""
        deck = self.deck.copy()
        random.shuffle(deck)
        
        for player in self.players:
            player.hole_cards = [deck.pop(), deck.pop()]
        
        self.remaining_deck = deck
    
    def _execute_action(self, player_id: int, action: int) -> float:
        """액션 실행 및 보상 계산"""
        player = self.players[player_id]
        reward = 0.0
        
        if action == PokerAction.FOLD.value:
            player.is_folded = True
            self.street_actions.append({'player': player_id, 'action': 'fold', 'amount': 0})
            
        elif action == PokerAction.CHECK.value:
            if self.current_bet == player.current_bet:
                self.street_actions.append({'player': player_id, 'action': 'check', 'amount': 0})
            else:
                # 체크 불가능하면 폴드로 처리
                player.is_folded = True
                self.street_actions.append({'player': player_id, 'action': 'fold', 'amount': 0})
                
        elif action == PokerAction.CALL.value:
            call_amount = min(self.current_bet - player.current_bet, player.stack)
            player.stack -= call_amount
            player.current_bet += call_amount
            player.total_invested += call_amount
            self.pot += call_amount
            
            if player.stack == 0:
                player.is_all_in = True
            
            self.street_actions.append({'player': player_id, 'action': 'call', 'amount': call_amount})
            
        elif action >= 3 and action < 3 + len(self.bet_sizes):
            # 베팅/레이징
            bet_size_ratio = self.bet_sizes[action - 3]
            bet_amount = self.pot * bet_size_ratio
            
            # 최소 레이즈 체크
            if self.current_bet > 0:
                min_raise = self.current_bet * 2 - player.current_bet
                bet_amount = max(bet_amount, min_raise)
            
            bet_amount = min(bet_amount, player.stack + player.current_bet)
            actual_bet = bet_amount - player.current_bet
            
            if actual_bet > 0 and self.raises_this_street < self.max_raises_per_street:
                player.stack -= actual_bet
                player.current_bet = bet_amount
                player.total_invested += actual_bet
                self.pot += actual_bet
                self.current_bet = bet_amount
                self.raises_this_street += 1
                
                if player.stack == 0:
                    player.is_all_in = True
                
                action_name = 'bet' if self.current_bet == actual_bet else 'raise'
                self.street_actions.append({'player': player_id, 'action': action_name, 'amount': actual_bet})
            else:
                # 베팅 불가능하면 콜 처리
                self._execute_action(player_id, PokerAction.CALL.value)
                
        elif action == 3 + len(self.bet_sizes):  # All-in
            all_in_amount = player.stack
            player.stack = 0
            player.current_bet += all_in_amount
            player.total_invested += all_in_amount
            player.is_all_in = True
            self.pot += all_in_amount
            
            if player.current_bet > self.current_bet:
                self.current_bet = player.current_bet
                self.raises_this_street += 1
                
            self.street_actions.append({'player': player_id, 'action': 'all_in', 'amount': all_in_amount})
        
        # 베팅 히스토리에 추가
        self.betting_history.extend(self.street_actions[-1:])
        
        return reward
    
    def _is_betting_round_complete(self) -> bool:
        """베팅 라운드 완료 여부 확인"""
        active_players = [p for p in self.players if not p.is_folded and not p.is_all_in]
        
        if len(active_players) <= 1:
            return True
        
        # 모든 활성 플레이어가 같은 금액을 베팅했는지 확인
        if len(active_players) == 0:
            return True
            
        current_bets = [p.current_bet for p in active_players]
        return len(set(current_bets)) <= 1 and len(self.street_actions) >= len(active_players)
    
    def _complete_betting_round(self):
        """베팅 라운드 완료 처리"""
        self.betting_round_complete = True
        
        # 다음 스트리트로 진행
        if self.street == 'preflop':
            self.street = 'flop'
            self._deal_community_cards(3)
        elif self.street == 'flop':
            self.street = 'turn'
            self._deal_community_cards(1)
        elif self.street == 'turn':
            self.street = 'river'
            self._deal_community_cards(1)
        elif self.street == 'river':
            self._showdown()
            return
        
        # 새 베팅 라운드 시작
        self._start_new_betting_round()
    
    def _deal_community_cards(self, num_cards: int):
        """커뮤니티 카드 딜링"""
        for _ in range(num_cards):
            self.community_cards.append(self.remaining_deck.pop())
    
    def _start_new_betting_round(self):
        """새 베팅 라운드 시작"""
        # 베팅 상태 리셋
        for player in self.players:
            player.current_bet = 0
        
        self.current_bet = 0
        self.raises_this_street = 0
        self.street_actions = []
        self.betting_round_complete = False
        
        # 첫 액션 플레이어 찾기 (딜러 다음부터)
        for i in range(1, self.num_players):
            pos = (self.dealer_position + i) % self.num_players
            if not self.players[pos].is_folded and not self.players[pos].is_all_in:
                self.active_player = pos
                break
    
    def _next_player(self):
        """다음 플레이어로 이동"""
        for i in range(1, self.num_players):
            next_pos = (self.active_player + i) % self.num_players
            player = self.players[next_pos]
            if not player.is_folded and not player.is_all_in:
                self.active_player = next_pos
                return
        
        # 모든 플레이어가 폴드/올인이면 베팅 라운드 완료
        self.betting_round_complete = True
    
    def _check_game_over(self) -> bool:
        """게임 종료 확인"""
        active_players = [p for p in self.players if not p.is_folded]
        
        if len(active_players) <= 1:
            self.game_over = True
            if active_players:
                self.winner_id = active_players[0].player_id
            return True
        
        if self.street == 'river' and self.betting_round_complete:
            self._showdown()
            return True
        
        return False
    
    def _showdown(self):
        """쇼다운 처리"""
        # 간단한 핸드 평가 (실제로는 더 복잡한 로직 필요)
        active_players = [p for p in self.players if not p.is_folded]
        
        if len(active_players) == 1:
            winner = active_players[0]
        else:
            # 랜덤하게 승자 결정 (실제로는 핸드 강도 비교)
            winner = random.choice(active_players)
        
        self.winner_id = winner.player_id
        winner.stack += self.pot
        self.game_over = True
    
    def _get_observation(self, player_id: int) -> np.ndarray:
        """관찰 상태 생성"""
        # 실제로는 feature_extractor를 사용하여 더 정교한 특징 추출
        obs = np.zeros(800, dtype=np.float32)
        
        player = self.players[player_id]
        
        # 기본 정보 인코딩 (예시)
        obs[0] = self.pot / 1000.0  # 정규화된 팟 사이즈
        obs[1] = player.stack / self.starting_stack  # 정규화된 스택
        obs[2] = self.current_bet / self.pot if self.pot > 0 else 0
        obs[3] = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}.get(self.street, 0) / 3.0
        
        # 포지션 정보
        obs[4] = player.position / self.num_players
        
        # 상대방 정보 (간단히)
        for i, p in enumerate(self.players):
            if i != player_id:
                base_idx = 10 + i * 5
                obs[base_idx] = p.stack / self.starting_stack
                obs[base_idx + 1] = 1.0 if p.is_folded else 0.0
                obs[base_idx + 2] = 1.0 if p.is_all_in else 0.0
                obs[base_idx + 3] = p.current_bet / self.pot if self.pot > 0 else 0
        
        return obs
    
    def _calculate_reward(self, player_id: int) -> float:
        """보상 계산"""
        if not self.game_over:
            return 0.0
        
        player = self.players[player_id]
        
        # 최종 스택 변화를 기반으로 보상
        stack_change = player.stack - self.starting_stack
        
        # 정규화된 보상 (-1 to 1 범위)
        max_possible_win = self.starting_stack * (self.num_players - 1)
        normalized_reward = stack_change / max_possible_win
        
        return np.clip(normalized_reward, -1.0, 1.0)
    
    def _get_info(self) -> Dict:
        """추가 정보 반환"""
        return {
            'pot': self.pot,
            'street': self.street,
            'active_player': self.active_player,
            'game_over': self.game_over,
            'winner_id': self.winner_id,
            'num_active_players': len([p for p in self.players if not p.is_folded])
        }
    
    def render(self, mode='human'):
        """게임 상태 출력"""
        print(f"\n{'='*50}")
        print(f"Street: {self.street.upper()}, Pot: ${self.pot:.1f}")
        print(f"Community Cards: {' '.join(self.community_cards)}")
        print(f"Current Bet: ${self.current_bet:.1f}")
        
        for i, player in enumerate(self.players):
            status = ""
            if player.is_folded:
                status = "[FOLDED]"
            elif player.is_all_in:
                status = "[ALL-IN]"
            elif i == self.active_player:
                status = "[ACTIVE]"
                
            print(f"Player {i}: Stack=${player.stack:.1f}, "
                  f"Bet=${player.current_bet:.1f}, "
                  f"Cards={getattr(player, 'hole_cards', ['??', '??'])} {status}")
        
        if self.game_over:
            print(f"\nGame Over! Winner: Player {self.winner_id}")
        print(f"{'='*50}")


class PokerMultiAgentWrapper:
    """다중 에이전트 포커 환경 래퍼"""
    
    def __init__(self, num_players: int = 6):
        self.env = PokerEnvironment(num_players=num_players)
        self.num_players = num_players
        
    def reset(self):
        """모든 에이전트에 대해 초기 관찰 반환"""
        obs = self.env.reset()
        return {f'player_{i}': self.env._get_observation(i) for i in range(self.num_players)}
    
    def step(self, actions: Dict[str, int]):
        """다중 에이전트 스텝"""
        # 현재 활성 플레이어의 액션만 사용
        active_player_key = f'player_{self.env.active_player}'
        action = actions.get(active_player_key, 0)
        
        obs, reward, done, info = self.env.step(action)
        
        # 모든 플레이어에 대한 관찰과 보상 반환
        observations = {f'player_{i}': self.env._get_observation(i) for i in range(self.num_players)}
        rewards = {f'player_{i}': self.env._calculate_reward(i) if done else 0.0 for i in range(self.num_players)}
        dones = {f'player_{i}': done for i in range(self.num_players)}
        infos = {f'player_{i}': info for i in range(self.num_players)}
        
        return observations, rewards, dones, infos


if __name__ == "__main__":
    # 환경 테스트
    env = PokerEnvironment(num_players=3)
    
    obs = env.reset()
    env.render()
    
    # 몇 턴 플레이
    for turn in range(10):
        if env.game_over:
            break
            
        # 랜덤 액션
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nTurn {turn + 1}: Player {env.active_player} action {action}")
        env.render()
        
        if done:
            print(f"Game finished! Final rewards calculated.")
            break