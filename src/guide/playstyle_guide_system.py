"""
포커 플레이 스타일 가이드 시스템
실제 플레이가 아닌 전략 분석 및 조언 제공
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns


class PlayStyle(Enum):
    """플레이 스타일 분류"""
    TIGHT_AGGRESSIVE = "tight_aggressive"     # 타이트-어그레시브 (TAG)
    LOOSE_AGGRESSIVE = "loose_aggressive"     # 루즈-어그레시브 (LAG)  
    TIGHT_PASSIVE = "tight_passive"           # 타이트-패시브 (Rock)
    LOOSE_PASSIVE = "loose_passive"           # 루즈-패시브 (Fish)
    BALANCED = "balanced"                     # 밸런스드 (GTO)


@dataclass
class HandAnalysis:
    """핸드 분석 결과"""
    hand_strength: str
    position: str
    stack_depth: str
    board_texture: str
    opponent_type: str
    recommended_action: str
    reasoning: str
    alternative_plays: List[str]
    frequency: Dict[str, float]  # 각 액션 빈도
    sizing: Dict[str, float]     # 각 액션별 사이징


class PokerPlayStyleGuide:
    """포커 플레이 스타일 가이드 시스템"""
    
    def __init__(self):
        self.style_profiles = self._create_style_profiles()
        self.expert_database = self._load_expert_patterns()
        
    def _create_style_profiles(self) -> Dict:
        """각 플레이 스타일별 프로필"""
        return {
            PlayStyle.TIGHT_AGGRESSIVE: {
                'vpip': 0.22,           # 22% 핸드만 플레이
                'pfr': 0.18,            # 18% 레이즈
                'aggression_factor': 3.5, # 높은 공격성
                'c_bet_frequency': 0.75,  # 높은 컨티뉴에이션 베팅
                'fold_to_3bet': 0.65,     # 3벳에 자주 폴드
                'steal_frequency': 0.35,  # 적당한 스틸 시도
                'description': '선택적이지만 어그레시브한 플레이',
                'pros': ['안정적 수익', '읽기 쉬운 패턴', '초보자 친화적'],
                'cons': ['예측 가능', '최대 수익 한계', '어그레시브 상대 취약']
            },
            PlayStyle.LOOSE_AGGRESSIVE: {
                'vpip': 0.32,
                'pfr': 0.28,
                'aggression_factor': 4.2,
                'c_bet_frequency': 0.85,
                'fold_to_3bet': 0.45,
                'steal_frequency': 0.55,
                'description': '광범위하고 매우 어그레시브한 플레이',
                'pros': ['최대 수익 가능', '상대방 압박', '이미지 활용'],
                'cons': ['높은 분산', '복잡한 스킬', '뱅크롤 위험']
            },
            PlayStyle.TIGHT_PASSIVE: {
                'vpip': 0.18,
                'pfr': 0.08,
                'aggression_factor': 1.5,
                'c_bet_frequency': 0.45,
                'fold_to_3bet': 0.85,
                'steal_frequency': 0.15,
                'description': '매우 보수적이고 수동적인 플레이',
                'pros': ['낮은 분산', '안전함', '이해하기 쉬움'],
                'cons': ['낮은 수익', '착취당하기 쉬움', '발전 한계']
            },
            PlayStyle.LOOSE_PASSIVE: {
                'vpip': 0.45,
                'pfr': 0.12,
                'aggression_factor': 1.8,
                'c_bet_frequency': 0.35,
                'fold_to_3bet': 0.25,
                'steal_frequency': 0.25,
                'description': '많은 핸드를 콜로 플레이',
                'pros': ['상대방에게 예상외', '임플라이드 오즈 극대화'],
                'cons': ['수익성 매우 낮음', '쉽게 착취당함', '추천하지 않음']
            },
            PlayStyle.BALANCED: {
                'vpip': 0.25,
                'pfr': 0.20,
                'aggression_factor': 2.8,
                'c_bet_frequency': 0.65,
                'fold_to_3bet': 0.55,
                'steal_frequency': 0.42,
                'description': 'GTO 기반 밸런스드 플레이',
                'pros': ['착취당하기 어려움', '이론적 최적', '모든 상황 대응'],
                'cons': ['최대 수익 제한', '복잡함', '상대방별 조정 부족']
            }
        }
    
    def _load_expert_patterns(self) -> Dict:
        """전문가 플레이 패턴 데이터베이스"""
        return {
            'preflop_ranges': {
                'UTG': {
                    'tight': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo', 'AQs'],
                    'standard': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'KQs'],
                    'loose': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'KQs', 'KQo', 'KJs']
                },
                'BTN': {
                    'tight': ['All pairs', 'Ax suited', 'Broadway cards', 'Suited connectors 65s+'],
                    'standard': ['All pairs', 'Any ace', 'King-high', 'Suited connectors', 'One gappers'],
                    'loose': ['60%+ of all hands']
                }
            },
            'postflop_concepts': {
                'continuation_betting': {
                    'dry_boards': {'frequency': 0.85, 'size': '0.5-0.65 pot'},
                    'wet_boards': {'frequency': 0.65, 'size': '0.75-1.0 pot'},
                    'paired_boards': {'frequency': 0.45, 'size': '0.33-0.5 pot'}
                },
                'bluffing_spots': {
                    'turn_barreling': {'frequency': 0.35, 'conditions': ['Equity + fold equity', 'Good runouts']},
                    'river_bluffs': {'frequency': 0.25, 'conditions': ['Polarized range', 'Credible story']}
                }
            }
        }
    
    def analyze_hand_situation(
        self,
        hole_cards: List[str],
        board: List[str],
        position: str,
        stack_bb: float,
        opponent_stats: Dict,
        betting_action: str = "facing_bet"
    ) -> HandAnalysis:
        """특정 상황에 대한 상세한 분석 및 가이드"""
        
        # 1. 핸드 강도 분석
        hand_strength = self._evaluate_hand_strength(hole_cards, board)
        
        # 2. 보드 텍스처 분석
        board_texture = self._analyze_board_texture(board)
        
        # 3. 상대방 타입 분류
        opponent_type = self._classify_opponent(opponent_stats)
        
        # 4. 스택 깊이 카테고리
        stack_category = self._get_stack_category(stack_bb)
        
        # 5. 추천 액션 및 빈도 계산
        recommendations = self._calculate_recommendations(
            hand_strength, board_texture, position, stack_category, opponent_type, betting_action
        )
        
        return HandAnalysis(
            hand_strength=hand_strength,
            position=position,
            stack_depth=stack_category,
            board_texture=board_texture,
            opponent_type=opponent_type,
            recommended_action=recommendations['primary'],
            reasoning=recommendations['reasoning'],
            alternative_plays=recommendations['alternatives'],
            frequency=recommendations['frequencies'],
            sizing=recommendations['sizing']
        )
    
    def generate_style_comparison(self, situation: Dict) -> pd.DataFrame:
        """플레이 스타일별 액션 비교"""
        
        styles = [PlayStyle.TIGHT_AGGRESSIVE, PlayStyle.LOOSE_AGGRESSIVE, 
                 PlayStyle.BALANCED, PlayStyle.TIGHT_PASSIVE]
        
        comparison_data = []
        
        for style in styles:
            # 각 스타일별 액션 빈도 계산
            action_freq = self._get_style_action_frequencies(style, situation)
            
            comparison_data.append({
                'Style': style.value.replace('_', ' ').title(),
                'Fold %': f"{action_freq['fold']*100:.1f}%",
                'Call %': f"{action_freq['call']*100:.1f}%", 
                'Bet/Raise %': f"{action_freq['bet_raise']*100:.1f}%",
                'Primary Action': max(action_freq.items(), key=lambda x: x[1])[0].title(),
                'Aggression Level': self._get_aggression_description(action_freq['bet_raise'])
            })
        
        return pd.DataFrame(comparison_data)
    
    def create_learning_roadmap(self, current_stats: Dict, target_style: PlayStyle) -> Dict:
        """현재 통계에서 목표 스타일까지의 학습 로드맵"""
        
        target_profile = self.style_profiles[target_style]
        
        # 개선이 필요한 영역 식별
        improvements_needed = []
        
        current_vpip = current_stats.get('vpip', 0.25)
        current_pfr = current_stats.get('pfr', 0.15)
        current_af = current_stats.get('aggression_factor', 2.0)
        
        # VPIP 조정
        vpip_diff = target_profile['vpip'] - current_vpip
        if abs(vpip_diff) > 0.05:
            direction = "더 많은" if vpip_diff > 0 else "더 적은"
            improvements_needed.append({
                'area': 'Hand Selection (VPIP)',
                'current': f"{current_vpip:.1%}",
                'target': f"{target_profile['vpip']:.1%}",
                'action': f"{direction} 핸드 플레이하기",
                'priority': 'High' if abs(vpip_diff) > 0.1 else 'Medium'
            })
        
        # PFR 조정
        pfr_diff = target_profile['pfr'] - current_pfr
        if abs(pfr_diff) > 0.03:
            direction = "더 자주" if pfr_diff > 0 else "덜 자주"
            improvements_needed.append({
                'area': 'Preflop Aggression (PFR)',
                'current': f"{current_pfr:.1%}",
                'target': f"{target_profile['pfr']:.1%}",
                'action': f"{direction} 레이즈하기",
                'priority': 'High'
            })
        
        # 공격성 조정
        af_diff = target_profile['aggression_factor'] - current_af
        if abs(af_diff) > 0.5:
            direction = "더 어그레시브하게" if af_diff > 0 else "더 보수적으로"
            improvements_needed.append({
                'area': 'Postflop Aggression (AF)',
                'current': f"{current_af:.1f}",
                'target': f"{target_profile['aggression_factor']:.1f}",
                'action': f"{direction} 베팅/레이즈하기",
                'priority': 'Medium'
            })
        
        # 학습 단계별 계획
        learning_phases = {
            'Phase 1 (1-2주)': {
                'focus': '프리플랍 핸드 선택 조정',
                'goals': ['목표 VPIP 달성', '포지션별 레인지 숙지'],
                'practice': ['레인지 차트 암기', '실제 게임에서 적용']
            },
            'Phase 2 (3-4주)': {
                'focus': '포스트플랍 어그레시브니스 조정',
                'goals': ['목표 AF 달성', 'C-bet 빈도 조정'],
                'practice': ['베팅 사이징 연습', '블러핑 빈도 조절']
            },
            'Phase 3 (5-8주)': {
                'focus': '고급 개념 통합',
                'goals': ['스타일 완전 정착', '상대방별 조정 능력'],
                'practice': ['실전 적용', '결과 분석 및 미세조정']
            }
        }
        
        return {
            'target_style': target_style.value,
            'improvements_needed': improvements_needed,
            'learning_phases': learning_phases,
            'expected_timeline': '6-8주',
            'difficulty': self._assess_transition_difficulty(current_stats, target_profile)
        }
    
    def generate_situation_guide(
        self,
        hole_cards: List[str],
        board: List[str], 
        position: str,
        stack_bb: float,
        pot_size: float,
        bet_to_call: float
    ) -> Dict:
        """특정 상황에 대한 상세 가이드"""
        
        analysis = self.analyze_hand_situation(
            hole_cards, board, position, stack_bb, {'vpip': 0.25, 'pfr': 0.2}
        )
        
        # 수학적 계산
        pot_odds = bet_to_call / (pot_size + bet_to_call)
        
        # 추천 사항
        guide = {
            'situation_summary': {
                'hand': f"{hole_cards[0]} {hole_cards[1]}",
                'board': ' '.join(board) if board else 'Preflop',
                'position': position,
                'stack': f"{stack_bb:.1f} BB",
                'pot_odds': f"{pot_odds:.1%}"
            },
            'hand_analysis': {
                'strength': analysis.hand_strength,
                'board_texture': analysis.board_texture,
                'key_factors': self._identify_key_factors(analysis)
            },
            'recommendations': {
                'primary_action': analysis.recommended_action,
                'reasoning': analysis.reasoning,
                'alternatives': analysis.alternative_plays,
                'frequencies': analysis.frequency
            },
            'style_variations': self._get_style_variations(analysis),
            'learning_points': self._extract_learning_points(analysis),
            'common_mistakes': self._identify_common_mistakes(analysis)
        }
        
        return guide
    
    def _evaluate_hand_strength(self, hole_cards: List[str], board: List[str]) -> str:
        """핸드 강도 평가 (간단한 버전)"""
        if not board:  # 프리플랍
            if hole_cards[0][0] == hole_cards[1][0]:  # 페어
                return "Premium Pair" if hole_cards[0][0] in 'AKQJ' else "Medium Pair"
            elif hole_cards[0][0] in 'AK' and hole_cards[1][0] in 'AK':
                return "Premium Ace"
            else:
                return "Speculative Hand"
        else:
            return "Made Hand"  # 실제로는 더 복잡한 계산 필요
    
    def _analyze_board_texture(self, board: List[str]) -> str:
        """보드 텍스처 분석"""
        if len(board) < 3:
            return "Preflop"
        
        # 간단한 보드 분류
        suits = [card[1] for card in board[:3]]
        ranks = [card[0] for card in board[:3]]
        
        if len(set(suits)) <= 2:
            return "Wet (Flush draws)"
        elif len(set(ranks)) == len(ranks):
            return "Dry (Rainbow)"
        else:
            return "Paired Board"
    
    def _classify_opponent(self, stats: Dict) -> str:
        """상대방 타입 분류"""
        vpip = stats.get('vpip', 0.25)
        pfr = stats.get('pfr', 0.18)
        
        if vpip < 0.2 and pfr < 0.15:
            return "Tight-Passive (Rock)"
        elif vpip < 0.25 and pfr > 0.15:
            return "Tight-Aggressive (TAG)"
        elif vpip > 0.3 and pfr > 0.2:
            return "Loose-Aggressive (LAG)"
        else:
            return "Loose-Passive (Fish)"
    
    def _get_stack_category(self, stack_bb: float) -> str:
        """스택 카테고리 분류"""
        if stack_bb < 20:
            return "Short Stack"
        elif stack_bb < 50:
            return "Medium Stack"
        else:
            return "Deep Stack"
    
    def _calculate_recommendations(self, hand_strength: str, board_texture: str, 
                                 position: str, stack_category: str, opponent_type: str,
                                 betting_action: str) -> Dict:
        """추천 액션 계산"""
        
        # 간단한 결정 트리 (실제로는 더 복잡)
        if hand_strength == "Premium Pair":
            primary = "Bet/Raise"
            reasoning = "강한 핸드로 밸류를 얻어야 합니다"
            frequencies = {'fold': 0.0, 'call': 0.2, 'bet_raise': 0.8}
        elif hand_strength == "Premium Ace":
            primary = "Bet/Raise"
            reasoning = "프리미엄 핸드로 어그레시브하게 플레이"
            frequencies = {'fold': 0.05, 'call': 0.25, 'bet_raise': 0.7}
        else:
            primary = "Check/Call"
            reasoning = "약한 핸드로 팟 컨트롤"
            frequencies = {'fold': 0.4, 'call': 0.5, 'bet_raise': 0.1}
        
        return {
            'primary': primary,
            'reasoning': reasoning,
            'alternatives': ['Check/Fold', 'Bluff'],
            'frequencies': frequencies,
            'sizing': {'bet': '0.75 pot', 'raise': '2.5x'}
        }
    
    def _get_style_action_frequencies(self, style: PlayStyle, situation: Dict) -> Dict:
        """스타일별 액션 빈도"""
        profile = self.style_profiles[style]
        
        if style == PlayStyle.TIGHT_AGGRESSIVE:
            return {'fold': 0.4, 'call': 0.25, 'bet_raise': 0.35}
        elif style == PlayStyle.LOOSE_AGGRESSIVE:
            return {'fold': 0.25, 'call': 0.25, 'bet_raise': 0.5}
        elif style == PlayStyle.TIGHT_PASSIVE:
            return {'fold': 0.6, 'call': 0.35, 'bet_raise': 0.05}
        else:  # BALANCED
            return {'fold': 0.35, 'call': 0.35, 'bet_raise': 0.3}
    
    def _get_aggression_description(self, aggression_freq: float) -> str:
        """공격성 수준 설명"""
        if aggression_freq > 0.4:
            return "Very High"
        elif aggression_freq > 0.3:
            return "High"
        elif aggression_freq > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _assess_transition_difficulty(self, current: Dict, target: Dict) -> str:
        """스타일 전환 난이도 평가"""
        total_diff = abs(target['vpip'] - current.get('vpip', 0.25)) + \
                    abs(target['pfr'] - current.get('pfr', 0.18))
        
        if total_diff > 0.2:
            return "Hard"
        elif total_diff > 0.1:
            return "Medium"
        else:
            return "Easy"
    
    def _identify_key_factors(self, analysis: HandAnalysis) -> List[str]:
        """핵심 고려 요소 식별"""
        return [
            f"핸드 스트렝스: {analysis.hand_strength}",
            f"포지션: {analysis.position}",
            f"보드 텍스처: {analysis.board_texture}",
            f"상대방 타입: {analysis.opponent_type}"
        ]
    
    def _get_style_variations(self, analysis: HandAnalysis) -> Dict:
        """스타일별 플레이 차이"""
        return {
            'TAG': '보수적이지만 밸류 중심',
            'LAG': '어그레시브하고 블러프 많음',
            'Balanced': '수학적으로 최적화된 플레이'
        }
    
    def _extract_learning_points(self, analysis: HandAnalysis) -> List[str]:
        """학습 포인트 추출"""
        return [
            "포지션의 중요성을 인식하세요",
            "상대방 타입에 따라 전략을 조정하세요", 
            "팟 오즈를 항상 계산하세요",
            "보드 텍스처를 분석하세요"
        ]
    
    def _identify_common_mistakes(self, analysis: HandAnalysis) -> List[str]:
        """흔한 실수들"""
        return [
            "약한 핸드로 과도한 어그레시브니스",
            "상대방 타입 무시하고 동일한 플레이",
            "포지션 무시하고 모든 자리에서 같은 플레이",
            "팟 오즈 계산 없이 감정적 결정"
        ]


# 사용 예시
if __name__ == "__main__":
    guide_system = PokerPlayStyleGuide()
    
    print("🎯 포커 플레이 스타일 가이드 시스템")
    print("=" * 60)
    
    # 1. 특정 상황 분석
    situation_guide = guide_system.generate_situation_guide(
        hole_cards=['As', 'Kd'],
        board=['Qh', '7c', '2s'],
        position='BTN',
        stack_bb=65,
        pot_size=15,
        bet_to_call=8
    )
    
    print("📊 상황별 가이드:")
    print(f"핸드: {situation_guide['situation_summary']['hand']}")
    print(f"추천 액션: {situation_guide['recommendations']['primary_action']}")
    print(f"이유: {situation_guide['recommendations']['reasoning']}")
    
    # 2. 스타일별 비교
    comparison = guide_system.generate_style_comparison({
        'hand_strength': 'medium',
        'position': 'BTN',
        'board_texture': 'dry'
    })
    
    print(f"\n📈 스타일별 비교:")
    print(comparison.to_string(index=False))
    
    # 3. 학습 로드맵
    current_stats = {'vpip': 0.35, 'pfr': 0.15, 'aggression_factor': 1.8}
    roadmap = guide_system.create_learning_roadmap(current_stats, PlayStyle.TIGHT_AGGRESSIVE)
    
    print(f"\n🗺️ TAG 스타일 학습 로드맵:")
    print(f"예상 기간: {roadmap['expected_timeline']}")
    print(f"난이도: {roadmap['difficulty']}")
    
    for improvement in roadmap['improvements_needed']:
        print(f"• {improvement['area']}: {improvement['current']} → {improvement['target']}")
    
    print("\n✅ 플레이 스타일 가이드로는 완벽하게 활용 가능!")