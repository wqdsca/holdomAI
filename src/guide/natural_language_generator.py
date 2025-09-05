"""
포커 AI 자연어 생성 시스템
복잡한 포커 분석을 이해하기 쉬운 한국어로 설명
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ExplanationStyle(Enum):
    """설명 스타일"""
    BEGINNER = "beginner"      # 초보자용 (기본 용어 설명)
    INTERMEDIATE = "intermediate"  # 중급자용 (표준 설명)
    ADVANCED = "advanced"      # 고급자용 (전문 용어)
    CASUAL = "casual"          # 캐주얼 (친근한 톤)


class KoreanPokerExplainer:
    """한국어 포커 설명 생성기"""
    
    def __init__(self):
        self.templates = self._load_explanation_templates()
        self.poker_terms = self._load_poker_terminology()
        
    def _load_explanation_templates(self) -> Dict:
        """설명 템플릿 로드"""
        return {
            'hand_analysis': {
                'strong_hand': [
                    "{hand}는 매우 강한 핸드입니다. {reason}",
                    "이 상황에서 {hand}는 확실한 밸류 핸드예요. {reason}",
                    "{hand} - 이건 절대 놓칠 수 없는 핸드네요! {reason}"
                ],
                'medium_hand': [
                    "{hand}는 적당한 강도의 핸드입니다. {reason}",
                    "이 핸드는 상황에 따라 플레이가 달라질 수 있어요. {reason}",
                    "{hand} - 신중하게 접근해야 하는 핸드예요. {reason}"
                ],
                'weak_hand': [
                    "{hand}는 약한 핸드입니다. {reason}",
                    "이 핸드로는 조심스럽게 플레이해야 해요. {reason}",
                    "{hand} - 무리하지 말고 보수적으로 가세요. {reason}"
                ]
            },
            'position_analysis': {
                'early_position': [
                    "얼리 포지션에서는 더욱 타이트하게 플레이해야 합니다.",
                    "앞자리는 불리하니까 강한 핸드만 플레이하는 게 좋아요.",
                    "UTG에서는 정말 좋은 핸드가 아니면 폴드가 답입니다."
                ],
                'middle_position': [
                    "미들 포지션에서는 조금 더 넓은 레인지로 플레이할 수 있어요.",
                    "중간 자리니까 적당히 어그레시브하게 가도 됩니다.",
                    "MP에서는 상황을 보면서 유연하게 대응하세요."
                ],
                'late_position': [
                    "레이트 포지션의 장점을 최대한 활용하세요!",
                    "뒤에 앉아있으니 더 넓은 레인지로 플레이 가능해요.",
                    "버튼이나 컷오프에서는 어그레시브하게 공격해보세요."
                ]
            },
            'board_analysis': {
                'dry_board': [
                    "드라이한 보드라서 블러핑하기 좋은 상황이에요.",
                    "이런 보드에서는 C-bet이 자주 통합니다.",
                    "상대방도 맞은 게 별로 없을 가능성이 높아요."
                ],
                'wet_board': [
                    "보드가 위험해 보이네요. 조심스럽게 접근하세요.",
                    "드로우가 많은 보드라서 상대방이 부를 이유가 많아요.",
                    "이런 보드에서는 강하게 베팅해서 보호하거나 체크해서 안전하게 가세요."
                ],
                'paired_board': [
                    "페어드 보드는 항상 조심해야 해요.",
                    "상대방이 트립스나 풀하우스를 만들었을 수도 있거든요.",
                    "이런 보드에서는 과도한 블러핑보다는 밸류에 집중하세요."
                ]
            }
        }
    
    def _load_poker_terminology(self) -> Dict:
        """포커 용어 및 설명"""
        return {
            'beginner_terms': {
                'VPIP': 'VPIP는 자발적으로 돈을 넣는 비율이에요. 높을수록 많은 핸드를 플레이한다는 뜻',
                'PFR': 'PFR은 프리플랍에서 레이즈하는 비율입니다. 어그레시브한 정도를 나타내요',
                'C-bet': 'C-bet(컨티뉴에이션 베팅)은 프리플랍 레이저가 플랍에서도 계속 베팅하는 거예요',
                'TAG': 'TAG는 타이트-어그레시브의 줄임말로, 좋은 핸드만 플레이하되 어그레시브하게 하는 스타일',
                'LAG': 'LAG는 루즈-어그레시브로, 많은 핸드를 어그레시브하게 플레이하는 고수들의 스타일'
            },
            'hand_descriptions': {
                'premium_pair': '프리미엄 페어 (AA, KK, QQ)',
                'strong_ace': '강한 에이스 (AK, AQ)',
                'suited_connectors': '수티드 커넥터 (연결된 같은 무늬)',
                'broadway': '브로드웨이 카드 (T, J, Q, K, A)',
                'pocket_pair': '포켓 페어 (같은 숫자 두 장)'
            }
        }
    
    def generate_hand_explanation(
        self,
        hole_cards: List[str],
        board: List[str],
        position: str,
        analysis_result: Dict,
        style: ExplanationStyle = ExplanationStyle.INTERMEDIATE
    ) -> str:
        """핸드 상황에 대한 자연어 설명 생성"""
        
        explanation_parts = []
        
        # 1. 핸드 소개 및 기본 평가
        hand_intro = self._generate_hand_intro(hole_cards, analysis_result['hand_strength'])
        explanation_parts.append(hand_intro)
        
        # 2. 포지션 분석
        position_analysis = self._generate_position_analysis(position, style)
        explanation_parts.append(position_analysis)
        
        # 3. 보드 분석 (포스트플랍인 경우)
        if board:
            board_analysis = self._generate_board_analysis(board, style)
            explanation_parts.append(board_analysis)
        
        # 4. 추천 액션 및 이유
        action_explanation = self._generate_action_explanation(
            analysis_result['recommended_action'],
            analysis_result['reasoning'],
            style
        )
        explanation_parts.append(action_explanation)
        
        # 5. 대안 및 주의사항
        alternatives = self._generate_alternatives_explanation(
            analysis_result['alternative_plays'],
            style
        )
        explanation_parts.append(alternatives)
        
        # 6. 스타일별 차이점
        style_differences = self._generate_style_differences(analysis_result.get('style_variations', {}))
        explanation_parts.append(style_differences)
        
        return '\n\n'.join(explanation_parts)
    
    def generate_learning_explanation(
        self,
        current_stats: Dict,
        target_style: str,
        improvements: List[Dict],
        style: ExplanationStyle = ExplanationStyle.CASUAL
    ) -> str:
        """학습 로드맵 설명 생성"""
        
        explanations = []
        
        # 인트로
        intro = f"""
🎯 **포커 실력 향상 가이드**

안녕하세요! 현재 회원님의 플레이 스타일을 분석해보니, {target_style} 스타일로 발전시키면 
더 좋은 결과를 얻을 수 있을 것 같아요.
"""
        explanations.append(intro.strip())
        
        # 현재 상태 분석
        current_analysis = f"""
📊 **현재 플레이 스타일 분석**

• VPIP (플레이하는 핸드 비율): {current_stats.get('vpip', 0)*100:.1f}%
• PFR (프리플랍 레이즈 비율): {current_stats.get('pfr', 0)*100:.1f}%
• 어그레시브 팩터: {current_stats.get('aggression_factor', 0):.1f}

{self._analyze_current_tendencies(current_stats)}
"""
        explanations.append(current_analysis.strip())
        
        # 개선 사항
        improvement_text = "🚀 **개선이 필요한 부분**\n\n"
        
        for i, improvement in enumerate(improvements, 1):
            priority_emoji = "🔥" if improvement['priority'] == 'High' else "⭐"
            improvement_text += f"{priority_emoji} **{improvement['area']}**\n"
            improvement_text += f"   현재: {improvement['current']} → 목표: {improvement['target']}\n"
            improvement_text += f"   액션: {improvement['action']}\n\n"
        
        explanations.append(improvement_text.strip())
        
        # 실천 방법
        practice_methods = f"""
💡 **구체적인 실천 방법**

**1주차: 핸드 선택 개선**
- 포지션별 레인지 차트를 만들어서 참고하세요
- 게임 중에 "이 핸드를 정말 플레이해야 하나?" 자문하기
- 하루에 100핸드씩 플레이하며 핸드 선택 연습

**2-3주차: 어그레시브니스 조정**  
- 좋은 핸드를 가졌을 때는 벨류를 얻기 위해 베팅하기
- 블러핑할 때는 스토리가 있는지 생각해보기
- C-bet 빈도를 조금씩 늘려가기

**4주차 이후: 고급 개념 적용**
- 상대방 타입에 따른 전략 조정 연습  
- 베팅 사이징 최적화
- 핸드 리뷰를 통한 지속적 개선
"""
        explanations.append(practice_methods.strip())
        
        return '\n\n'.join(explanations)
    
    def generate_situation_explanation(
        self,
        situation: Dict,
        analysis: Dict,
        style: ExplanationStyle = ExplanationStyle.INTERMEDIATE
    ) -> str:
        """특정 상황에 대한 상세 설명"""
        
        explanation = f"""
🎴 **상황 분석: {situation['hand']} @ {situation['board']}**

**포지션**: {situation['position']} ({self._get_position_advantage(situation['position'])})
**스택**: {situation['stack']} ({self._get_stack_category_description(situation['stack'])})
**팟 오즈**: {situation['pot_odds']} ({self._interpret_pot_odds(situation['pot_odds'])})

---

🔍 **핸드 분석**

{analysis['hand_analysis']['strength']} - {self._elaborate_hand_strength(analysis['hand_analysis']['strength'])}

보드 상태: {analysis['hand_analysis']['board_texture']}
{self._elaborate_board_texture(analysis['hand_analysis']['board_texture'])}

---

🎯 **추천 전략**

**주요 액션**: {analysis['recommendations']['primary_action']}

**이유**: {analysis['recommendations']['reasoning']}

**대안들**:
{self._format_alternatives(analysis['recommendations']['alternatives'])}

---

📚 **학습 포인트**

{self._format_learning_points(analysis.get('learning_points', []))}

---

⚠️ **주의사항**

{self._format_common_mistakes(analysis.get('common_mistakes', []))}
"""
        
        return explanation.strip()
    
    def generate_style_comparison_explanation(self, comparison_df) -> str:
        """스타일별 비교 설명"""
        
        explanation = """
📊 **플레이 스타일별 비교 분석**

같은 상황이라도 플레이 스타일에 따라 완전히 다른 접근을 합니다.
각 스타일의 특징을 이해하고 본인에게 맞는 스타일을 찾아보세요.

"""
        
        for _, row in comparison_df.iterrows():
            style_name = row['Style']
            primary_action = row['Primary Action']
            aggression = row['Aggression Level']
            
            explanation += f"""
**{style_name}** (어그레시브 레벨: {aggression})
- 주요 액션: {primary_action}
- 폴드: {row['Fold %']} | 콜: {row['Call %']} | 베팅/레이즈: {row['Bet/Raise %']}
- {self._get_style_personality(style_name)}

"""
        
        explanation += """
💡 **어떤 스타일을 선택할까요?**

• **초보자**: Tight Aggressive (TAG) 추천 - 안정적이고 배우기 쉬워요
• **중급자**: Balanced 추천 - 상황에 따라 유연하게 대응 가능
• **고급자**: Loose Aggressive (LAG) 도전 - 최대 수익 가능하지만 어려워요
"""
        
        return explanation.strip()
    
    def _generate_hand_intro(self, hole_cards: List[str], hand_strength: str) -> str:
        """핸드 소개 생성"""
        hand_display = f"{hole_cards[0]} {hole_cards[1]}"
        
        strength_descriptions = {
            'Premium Pair': f"🔥 {hand_display} - 프리미엄 페어네요! 이런 핸드는 자주 오지 않으니 최대한 활용해야 해요.",
            'Premium Ace': f"⭐ {hand_display} - 강한 에이스 핸드입니다. 어그레시브하게 플레이할 가치가 있어요.",
            'Medium Pair': f"👍 {hand_display} - 중간 강도의 페어예요. 상황에 따라 신중하게 플레이하세요.",
            'Speculative Hand': f"🤔 {hand_display} - 투기적인 핸드네요. 포지션과 상황을 잘 고려해서 플레이하세요."
        }
        
        return strength_descriptions.get(hand_strength, f"{hand_display} - 이 핸드로는 조심스럽게 접근하세요.")
    
    def _generate_position_analysis(self, position: str, style: ExplanationStyle) -> str:
        """포지션 분석 생성"""
        position_explanations = {
            'UTG': "👥 **UTG (언더 더 건)**: 가장 먼저 액션해야 하는 불리한 자리입니다. 강한 핸드만 플레이하는 게 좋아요.",
            'MP': "👥 **MP (미들 포지션)**: 중간 자리로 적당한 수준의 어그레시브니스가 필요해요.",
            'CO': "👥 **CO (컷오프)**: 좋은 포지션이에요! 레인지를 조금 넓혀서 플레이할 수 있습니다.",
            'BTN': "👥 **BTN (버튼)**: 최고의 포지션! 가장 마지막에 액션하므로 어그레시브하게 갈 수 있어요.",
            'SB': "👥 **SB (스몰 블라인드)**: 이미 돈을 넣었지만 포지션이 안 좋아서 조심해야 해요.",
            'BB': "👥 **BB (빅 블라인드)**: 마지막에 액션할 수 있지만 포스트플랍에서는 불리합니다."
        }
        
        return position_explanations.get(position, f"👥 **포지션 {position}**: 포지션을 고려한 플레이가 필요해요.")
    
    def _generate_board_analysis(self, board: List[str], style: ExplanationStyle) -> str:
        """보드 분석 생성"""
        board_display = ' '.join(board)
        
        # 간단한 보드 분류
        suits = [card[1] for card in board[:3]]
        is_flush_draw = len(set(suits)) <= 2
        
        if is_flush_draw:
            return f"🌊 **보드: {board_display}** - 플러시 드로우가 있는 위험한 보드예요. 상대방이 부를 이유가 많아서 밸류베팅할 때는 크게, 블러핑할 때는 조심해야 해요."
        else:
            return f"🏜️ **보드: {board_display}** - 드라이한 보드네요. 상대방이 맞은 게 별로 없을 가능성이 높아서 블러핑하기 좋은 상황이에요."
    
    def _generate_action_explanation(self, action: str, reasoning: str, style: ExplanationStyle) -> str:
        """액션 추천 설명"""
        action_emojis = {
            'Bet/Raise': '🚀',
            'Check/Call': '👍', 
            'Fold': '❌',
            'Check/Fold': '⚠️'
        }
        
        emoji = action_emojis.get(action, '🎯')
        
        return f"""
{emoji} **추천 액션: {action}**

**이유**: {reasoning}

이 상황에서는 {action}이 가장 이론적으로 올바른 선택입니다. 
물론 상대방의 성향이나 게임 플로우에 따라 조정할 수 있어요.
"""
    
    def _generate_alternatives_explanation(self, alternatives: List[str], style: ExplanationStyle) -> str:
        """대안 설명"""
        if not alternatives:
            return ""
            
        alt_text = "🔄 **다른 옵션들**:\n\n"
        
        for alt in alternatives:
            alt_text += f"• **{alt}**: {self._get_alternative_reasoning(alt)}\n"
        
        return alt_text.strip()
    
    def _generate_style_differences(self, style_variations: Dict) -> str:
        """스타일별 차이점 설명"""
        if not style_variations:
            return ""
            
        return f"""
🎨 **스타일별 접근법**

• **TAG**: {style_variations.get('TAG', '보수적이지만 확실한 플레이')}
• **LAG**: {style_variations.get('LAG', '어그레시브하고 창의적인 플레이')} 
• **Balanced**: {style_variations.get('Balanced', '수학적으로 최적화된 플레이')}

각자의 성향과 실력에 맞는 스타일을 선택하세요!
"""
    
    def _analyze_current_tendencies(self, stats: Dict) -> str:
        """현재 경향 분석"""
        vpip = stats.get('vpip', 0)
        pfr = stats.get('pfr', 0)
        
        if vpip > 0.3:
            tendency = "다소 루즈한 편이에요. 핸드 선택을 조금 더 까다롭게 해보세요."
        elif vpip < 0.2:
            tendency = "매우 타이트하게 플레이하고 계시네요. 조금 더 다양한 핸드를 시도해도 좋을 것 같아요."
        else:
            tendency = "적당한 수준의 핸드 선택을 하고 계시네요."
            
        if pfr / max(vpip, 0.01) > 0.8:
            tendency += " 어그레시브한 성향이 강하네요!"
        elif pfr / max(vpip, 0.01) < 0.5:
            tendency += " 조금 더 어그레시브하게 플레이해도 좋을 것 같아요."
            
        return tendency
    
    def _get_position_advantage(self, position: str) -> str:
        """포지션 우위 설명"""
        advantages = {
            'UTG': '정보 부족, 불리함',
            'MP': '중간 정도 정보',
            'CO': '좋은 정보력',
            'BTN': '최고의 정보 우위',
            'SB': '돈은 넣었지만 불리한 포지션',
            'BB': '마지막 액션, 하지만 포스트플랍 불리'
        }
        return advantages.get(position, '일반적인 포지션')
    
    def _get_stack_category_description(self, stack: str) -> str:
        """스택 카테고리 설명"""
        if 'Short' in stack:
            return '숏스택 - 올인 압박 있음'
        elif 'Deep' in stack:
            return '딥스택 - 임플라이드 오즈 고려'
        else:
            return '미디엄스택 - 표준 플레이'
    
    def _interpret_pot_odds(self, pot_odds: str) -> str:
        """팟 오즈 해석"""
        try:
            odds_value = float(pot_odds.strip('%')) / 100
            if odds_value > 0.3:
                return '높은 팟 오즈 - 콜하기 유리'
            elif odds_value < 0.2:
                return '낮은 팟 오즈 - 강한 핸드 필요'
            else:
                return '적당한 팟 오즈 - 상황에 따라 결정'
        except:
            return '팟 오즈 고려 필요'
    
    def _elaborate_hand_strength(self, strength: str) -> str:
        """핸드 강도 상세 설명"""
        elaborations = {
            'Premium Pair': '이런 핸드는 프리플랍에서 거의 항상 레이즈해야 하고, 포스트플랍에서도 적극적으로 밸류를 얻어야 합니다.',
            'Premium Ace': '탑페어 이상을 만들 가능성이 높고, 넛츠를 만들 수도 있는 강력한 핸드입니다.',
            'Medium Pair': '셋을 만들면 매우 강하지만, 그렇지 않으면 신중하게 플레이해야 하는 핸드입니다.'
        }
        return elaborations.get(strength, '상황에 따라 강도가 달라질 수 있는 핸드입니다.')
    
    def _elaborate_board_texture(self, texture: str) -> str:
        """보드 텍스처 상세 설명"""
        elaborations = {
            'Dry (Rainbow)': '드로우가 별로 없어서 상대방이 강한 핸드를 가질 확률이 낮습니다. 블러핑하기 좋은 보드예요.',
            'Wet (Flush draws)': '플러시나 스트레이트 드로우가 많아서 상대방이 콜할 이유가 많습니다. 블러핑보다는 밸류에 집중하세요.',
            'Paired Board': '누군가 트립스나 풀하우스를 가졌을 수 있어서 항상 조심해야 하는 보드입니다.'
        }
        return elaborations.get(texture, '보드의 특성을 잘 파악하고 그에 맞는 전략을 세우세요.')
    
    def _format_alternatives(self, alternatives: List[str]) -> str:
        """대안 형식화"""
        if not alternatives:
            return "특별한 대안은 없습니다."
            
        formatted = ""
        for alt in alternatives:
            formatted += f"• {alt}: {self._get_alternative_reasoning(alt)}\n"
        return formatted.strip()
    
    def _get_alternative_reasoning(self, alternative: str) -> str:
        """대안 이유 설명"""
        reasonings = {
            'Check/Fold': '보수적인 접근으로, 확실하지 않을 때 손실을 최소화',
            'Bluff': '상대방이 약하다고 판단될 때 시도할 수 있는 공격적 플레이',
            'Call': '팟 오즈가 맞거나 드로우가 있을 때 고려',
            'Raise': '강한 핸드거나 블러프로 압박할 때 사용'
        }
        return reasonings.get(alternative, '상황에 따라 고려할 수 있는 옵션')
    
    def _format_learning_points(self, points: List[str]) -> str:
        """학습 포인트 형식화"""
        if not points:
            return "기본적인 전략을 충실히 따르세요."
            
        formatted = ""
        for i, point in enumerate(points, 1):
            formatted += f"{i}. {point}\n"
        return formatted.strip()
    
    def _format_common_mistakes(self, mistakes: List[str]) -> str:
        """흔한 실수 형식화"""
        if not mistakes:
            return "큰 실수는 없을 것 같아요!"
            
        formatted = ""
        for mistake in mistakes:
            formatted += f"❌ {mistake}\n"
        return formatted.strip()
    
    def _get_style_personality(self, style_name: str) -> str:
        """스타일 성격 설명"""
        personalities = {
            'Tight Aggressive': '신중하지만 기회가 오면 과감하게! 초보자에게 가장 추천하는 안정적인 스타일',
            'Loose Aggressive': '많은 핸드를 어그레시브하게! 고수들이 선호하지만 어려운 스타일',
            'Balanced': '수학적으로 완벽하게! GTO 기반의 이론적 최적 스타일',
            'Tight Passive': '매우 조심스럽게... 안전하지만 수익성이 낮은 스타일'
        }
        return personalities.get(style_name, '각자의 특색이 있는 스타일')


# 사용 예시
if __name__ == "__main__":
    explainer = KoreanPokerExplainer()
    
    # 상황별 설명 생성
    situation = {
        'hand': 'As Kd',
        'board': 'Qh 7c 2s',
        'position': 'BTN',
        'stack': '65.0 BB',
        'pot_odds': '25%'
    }
    
    analysis = {
        'hand_analysis': {
            'strength': 'Premium Ace',
            'board_texture': 'Dry (Rainbow)'
        },
        'recommendations': {
            'primary_action': 'Bet/Raise',
            'reasoning': '강한 핸드로 밸류를 얻어야 합니다',
            'alternatives': ['Check/Call', 'Bluff']
        },
        'learning_points': [
            '포지션의 중요성을 인식하세요',
            '보드 텍스처를 분석하세요'
        ],
        'common_mistakes': [
            '약한 핸드로 과도한 어그레시브니스'
        ]
    }
    
    explanation = explainer.generate_situation_explanation(situation, analysis)
    
    print("🎯 포커 AI 자연어 설명 예시")
    print("=" * 60)
    print(explanation)
    print("\n" + "=" * 60)
    print("✅ 완전히 한국어로 자세한 설명 가능!")