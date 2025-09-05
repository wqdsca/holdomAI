"""
포커 정확도 심화 분석
실제 포커 데이터의 특성과 예상 성능을 구체적으로 분석
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

class PokerAccuracyAnalyzer:
    """포커 AI 정확도 심화 분석기"""
    
    def __init__(self):
        # 실제 포커 통계 (온라인 포커 기준)
        self.real_poker_stats = {
            'vpip_by_position': {
                'UTG': 0.12, 'MP': 0.16, 'CO': 0.25, 'BTN': 0.35, 'SB': 0.28, 'BB': 0.22
            },
            'pfr_by_position': {
                'UTG': 0.09, 'MP': 0.12, 'CO': 0.18, 'BTN': 0.28, 'SB': 0.15, 'BB': 0.08  
            },
            'action_frequency': {
                'fold': 0.65, 'check': 0.08, 'call': 0.15, 'bet': 0.07, 'raise': 0.04, 'all_in': 0.01
            },
            'street_complexity': {
                'preflop': 0.3,  # 상대적으로 단순
                'flop': 0.8,     # 복잡도 급증  
                'turn': 0.9,     # 매우 복잡
                'river': 1.0     # 최고 복잡도
            }
        }
        
        # 프로 플레이어vs아마추어 차이
        self.skill_differences = {
            'pro': {
                'decision_consistency': 0.85,  # 같은 상황에서 같은 결정
                'optimal_frequency': 0.78,     # GTO에 가까운 빈도
                'bet_sizing_accuracy': 0.82,   # 적절한 베팅 사이즈
                'position_awareness': 0.90     # 포지션 고려
            },
            'amateur': {
                'decision_consistency': 0.45,
                'optimal_frequency': 0.35,
                'bet_sizing_accuracy': 0.28,
                'position_awareness': 0.40
            }
        }
    
    def analyze_learning_difficulty(self) -> Dict:
        """포커 학습의 어려움 분석"""
        
        difficulties = {
            'preflop': {
                'complexity': 0.3,
                'optimal_accuracy_ceiling': 0.85,  # 이론적 최대
                'realistic_ai_accuracy': 0.78,     # AI 예상 달성
                'factors': [
                    '상대적으로 단순한 의사결정',
                    '카드 조합이 정해진 패턴',
                    '포지션별 전략이 명확',
                    '충분한 데이터로 학습 가능'
                ]
            },
            'flop': {
                'complexity': 0.8,
                'optimal_accuracy_ceiling': 0.65,
                'realistic_ai_accuracy': 0.62,
                'factors': [
                    '보드 텍스처의 다양성',
                    '드로우와 메이드 핸드의 복합',
                    '팟 오즈 계산의 복잡성',
                    '상대방 레인지 추정 필요'
                ]
            },
            'turn': {
                'complexity': 0.9,
                'optimal_accuracy_ceiling': 0.58,
                'realistic_ai_accuracy': 0.55,
                'factors': [
                    '복잡한 에퀴티 계산',
                    '임플라이드 오즈 고려',
                    '리버 계획 수립 필요',
                    '상대방의 턴 전략 고려'
                ]
            },
            'river': {
                'complexity': 1.0,
                'optimal_accuracy_ceiling': 0.50,
                'realistic_ai_accuracy': 0.48,
                'factors': [
                    '완전한 정보 부족 (상대 카드)',
                    '복잡한 심리전 요소',
                    '블러핑 빈도 최적화',
                    '상대방의 텔과 패턴'
                ]
            }
        }
        
        return difficulties
    
    def calculate_realistic_accuracy(self, model_size: str = 'medium') -> Dict:
        """모델 크기별 현실적 정확도 계산"""
        
        base_accuracies = {
            'small': 0.62,
            'medium': 0.68, 
            'large': 0.73,
            'xlarge': 0.76
        }
        
        base_acc = base_accuracies[model_size]
        difficulties = self.analyze_learning_difficulty()
        
        # 스트리트별 정확도 계산
        street_accuracies = {}
        for street, diff_info in difficulties.items():
            # 모델 크기에 따른 보정
            model_boost = (base_acc - 0.62) * 0.5  # 모델이 클수록 복잡한 상황도 잘 처리
            street_acc = diff_info['realistic_ai_accuracy'] + model_boost
            street_accuracies[street] = min(street_acc, diff_info['optimal_accuracy_ceiling'])
        
        # 전체 정확도 (스트리트별 가중 평균)
        street_weights = {'preflop': 0.4, 'flop': 0.3, 'turn': 0.2, 'river': 0.1}  # 각 스트리트 빈도
        overall_accuracy = sum(street_accuracies[street] * weight 
                              for street, weight in street_weights.items())
        
        return {
            'overall_accuracy': overall_accuracy,
            'street_accuracies': street_accuracies,
            'model_size': model_size,
            'expected_performance': self._interpret_accuracy(overall_accuracy)
        }
    
    def _interpret_accuracy(self, accuracy: float) -> Dict:
        """정확도를 실제 성능으로 해석"""
        
        if accuracy >= 0.75:
            level = "전문가 수준"
            description = "대부분의 아마추어를 상대로 수익 가능"
            win_rate_vs_amateur = 0.65
            win_rate_vs_semi_pro = 0.52
        elif accuracy >= 0.68:
            level = "중급 수준" 
            description = "초보자 상대로는 승률 높음, 고수 상대로는 어려움"
            win_rate_vs_amateur = 0.58
            win_rate_vs_semi_pro = 0.48
        elif accuracy >= 0.62:
            level = "초보 탈출 수준"
            description = "기본적인 전략 이해, 꾸준한 학습 필요"  
            win_rate_vs_amateur = 0.52
            win_rate_vs_semi_pro = 0.45
        else:
            level = "초보 수준"
            description = "더 많은 학습과 개선 필요"
            win_rate_vs_amateur = 0.45
            win_rate_vs_semi_pro = 0.40
        
        return {
            'level': level,
            'description': description,
            'win_rate_vs_amateur': win_rate_vs_amateur,
            'win_rate_vs_semi_pro': win_rate_vs_semi_pro,
            'hourly_bb_expectation': self._calculate_hourly_bb(win_rate_vs_amateur),
            'bankroll_requirement': self._calculate_bankroll_requirement(win_rate_vs_amateur)
        }
    
    def _calculate_hourly_bb(self, win_rate: float) -> float:
        """시간당 빅블라인드 기댓값 계산"""
        # 간단한 추정식 (실제로는 더 복잡함)
        if win_rate > 0.55:
            return (win_rate - 0.5) * 40  # 40핸드/시간 가정
        else:
            return (win_rate - 0.5) * 35
    
    def _calculate_bankroll_requirement(self, win_rate: float) -> int:
        """필요한 뱅크롤 (빅블라인드 단위)"""
        if win_rate > 0.55:
            return 20  # 20 바이인
        elif win_rate > 0.52:
            return 30
        else:
            return 50  # 더 큰 분산 대비
    
    def compare_ai_vs_humans(self) -> Dict:
        """AI vs 인간 플레이어 비교"""
        
        comparison = {
            'strengths_ai': [
                '일관된 의사결정 (감정 없음)',
                '정확한 확률 계산',  
                '피로도 없는 장시간 플레이',
                '틸트 없음 (나쁜 비트 후에도 냉정)',
                '대량 데이터 기반 패턴 학습',
                '수학적 최적화된 베팅 사이즈'
            ],
            'weaknesses_ai': [
                '상대방 심리 읽기 불가',
                '동적 조정 능력 부족',
                '창의적/예상외 플레이 어려움',
                '상황별 메타게임 적응 한계',
                '복잡한 다중 레벨 씽킹 부족',
                '실시간 상대방 조정 불가'
            ],
            'strengths_human': [
                '심리전과 블러핑',
                '상대방 패턴 빠른 파악',
                '창의적이고 예상외 플레이', 
                '동적 전략 조정',
                '테이블 이미지 활용',
                '메타게임 이해'
            ],
            'weaknesses_human': [
                '감정적 결정 (틸트)',
                '계산 실수',
                '피로도와 집중력 저하',
                '일관성 부족',
                '편향과 고정관념',
                '복잡한 수학 계산 어려움'
            ]
        }
        
        return comparison
    
    def estimate_real_world_performance(self, model_accuracy: float) -> Dict:
        """실제 환경에서의 성능 예측"""
        
        # 환경별 성능 보정 계수
        environment_factors = {
            'online_micro': 1.0,      # 1-5NL, 기준점
            'online_low': 0.92,       # 10-25NL, 약간 어려움
            'online_mid': 0.85,       # 50-100NL, 더 어려움  
            'online_high': 0.75,      # 200NL+, 매우 어려움
            'live_low': 0.88,         # 1/2, 2/5 라이브
            'live_mid': 0.80,         # 5/10+ 라이브
            'tournament': 0.70,       # 토너먼트 (ICM 고려)
        }
        
        results = {}
        for env, factor in environment_factors.items():
            adjusted_acc = model_accuracy * factor
            
            # 승률로 변환 (매우 단순한 추정)
            if adjusted_acc > 0.65:
                win_rate = 0.52 + (adjusted_acc - 0.65) * 0.5
            elif adjusted_acc > 0.60:  
                win_rate = 0.50 + (adjusted_acc - 0.60) * 0.4
            else:
                win_rate = 0.45 + (adjusted_acc - 0.55) * 0.5
                
            results[env] = {
                'adjusted_accuracy': adjusted_acc,
                'win_rate': max(0.40, min(0.65, win_rate)),  # 40-65% 범위로 제한
                'bb_per_hour': self._calculate_hourly_bb(win_rate),
                'variance': self._estimate_variance(win_rate)
            }
        
        return results
    
    def _estimate_variance(self, win_rate: float) -> float:
        """분산 추정 (bb/100hands 단위)"""
        base_variance = 80  # 일반적인 온라인 포커 분산
        
        if win_rate > 0.55:
            return base_variance * 1.2  # 어그레시브 플레이로 분산 증가
        elif win_rate < 0.48:
            return base_variance * 0.8   # 타이트 플레이로 분산 감소
        else:
            return base_variance
    
    def create_performance_report(self, model_size: str = 'medium') -> str:
        """종합 성능 리포트 생성"""
        
        accuracy_info = self.calculate_realistic_accuracy(model_size)
        real_world = self.estimate_real_world_performance(accuracy_info['overall_accuracy'])
        ai_vs_human = self.compare_ai_vs_humans()
        
        report = f"""
🎯 포커 AI 성능 분석 리포트 ({model_size.upper()} 모델)
{'='*60}

📊 예상 정확도
• 전체 정확도: {accuracy_info['overall_accuracy']*100:.1f}%
• 프리플랍: {accuracy_info['street_accuracies']['preflop']*100:.1f}%
• 플랍: {accuracy_info['street_accuracies']['flop']*100:.1f}%  
• 턴: {accuracy_info['street_accuracies']['turn']*100:.1f}%
• 리버: {accuracy_info['street_accuracies']['river']*100:.1f}%

🏆 실력 평가
• 수준: {accuracy_info['expected_performance']['level']}
• 설명: {accuracy_info['expected_performance']['description']}
• vs 아마추어 승률: {accuracy_info['expected_performance']['win_rate_vs_amateur']*100:.1f}%
• vs 준전문가 승률: {accuracy_info['expected_performance']['win_rate_vs_semi_pro']*100:.1f}%

💰 수익성 분석 (환경별)
"""
        
        for env, perf in real_world.items():
            env_name = {
                'online_micro': '온라인 마이크로 (1-5NL)',
                'online_low': '온라인 로우 (10-25NL)', 
                'online_mid': '온라인 미드 (50-100NL)',
                'online_high': '온라인 하이 (200NL+)',
                'live_low': '라이브 로우 (1/2, 2/5)',
                'live_mid': '라이브 미드 (5/10+)',
                'tournament': '토너먼트'
            }.get(env, env)
            
            report += f"• {env_name}:\n"
            report += f"  - 승률: {perf['win_rate']*100:.1f}%\n"
            report += f"  - 시간당 BB: {perf['bb_per_hour']:.1f}\n"
            report += f"  - 분산: {perf['variance']:.0f} bb/100hands\n\n"
        
        report += f"""
🤖 AI vs 인간 비교

AI의 강점:
{chr(10).join(f"• {strength}" for strength in ai_vs_human['strengths_ai'])}

AI의 약점:  
{chr(10).join(f"• {weakness}" for weakness in ai_vs_human['weaknesses_ai'])}

💡 활용 권장사항
• 학습 도구: 핸드 분석 및 전략 연구에 활용
• 연습 상대: 기본 전략 습득을 위한 상대
• 베이스라인: 자신의 플레이와 비교 분석
• 주의사항: 실제 고수 상대로는 한계 존재

⚠️ 현실적 기대치 조정 필요
포커는 불완전 정보 게임으로, 68.5% 정확도도 상당히 높은 수준입니다.
실제 프로도 같은 상황에서 다른 결정을 내리는 경우가 많으며,
'정답'이 명확하지 않은 상황이 빈번합니다.
"""
        
        return report


if __name__ == "__main__":
    analyzer = PokerAccuracyAnalyzer()
    
    # 모델별 성능 분석
    for model_size in ['small', 'medium', 'large']:
        print(analyzer.create_performance_report(model_size))
        print("\n" + "="*80 + "\n")