"""
RTX 3080 Training Analysis for Poker Imitation Learning
메모리 사용량, 훈련 시간, 예상 정확도 분석
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Tuple
import matplotlib.pyplot as plt

class RTX3080TrainingAnalyzer:
    """RTX 3080에서의 훈련 가능성 분석"""
    
    def __init__(self):
        # RTX 3080 사양
        self.gpu_specs = {
            'vram_gb': 10,
            'cuda_cores': 8704,
            'tensor_cores': 272,
            'base_clock_mhz': 1440,
            'boost_clock_mhz': 1710,
            'memory_bandwidth_gb_s': 760,
            'fp32_tflops': 29.77,
            'tensor_tflops': 238  # Mixed precision
        }
        
        # 모델 구성
        self.model_configs = {
            'small': {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 3,
                'parameters': 2.1e6,
                'memory_mb': 25
            },
            'medium': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 4,
                'parameters': 8.4e6,
                'memory_mb': 85
            },
            'large': {
                'd_model': 512,
                'n_heads': 8,
                'n_layers': 6,
                'parameters': 33.6e6,
                'memory_mb': 320
            },
            'xlarge': {
                'd_model': 1024,
                'n_heads': 16,
                'n_layers': 8,
                'parameters': 134.4e6,
                'memory_mb': 1280
            }
        }
        
        # 데이터셋 정보
        self.dataset_info = {
            'total_hands': 21_605_687,
            'avg_actions_per_hand': 8.5,
            'total_actions': 21_605_687 * 8.5,
            'feature_dim': 800,
            'estimated_size_gb': 45.2
        }
    
    def calculate_memory_usage(
        self,
        model_config: str,
        batch_size: int,
        sequence_length: int = 20
    ) -> Dict:
        """메모리 사용량 계산"""
        
        config = self.model_configs[model_config]
        
        # 모델 파라미터 메모리 (FP32)
        model_memory = config['parameters'] * 4 / (1024**2)  # MB
        
        # Optimizer 메모리 (Adam: 2x parameters for momentum + variance)
        optimizer_memory = model_memory * 2
        
        # 배치 데이터 메모리
        input_size = batch_size * sequence_length * self.dataset_info['feature_dim'] * 4 / (1024**2)
        
        # Gradient 메모리
        gradient_memory = model_memory
        
        # Activation 메모리 (추정치)
        activation_memory = batch_size * sequence_length * config['d_model'] * config['n_layers'] * 4 / (1024**2)
        
        total_memory_mb = (
            model_memory + 
            optimizer_memory + 
            input_size + 
            gradient_memory + 
            activation_memory + 
            200  # 기타 오버헤드
        )
        
        return {
            'model_mb': model_memory,
            'optimizer_mb': optimizer_memory,
            'input_mb': input_size,
            'gradient_mb': gradient_memory,
            'activation_mb': activation_memory,
            'total_mb': total_memory_mb,
            'total_gb': total_memory_mb / 1024,
            'gpu_utilization': (total_memory_mb / 1024) / self.gpu_specs['vram_gb']
        }
    
    def estimate_training_time(
        self,
        model_config: str,
        batch_size: int,
        num_epochs: int,
        data_fraction: float = 1.0
    ) -> Dict:
        """훈련 시간 추정"""
        
        config = self.model_configs[model_config]
        total_samples = int(self.dataset_info['total_actions'] * data_fraction)
        
        # 배치당 연산량 추정 (FLOPS)
        sequence_length = 20
        d_model = config['d_model']
        n_heads = config['n_heads']
        n_layers = config['n_layers']
        
        # Transformer 연산량 (간단한 추정)
        attention_flops = batch_size * sequence_length**2 * d_model * n_heads * n_layers * 2
        ffn_flops = batch_size * sequence_length * d_model * (d_model * 4) * n_layers * 2
        output_flops = batch_size * d_model * 6 * 2  # 6 actions
        
        total_flops_per_batch = attention_flops + ffn_flops + output_flops
        
        # 배치 수 계산
        batches_per_epoch = total_samples // batch_size
        total_batches = batches_per_epoch * num_epochs
        total_flops = total_batches * total_flops_per_batch
        
        # 처리 시간 추정 (Mixed Precision 사용)
        effective_tflops = self.gpu_specs['tensor_tflops'] * 0.6  # 실제 효율성 고려
        training_time_hours = (total_flops / 1e12) / effective_tflops / 3600
        
        # I/O 오버헤드 추가 (20%)
        total_time_hours = training_time_hours * 1.2
        
        return {
            'total_samples': total_samples,
            'batches_per_epoch': batches_per_epoch,
            'total_batches': total_batches,
            'total_flops': total_flops,
            'compute_time_hours': training_time_hours,
            'total_time_hours': total_time_hours,
            'time_per_epoch_minutes': (total_time_hours * 60) / num_epochs,
            'samples_per_second': total_samples / (total_time_hours * 3600)
        }
    
    def predict_accuracy(
        self,
        model_config: str,
        data_fraction: float = 1.0,
        training_quality: str = 'good'  # 'poor', 'fair', 'good', 'excellent'
    ) -> Dict:
        """예상 정확도 분석"""
        
        config = self.model_configs[model_config]
        
        # 기본 정확도 (모델 크기에 따라)
        base_accuracies = {
            'small': 0.62,
            'medium': 0.68,
            'large': 0.73,
            'xlarge': 0.76
        }
        
        base_acc = base_accuracies[model_config]
        
        # 데이터 양에 따른 보정
        data_boost = min(0.08 * np.log10(data_fraction * 10), 0.08)
        
        # 훈련 품질에 따른 보정
        quality_multipliers = {
            'poor': 0.85,
            'fair': 0.92,
            'good': 1.0,
            'excellent': 1.05
        }
        
        # 최종 정확도
        final_accuracy = (base_acc + data_boost) * quality_multipliers[training_quality]
        
        # 액션별 세부 정확도 추정
        action_accuracies = {
            'fold': final_accuracy + 0.05,  # 가장 쉬운 액션
            'check': final_accuracy + 0.02,
            'call': final_accuracy,
            'bet': final_accuracy - 0.03,
            'raise': final_accuracy - 0.05,
            'all_in': final_accuracy - 0.08  # 가장 어려운 액션
        }
        
        # 포커 스타일 메트릭 추정
        estimated_metrics = {
            'overall_accuracy': final_accuracy,
            'action_accuracies': action_accuracies,
            'vpip': 0.22,  # 일반적인 타이트-어그레시브
            'pfr': 0.16,
            'aggression_factor': 2.8,
            'bet_sizing_mae': 0.15,  # 팟 대비
            'convergence_epochs': max(10, 80 - config['parameters'] / 1e6)
        }
        
        return estimated_metrics
    
    def recommend_configuration(self) -> Dict:
        """최적 구성 추천"""
        
        print("\n" + "="*70)
        print("RTX 3080 포커 AI 훈련 분석 결과")
        print("="*70)
        
        results = {}
        
        # 각 모델 구성별 분석
        for model_name, config in self.model_configs.items():
            print(f"\n🔍 {model_name.upper()} 모델 분석:")
            
            # 최적 배치 사이즈 찾기
            optimal_batch = self.find_optimal_batch_size(model_name)
            
            # 메모리 사용량
            memory_info = self.calculate_memory_usage(model_name, optimal_batch)
            
            # 훈련 시간 (100 에포크, 전체 데이터)
            time_info = self.estimate_training_time(model_name, optimal_batch, 100, 1.0)
            
            # 정확도 예측
            accuracy_info = self.predict_accuracy(model_name, 1.0, 'good')
            
            # 실용성 점수 계산
            practicality_score = self.calculate_practicality_score(
                memory_info, time_info, accuracy_info
            )
            
            results[model_name] = {
                'optimal_batch_size': optimal_batch,
                'memory_usage': memory_info,
                'training_time': time_info,
                'accuracy': accuracy_info,
                'practicality_score': practicality_score
            }
            
            # 출력
            print(f"   📊 파라미터: {config['parameters']/1e6:.1f}M")
            print(f"   🔋 메모리 사용: {memory_info['total_gb']:.1f}GB ({memory_info['gpu_utilization']*100:.1f}%)")
            print(f"   ⏰ 훈련 시간: {time_info['total_time_hours']:.1f}시간")
            print(f"   🎯 예상 정확도: {accuracy_info['overall_accuracy']*100:.1f}%")
            print(f"   ⭐ 실용성 점수: {practicality_score:.1f}/10")
            
            # 가능/불가능 판정
            if memory_info['gpu_utilization'] > 0.95:
                print(f"   ❌ 메모리 부족으로 훈련 불가능")
            elif time_info['total_time_hours'] > 72:
                print(f"   ⚠️  훈련 시간이 너무 길어 비실용적")
            else:
                print(f"   ✅ 훈련 가능")
        
        # 최고 추천 구성
        best_config = max(results.keys(), key=lambda k: results[k]['practicality_score'])
        
        print(f"\n🏆 최적 추천 구성: {best_config.upper()}")
        print(f"   배치 사이즈: {results[best_config]['optimal_batch_size']}")
        print(f"   예상 훈련 시간: {results[best_config]['training_time']['total_time_hours']:.1f}시간")
        print(f"   예상 정확도: {results[best_config]['accuracy']['overall_accuracy']*100:.1f}%")
        
        return results
    
    def find_optimal_batch_size(self, model_config: str) -> int:
        """메모리 제약 하에서 최적 배치 사이즈 찾기"""
        batch_sizes = [8, 16, 32, 64, 128, 256]
        
        for batch_size in reversed(batch_sizes):
            memory_info = self.calculate_memory_usage(model_config, batch_size)
            if memory_info['gpu_utilization'] <= 0.85:  # 85% 이하 사용
                return batch_size
        
        return 8  # 최소값
    
    def calculate_practicality_score(
        self,
        memory_info: Dict,
        time_info: Dict,
        accuracy_info: Dict
    ) -> float:
        """실용성 점수 계산 (0-10)"""
        
        # 메모리 점수 (0-3)
        if memory_info['gpu_utilization'] > 0.95:
            memory_score = 0
        elif memory_info['gpu_utilization'] > 0.85:
            memory_score = 1
        elif memory_info['gpu_utilization'] > 0.70:
            memory_score = 2
        else:
            memory_score = 3
        
        # 시간 점수 (0-3)
        hours = time_info['total_time_hours']
        if hours > 72:
            time_score = 0
        elif hours > 36:
            time_score = 1
        elif hours > 12:
            time_score = 2
        else:
            time_score = 3
        
        # 정확도 점수 (0-4)
        acc = accuracy_info['overall_accuracy']
        if acc > 0.75:
            acc_score = 4
        elif acc > 0.70:
            acc_score = 3
        elif acc > 0.65:
            acc_score = 2
        elif acc > 0.60:
            acc_score = 1
        else:
            acc_score = 0
        
        return memory_score + time_score + acc_score
    
    def create_training_schedule(self, model_config: str) -> Dict:
        """단계별 훈련 스케줄 생성"""
        
        # 점진적 훈련 전략
        schedule = {
            'phase_1': {
                'description': '작은 데이터셋으로 빠른 프로토타입',
                'data_fraction': 0.01,
                'epochs': 20,
                'batch_size': self.find_optimal_batch_size(model_config),
                'learning_rate': 1e-3
            },
            'phase_2': {
                'description': '중간 데이터셋으로 검증',
                'data_fraction': 0.1,
                'epochs': 50,
                'batch_size': self.find_optimal_batch_size(model_config),
                'learning_rate': 5e-4
            },
            'phase_3': {
                'description': '전체 데이터셋으로 최종 훈련',
                'data_fraction': 1.0,
                'epochs': 100,
                'batch_size': self.find_optimal_batch_size(model_config),
                'learning_rate': 1e-4
            }
        }
        
        total_time = 0
        for phase_name, phase in schedule.items():
            time_info = self.estimate_training_time(
                model_config, 
                phase['batch_size'], 
                phase['epochs'], 
                phase['data_fraction']
            )
            phase['estimated_hours'] = time_info['total_time_hours']
            total_time += phase['estimated_hours']
        
        schedule['total_time_hours'] = total_time
        schedule['total_time_days'] = total_time / 24
        
        return schedule


if __name__ == "__main__":
    analyzer = RTX3080TrainingAnalyzer()
    
    # 분석 실행
    results = analyzer.recommend_configuration()
    
    # 추천 구성에 대한 상세 스케줄
    best_model = 'medium'  # 일반적으로 가장 실용적
    schedule = analyzer.create_training_schedule(best_model)
    
    print(f"\n📅 {best_model.upper()} 모델 훈련 스케줄:")
    for phase_name, phase in schedule.items():
        if phase_name.startswith('phase'):
            print(f"   {phase_name}: {phase['description']}")
            print(f"      데이터: {phase['data_fraction']*100:.1f}%, 에포크: {phase['epochs']}")
            print(f"      예상 시간: {phase['estimated_hours']:.1f}시간")
    
    print(f"\n총 훈련 시간: {schedule['total_time_hours']:.1f}시간 ({schedule['total_time_days']:.1f}일)")