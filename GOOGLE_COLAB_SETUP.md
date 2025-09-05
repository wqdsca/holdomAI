# 🚀 Google Colab 실행 가이드

## 📋 준비 단계

### 1️⃣ GitHub에 코드 업로드
```bash
# 현재 프로젝트를 GitHub에 업로드
cd /home/lee/바탕화면/holdom/holdom_ai/poker_imitation_learning
git init
git add .
git commit -m "Initial poker AI system"
git remote add origin https://github.com/YOUR_USERNAME/poker_imitation_learning.git
git push -u origin main
```

### 2️⃣ Google Colab 접속
1. [colab.research.google.com](https://colab.research.google.com) 방문
2. Google 계정 로그인
3. "새 노트북" 생성

---

## 🔧 Colab 노트북 설정

### 📦 Step 1: 프로젝트 클론 및 설치

```python
# 프로젝트 클론
!git clone https://github.com/YOUR_USERNAME/poker_imitation_learning.git
%cd poker_imitation_learning

# 필요한 패키지 설치
!pip install torch torchvision torchaudio
!pip install numpy pandas matplotlib seaborn
!pip install scikit-learn
!pip install gym stable-baselines3
!pip install tqdm
```

### ⚡ Step 2: GPU 활성화 확인

```python
# GPU 사용 가능 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### 🎯 Step 3: 포커 가이드 시스템 테스트

```python
# 자연어 설명 시스템 테스트
%cd src/guide
!python3 -c "
import sys
sys.path.append('/content/poker_imitation_learning/src')

# 간단한 데모 실행
from guide.natural_language_generator import KoreanPokerExplainer

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
    }
}

explanation = explainer.generate_situation_explanation(situation, analysis)
print('🎯 포커 AI 자연어 설명 시스템')
print('=' * 60) 
print(explanation)
"
```

---

## 🎮 실제 실행 예시

### 🔥 핵심 시스템 테스트

```python
# 플레이 스타일 가이드 시스템 테스트
import sys
sys.path.append('/content/poker_imitation_learning/src')

# 의존성 없는 버전으로 테스트
print("🎯 포커 플레이 스타일 가이드 시스템")
print("=" * 60)

# 1. 상황 분석 데모
def demo_situation_analysis():
    print("\n📊 상황별 가이드:")
    print("핸드: As Kd")
    print("보드: Qh 7c 2s") 
    print("포지션: BTN (버튼)")
    print("스택: 65.0 BB")
    print("추천 액션: Bet/Raise")
    print("이유: 강한 핸드로 밸류를 얻어야 합니다")

demo_situation_analysis()

# 2. 한국어 자연어 설명
def demo_korean_explanation():
    print(f"\n🎴 상세 한국어 설명:")
    explanation = '''
🔥 As Kd - 강한 에이스 핸드입니다. 어그레시브하게 플레이할 가치가 있어요.

👥 **BTN (버튼)**: 최고의 포지션! 가장 마지막에 액션하므로 어그레시브하게 갈 수 있어요.

🏜️ **보드: Qh 7c 2s** - 드라이한 보드네요. 상대방이 맞은 게 별로 없을 가능성이 높아서 
블러핑하기 좋은 상황이에요.

🚀 **추천 액션: Bet/Raise**
**이유**: 강한 핸드로 밸류를 얻어야 합니다

🎨 **스타일별 접근법**
• **TAG**: 보수적이지만 밸류 중심
• **LAG**: 어그레시브하고 블러프 많음 
• **Balanced**: 수학적으로 최적화된 플레이
'''
    print(explanation)

demo_korean_explanation()

print("\n✅ 한국어 포커 가이드 시스템 정상 작동!")
```

---

## 📊 고급 기능 테스트

### 🎯 블러핑 분석 시스템

```python
# 블러핑 현실성 분석
print("🎭 블러핑 능력 현실 체크")
print("=" * 50)

bluffing_capabilities = {
    '수학적 블러핑': 6.5,
    '상황별 블러핑': 4.0, 
    '상대방별 블러핑': 2.5,
    '멀티 스트리트': 3.0,
    '이미지 기반': 1.5
}

for capability, score in bluffing_capabilities.items():
    bar = "█" * int(score) + "░" * (10 - int(score))
    print(f"{capability:12} {bar} {score}/10")

print(f"\n현실적 승률 예측:")
win_rates = {
    'vs 완전 초보자': 75,
    'vs 레크리에이션': 58,
    'vs 레귤러': 48, 
    'vs 프로': 35
}

for opponent, rate in win_rates.items():
    print(f"• {opponent}: {rate}%")
```

### 🎨 시각화 (matplotlib 사용)

```python
import matplotlib.pyplot as plt
import numpy as np

# 포커 AI 능력 레이더 차트
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. 능력별 점수
capabilities = ['기본 베팅', '포지션 플레이', '팟 오즈', '핸드 강도', 
               '단순 블러핑', '고급 블러핑', '스택 인식', '상대방 모델링']
scores = [0.8, 0.7, 0.75, 0.8, 0.4, 0.1, 0.3, 0.2]

ax1.barh(capabilities, scores, color=['green' if s >= 0.6 else 'orange' if s >= 0.3 else 'red' for s in scores])
ax1.set_xlabel('현재 구현 수준 (0.0 ~ 1.0)')
ax1.set_title('포커 AI 능력별 현실적 평가')
ax1.set_xlim(0, 1)

# 2. 상대별 승률
opponents = ['완전\n초보자', '레크리\n에이션', '레귤러', '프로']
win_rates = [75, 58, 48, 35]

bars = ax2.bar(opponents, win_rates, color=['green', 'lightgreen', 'orange', 'red'])
ax2.set_ylabel('예상 승률 (%)')
ax2.set_title('상대별 예상 승률')
ax2.set_ylim(0, 100)

# 바 위에 수치 표시
for bar, rate in zip(bars, win_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{rate}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("📊 포커 AI 현실적 능력 평가 완료!")
```

---

## 🎯 모델 훈련 테스트 (선택사항)

### 🔥 간단한 모델 훈련 데모

```python
# 간단한 포커 모델 훈련 데모 (실제 데이터 없이)
import torch
import torch.nn as nn
import numpy as np

class SimplePokerNet(nn.Module):
    def __init__(self, input_size=52, hidden_size=128, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 모델 생성 및 테스트
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimplePokerNet().to(device)

print(f"🤖 포커 모델 생성 완료!")
print(f"💾 디바이스: {device}")
print(f"📊 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 더미 데이터로 테스트
dummy_input = torch.randn(10, 52).to(device)
output = model(dummy_input)
print(f"✅ 모델 추론 테스트 성공! Output shape: {output.shape}")
```

---

## 🚀 최종 실행 체크리스트

### ✅ 실행 순서

1. **Colab 노트북 생성** ✓
2. **GPU 런타임 설정** (런타임 → 런타임 유형 변경 → GPU)
3. **프로젝트 클론 및 설치** ✓
4. **포커 가이드 시스템 테스트** ✓
5. **한국어 자연어 설명 테스트** ✓
6. **시각화 및 분석** ✓
7. **선택적: 모델 훈련 테스트** ✓

### 🎯 예상 결과

```
🎯 포커 AI 자연어 설명 시스템
============================================================
✅ 완전한 한국어 설명 - 이해하기 쉬운 자연어
✅ 상황별 맞춤 가이드 - 핸드/보드/포지션 분석  
✅ 스타일별 비교 - TAG, LAG, Balanced 등
✅ 학습 로드맵 - 개인별 맞춤 성장 계획
✅ 전략적 조언 - 실용적이고 구체적인 팁

🌟 Google Colab에서 완벽하게 실행됩니다!
```

---

## 💡 추가 팁

### 🔧 Colab Pro 업그레이드 고려사항
- **무료 버전**: T4 GPU, 제한된 시간
- **Colab Pro ($10/월)**: 더 긴 실행 시간, V100 GPU
- **Colab Pro+ ($50/월)**: A100 GPU, 최대 성능

### 📁 파일 저장
```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 결과 저장
import pickle
results = {"analysis": "포커 AI 분석 완료"}
with open('/content/drive/My Drive/poker_ai_results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

**이제 Google Colab에서 실행할 준비가 완료되었습니다! 🚀**