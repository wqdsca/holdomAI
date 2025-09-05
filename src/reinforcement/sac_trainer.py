"""
SAC (Soft Actor-Critic) 트레이너
포커 환경에서 강화학습을 통한 정책 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import wandb
from pathlib import Path
from tqdm import tqdm

from .poker_env import PokerEnvironment, PokerMultiAgentWrapper


class ReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
        
    def __len__(self):
        return len(self.buffer)


class PokerSACActor(nn.Module):
    """SAC Actor Network"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 공통 특징 추출
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 정책 헤드 (확률 출력)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # 온도 파라미터 (학습 가능)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, state):
        """정방향 전파"""
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        return action_logits
    
    def sample_action(self, state):
        """액션 샘플링"""
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Gumbel-Softmax를 사용한 연속 액션 근사
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, action_probs
    
    def evaluate_action(self, state, action):
        """액션 평가"""
        action_logits = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        
        dist = torch.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, action_probs


class PokerSACCritic(nn.Module):
    """SAC Critic Network (Q-function)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 네트워크
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 네트워크 (Double Q-Learning)
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        """Q값 계산"""
        # 액션을 원-핫 벡터로 변환
        if len(action.shape) == 1:
            action_onehot = F.one_hot(action, num_classes=10).float()  # action_dim=10
        else:
            action_onehot = action
            
        x = torch.cat([state, action_onehot], dim=-1)
        
        q1 = self.q1_network(x)
        q2 = self.q2_network(x)
        
        return q1, q2


class PokerSACTrainer:
    """SAC 트레이너"""
    
    def __init__(
        self,
        state_dim: int = 800,
        action_dim: int = 10,  # fold, check, call, 6 bet sizes, all-in
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # 네트워크 초기화
        self.actor = PokerSACActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = PokerSACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = PokerSACCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # 타겟 네트워크 초기화
        self._hard_update(self.critic_target, self.critic)
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.actor.log_alpha], lr=lr)
        
        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 타겟 엔트로피 (자동 알파 조정)
        self.target_entropy = -action_dim
        
        # 학습 통계
        self.total_steps = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        
    def select_action(self, state: np.ndarray, evaluate: bool = False):
        """액션 선택"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            with torch.no_grad():
                action_logits = self.actor(state)
                action = torch.argmax(action_logits, dim=-1)
        else:
            action, _, _ = self.actor.sample_action(state)
            
        return action.cpu().numpy()[0]
    
    def update_parameters(self):
        """네트워크 파라미터 업데이트"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)
        
        # 현재 알파값
        alpha = torch.exp(self.actor.log_alpha)
        
        # Critic 업데이트
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample_action(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs.unsqueeze(1)
            target_q = rewards + self.gamma * q_next * (~dones)
            
        q1_current, q2_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q1_current, target_q) + F.mse_loss(q2_current, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 업데이트
        current_actions, current_log_probs, _ = self.actor.sample_action(states)
        q1_new, q2_new = self.critic(states, current_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha * current_log_probs.unsqueeze(1) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Alpha (온도 파라미터) 업데이트
        alpha_loss = -(self.actor.log_alpha * (current_log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 타겟 네트워크 소프트 업데이트
        self._soft_update(self.critic_target, self.critic)
        
        # 통계 저장
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.alpha_losses.append(alpha_loss.item())
        
        self.total_steps += 1
    
    def _soft_update(self, target, source):
        """타겟 네트워크 소프트 업데이트"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def _hard_update(self, target, source):
        """타겟 네트워크 하드 업데이트"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def train_self_play(
        self,
        num_episodes: int = 10000,
        max_steps_per_episode: int = 200,
        update_frequency: int = 4,
        eval_frequency: int = 100,
        save_frequency: int = 1000,
        save_path: str = 'models/sac_checkpoints'
    ):
        """자가 대전 훈련"""
        
        env = PokerMultiAgentWrapper(num_players=6)
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        episode_rewards = []
        episode_lengths = []
        win_rates = []
        
        for episode in tqdm(range(num_episodes), desc="SAC Training"):
            observations = env.reset()
            episode_reward = 0
            episode_length = 0
            wins = 0
            
            for step in range(max_steps_per_episode):
                if env.env.game_over:
                    break
                
                # 현재 활성 플레이어의 액션 선택
                active_player_key = f'player_{env.env.active_player}'
                state = observations[active_player_key]
                action = self.select_action(state)
                
                # 환경 스텝
                actions = {active_player_key: action}
                next_observations, rewards, dones, infos = env.step(actions)
                
                # 경험 저장 (활성 플레이어만)
                if active_player_key in rewards:
                    reward = rewards[active_player_key]
                    next_state = next_observations[active_player_key]
                    done = dones[active_player_key]
                    
                    self.replay_buffer.push(
                        state, action, reward, next_state, done
                    )
                    
                    episode_reward += reward
                    
                    # 승리 체크
                    if done and env.env.winner_id == env.env.active_player:
                        wins += 1
                
                observations = next_observations
                episode_length += 1
                
                # 네트워크 업데이트
                if step % update_frequency == 0:
                    self.update_parameters()
                
                if any(dones.values()):
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            win_rates.append(wins / max(episode_length, 1))
            
            # 평가 및 로깅
            if episode % eval_frequency == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                avg_win_rate = np.mean(win_rates[-100:])
                
                print(f"\nEpisode {episode}:")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  Average Length: {avg_length:.1f}")
                print(f"  Win Rate: {avg_win_rate:.2%}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")
                
                if len(self.actor_losses) > 0:
                    print(f"  Actor Loss: {np.mean(self.actor_losses[-100:]):.4f}")
                    print(f"  Critic Loss: {np.mean(self.critic_losses[-100:]):.4f}")
                    print(f"  Alpha: {torch.exp(self.actor.log_alpha).item():.4f}")
            
            # 모델 저장
            if episode % save_frequency == 0 and episode > 0:
                self.save_model(save_path / f'sac_checkpoint_{episode}.pt')
        
        # 최종 모델 저장
        self.save_model(save_path / 'sac_final.pt')
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'win_rates': win_rates
        }
    
    def save_model(self, path: Path):
        """모델 저장"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'alpha_losses': self.alpha_losses
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.alpha_losses = checkpoint['alpha_losses']
        
        print(f"Model loaded from {path}")
    
    def evaluate(self, num_games: int = 100) -> Dict:
        """모델 평가"""
        env = PokerMultiAgentWrapper(num_players=6)
        
        total_rewards = []
        wins = 0
        
        for game in range(num_games):
            observations = env.reset()
            episode_reward = 0
            
            while not env.env.game_over:
                active_player_key = f'player_{env.env.active_player}'
                state = observations[active_player_key]
                action = self.select_action(state, evaluate=True)
                
                actions = {active_player_key: action}
                observations, rewards, dones, infos = env.step(actions)
                
                if active_player_key in rewards:
                    episode_reward += rewards[active_player_key]
                
                if any(dones.values()):
                    if env.env.winner_id == 0:  # 첫 번째 플레이어가 우리 AI라고 가정
                        wins += 1
                    break
            
            total_rewards.append(episode_reward)
        
        return {
            'average_reward': np.mean(total_rewards),
            'win_rate': wins / num_games,
            'total_games': num_games
        }


if __name__ == "__main__":
    # SAC 트레이너 초기화 및 훈련
    trainer = PokerSACTrainer()
    
    # 훈련 실행
    print("Starting SAC Training...")
    results = trainer.train_self_play(num_episodes=1000, eval_frequency=50)
    
    # 최종 평가
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate(num_games=100)
    print(f"Average Reward: {eval_results['average_reward']:.3f}")
    print(f"Win Rate: {eval_results['win_rate']:.2%}")
    
    print("\nSAC Training Complete!")